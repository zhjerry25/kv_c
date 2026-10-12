"""Small, reproducible CPU language-model screen. No pretrained weights.

Run with bundled Python and PYTHONPATH=work/torch_deps. The common backbone is
pre-RMSNorm, ordinary full causal GQA, full SwiGLU, and ordinary RoPE.
Outputs are evidence about this scale, not a language-model scaling claim.
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

class RMS(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.weight

def rope(x):
    n, p = x.shape[-2:]
    freq = 10000.0 ** (-torch.arange(0, p, 2, device=x.device).float() / p)
    phase = torch.arange(n, device=x.device).float()[:, None] * freq[None, :]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * phase.cos() - b * phase.sin(), a * phase.sin() + b * phase.cos()), -1).flatten(-2)

class Block(nn.Module):
    def __init__(self, d, heads, p, c, rank, mode, source_norm):
        super().__init__(); self.heads=heads; self.p=p; self.mode=mode
        self.n1=RMS(d); self.n2=RMS(d)
        self.q=nn.Linear(d, heads*p, bias=False); self.o=nn.Linear(heads*p,d,bias=False)
        self.up=nn.Linear(d, 2*4*d, bias=False); self.down=nn.Linear(4*d,d,bias=False)
        self.mn=RMS(c) if source_norm else nn.Identity()
        self.kv=nn.Linear(d,p if mode in ('vshared','vlatest') else c,bias=False) if mode in ('standard','cla2','vshared','vlatest','pair_cayley') else None
        self.pub=nn.Linear(d,rank,bias=False) if mode not in ('standard','cla2','shared','vshared') else None
        self.expand=nn.Linear(rank,p if mode=='vlatest' else c,bias=False) if self.pub is not None and mode not in ('isometric','isometric_tied','isometric_cayley','pair_cayley') else None
        if mode=='isometric': self.frequency=nn.Parameter(torch.ones(c//3))
        if mode=='isometric_tied': self.register_buffer('frequency',torch.ones(c//3))
        if mode=='isometric_cayley': self.register_buffer('frequency',torch.ones(c//3))
        if mode=='pair_cayley': self.register_buffer('frequency',torch.ones(c//3))
        if mode=='nonlinear':
            self.scale=nn.Parameter(torch.zeros(c)); self.gain=nn.Parameter(torch.ones(c)); self.bias=nn.Parameter(torch.zeros(c))
        if mode=='moments':
            self.xi=nn.Linear(rank,4,bias=False); self.eta=nn.Linear(rank,4,bias=False)
    def source(self,h,m,index,total):
        a=None
        if self.mode=='latent':
            a=self.pub(self.n1(h));m=self.expand(a)
        elif self.mode in ('vshared','vlatest'):
            value=m[...,self.p:]
            if index>0 and self.pub is not None:
                a=self.pub(self.n1(h)); value=value+self.expand(a)
            m=torch.cat((self.kv(self.n1(h)),value),-1)
        elif self.mode=='standard' or (self.mode in ('cla2','pair_cayley') and index%2==0):
            m=self.kv(self.n1(h))
        elif index>0 and self.pub is not None:
            a=self.pub(self.n1(h))
            if self.mode in ('isometric','isometric_tied','isometric_cayley','pair_cayley'):
                shift=(index-1)%3
                rolled=m.roll(shift,-1); count=m.shape[-1]//3
                triples=rolled[...,:count*3].view(*m.shape[:-1],count,3)
                group=torch.arange(count)%a.shape[-1]
                if self.mode in ('isometric_cayley','pair_cayley'):
                    inverse=torch.reciprocal(1+a.square())
                    cosine=((1-a.square())*inverse)[...,group]
                    sine=(2*a*inverse)[...,group]
                elif self.mode=='isometric_tied':
                    cosine=a.cos()[...,group]; sine=a.sin()[...,group]
                else:
                    theta=a[...,group]*self.frequency
                    cosine=theta.cos(); sine=theta.sin()
                mean=triples.mean(-1,keepdim=True)
                x,y,z=triples.unbind(-1)
                cross=torch.stack((z-y,x-z,y-x),-1)/math.sqrt(3)
                moved=mean+(triples-mean)*cosine[...,None]+cross*sine[...,None]
                if self.mode=='pair_cayley': moved=triples+(triples-mean)*(cosine[...,None]-1)+cross*sine[...,None]
                rolled=torch.cat((moved.flatten(-2),rolled[...,count*3:]),-1)
                m=rolled.roll(-shift,-1)
            else: m=m+self.expand(a)
            if self.mode=='nonlinear':
                m=m+self.scale*torch.tanh(self.gain*m+self.bias)
            if self.mode=='moments':
                xi,eta=self.xi(a),self.eta(a)
                # Per-feature common rescaling cancels exactly in all prefix ratios.
                weights=torch.exp(eta-eta.detach().amax(1,keepdim=True))
                mass=weights.cumsum(1)
                sums=(weights[:,:,:,None]*m[:,:,None,:]).cumsum(1)
                means=sums/mass[:,:,:,None]
                logmass=mass.log()+eta.detach().amax(1,keepdim=True)
                mix=(xi+logmass).softmax(-1)
                pooled=(mix[:,:,:,None]*means).sum(-2)
                lam=getattr(self,'mix_budget',0.25)/(total-1)
                m=(1-lam)*m+lam*pooled
            if self.mode=='moments_uniform':
                count=torch.arange(1,m.shape[1]+1,device=m.device,dtype=m.dtype)[None,:,None]
                pooled=m.cumsum(1)/count
                lam=getattr(self,'mix_budget',0.25)/(total-1)
                m=(1-lam)*m+lam*pooled
        return m,a
    def forward(self,h,m):
        b,n,_=h.shape
        source=self.mn(m); k,v=source.split(self.p,-1)
        q=rope(self.q(self.n1(h)).view(b,n,self.heads,self.p).transpose(1,2))
        k=rope(k[:,None,:,:]).expand(-1,self.heads,-1,-1)
        v=v[:,None,:,:].expand(-1,self.heads,-1,-1)
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        h=h+self.o(out.transpose(1,2).reshape(b,n,-1))
        a,b=self.up(self.n2(h)).chunk(2,-1)
        return h+self.down(F.silu(a)*b)

class Model(nn.Module):
    def __init__(self,vocab,mode,rank=16,d=128,depth=4,source_norm=False):
        super().__init__(); self.mode=mode; self.rank=rank; self.depth=depth; self.c=64
        self.embed=nn.Embedding(vocab,d)
        self.initial=nn.Linear(d,32 if mode in ('vshared','vlatest') else self.c,bias=False) if mode not in ('standard','cla2','pair_cayley','latent') else None
        self.initial_norm=RMS(d)
        self.blocks=nn.ModuleList([Block(d,4,32,self.c,rank,mode,source_norm) for _ in range(depth)])
        self.norm=RMS(d); self.output=nn.Linear(d,vocab,bias=False)
        self.apply(self.init)
        # Equal residual branch scale for all variants.
        for block in self.blocks:
            nn.init.normal_(block.o.weight,std=0.02/math.sqrt(2*depth))
            nn.init.normal_(block.down.weight,std=0.02/math.sqrt(2*depth))
        # Block 0 uses the initial source: remove unused publication parameters.
        if self.initial is not None:
            self.blocks[0].pub=None; self.blocks[0].expand=None
            if mode=='moments': self.blocks[0].xi=None; self.blocks[0].eta=None
            if mode=='nonlinear':
                self.blocks[0].register_parameter('scale',None)
                self.blocks[0].register_parameter('gain',None)
                self.blocks[0].register_parameter('bias',None)
            if mode=='isometric': self.blocks[0].register_parameter('frequency',None)
            if mode=='isometric_tied': self.blocks[0].frequency=None
            if mode=='isometric_cayley': self.blocks[0].frequency=None
        if mode=='cla2':
            for i in range(1,depth,2): self.blocks[i].kv=None
        if mode=='pair_cayley':
            for i,block in enumerate(self.blocks):
                if i%2: block.kv=None
                else: block.pub=None;block.frequency=None
        if self.initial is None: self.initial_norm=nn.Identity()
    @staticmethod
    def init(module):
        if isinstance(module,(nn.Linear,nn.Embedding)): nn.init.normal_(module.weight,std=.02)
    def forward(self,x,states=False):
        h=self.embed(x); m=self.initial(self.initial_norm(h)) if self.initial else None
        if self.mode in ('vshared','vlatest'): m=torch.cat((torch.zeros_like(m),m),-1)
        initial_m=m
        sources=[]; controls=[]
        for i,block in enumerate(self.blocks):
            if self.mode in ('vshared','vlatest'): m=initial_m
            m,a=block.source(h,m,i,self.depth)
            if states: sources.append(m); controls.append(a)
            h=block(h,m)
        logits=self.output(self.norm(h))
        return (logits,sources,controls) if states else logits
    def cache_scalars(self):
        if self.mode=='standard': return self.depth*self.c
        if self.mode=='cla2': return math.ceil(self.depth/2)*self.c
        if self.mode=='pair_cayley': return math.ceil(self.depth/2)*self.c+(self.depth//2)*self.rank
        if self.mode=='shared': return self.c
        if self.mode=='latent': return self.depth*self.rank
        if self.mode=='vshared': return self.depth*32+32
        if self.mode=='vlatest': return self.depth*32+32+(self.depth-1)*self.rank
        return self.c+(self.depth-1)*self.rank

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modes',default='standard,cla2,shared,linear,nonlinear,moments')
    parser.add_argument('--ranks',default='16'); parser.add_argument('--seeds',default='0')
    parser.add_argument('--steps',type=int,default=600); parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--length',type=int,default=128); parser.add_argument('--d',type=int,default=128)
    parser.add_argument('--depth',type=int,default=4); parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--source-norm',action='store_true')
    parser.add_argument('--zero-public',action='store_true',help='Initialize publication at the exact shared-source subfamily.')
    parser.add_argument('--match-public-init',action='store_true',help='Copy P/U from the same seed-specific linear reference; isolates extra public operations.')
    parser.add_argument('--factorized-reference-init',action='store_true',help='Initialize independent latent KV from a balanced truncated SVD of each standard projection, with matched Frobenius norm.')
    parser.add_argument('--tag',default='screen'); parser.add_argument('--save',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(args.threads)
    data_path=ROOT/'data/tinyshakespeare.txt'; raw=data_path.read_bytes(); text=raw.decode()
    alphabet=sorted(set(text)); stoi={s:i for i,s in enumerate(alphabet)}
    data=torch.tensor([stoi[s] for s in text],dtype=torch.long); split=int(len(data)*.9)
    train,val=data[:split],data[split:]
    out=ROOT/'outputs'/f'{args.tag}_training.json'
    results=[]
    manifest={'args':vars(args),'torch':torch.__version__,'device':'cpu','data_sha256':hashlib.sha256(raw).hexdigest(),
              'train_chars':len(train),'validation_chars':len(val),'vocab':len(alphabet),'validation_batches':20,
              'source_norm':args.source_norm,'note':'Chronological 90/10 split; validation random windows fixed independently of training; no dropout. Same token budgets, not parameter-matched. Common backbone tensors and initial KV are copied from the same seed-specific reference.'}
    def batch(part,generator):
        idx=torch.randint(len(part)-args.length-1,(args.batch,),generator=generator)
        offsets=torch.arange(args.length)
        return part[idx[:,None]+offsets],part[idx[:,None]+offsets+1]
    for seed in map(int,args.seeds.split(',')):
      for mode in args.modes.split(','):
       ranks=list(map(int,args.ranks.split(','))) if mode in ('linear','nonlinear','moments','moments_uniform','latent','vlatest','isometric','isometric_tied','isometric_cayley','pair_cayley') else [0]
       for rank in ranks:
        torch.manual_seed(seed)
        reference=Model(len(alphabet),'standard',0,args.d,args.depth,args.source_norm)
        torch.manual_seed(100000+seed)
        model=Model(len(alphabet),mode,rank,args.d,args.depth,args.source_norm)
        common=reference.state_dict(); own=model.state_dict()
        model.load_state_dict({k:v for k,v in common.items() if k in own and own[k].shape==v.shape},strict=False)
        if model.initial is not None:
            model.initial.weight.data.copy_(reference.blocks[0].kv.weight[32:] if mode in ('vshared','vlatest') else reference.blocks[0].kv.weight)
            model.initial_norm.weight.data.copy_(reference.blocks[0].n1.weight)
        if mode in ('vshared','vlatest'):
            for block,refblock in zip(model.blocks,reference.blocks): block.kv.weight.data.copy_(refblock.kv.weight[:32])
        if args.match_public_init:
            torch.manual_seed(100000+seed)
            public_reference=Model(len(alphabet),'linear',rank,args.d,args.depth,args.source_norm)
            with torch.no_grad():
                for block,refblock in zip(model.blocks,public_reference.blocks):
                    if block.pub is not None: block.pub.weight.copy_(refblock.pub.weight)
                    if block.expand is not None and block.expand.weight.shape==refblock.expand.weight.shape:
                        block.expand.weight.copy_(refblock.expand.weight)
            del public_reference
        if args.factorized_reference_init:
            if mode!='latent': raise ValueError('Factorized reference initialization is specific to the independent latent baseline.')
            with torch.no_grad():
                for block,refblock in zip(model.blocks,reference.blocks):
                    left,singular,right=torch.linalg.svd(refblock.kv.weight,full_matrices=False)
                    scale=refblock.kv.weight.norm()/singular[:rank].norm()
                    root=(scale*singular[:rank]).sqrt()
                    block.expand.weight.copy_(left[:,:rank]*root[None,:])
                    block.pub.weight.copy_(root[:,None]*right[:rank,:])
        if args.zero_public:
            for block in model.blocks:
                if block.pub is not None: nn.init.zeros_(block.pub.weight)
        del reference
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
        gen=torch.Generator().manual_seed(10000+seed)
        curves=[]; grad_max=0.; grad_clips=0; start=time.monotonic()
        for step in range(args.steps):
            model.train(); x,y=batch(train,gen); logits=model(x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
            if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss {mode} {step}')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.0))
            if not math.isfinite(grad): raise RuntimeError(f'Non-finite gradient {mode} {step}')
            grad_max=max(grad_max,grad); grad_clips+=grad>1
            lr=3e-4*min((step+1)/30,1)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.step()
            if step%100==0 or step==args.steps-1:
                curves.append({'step':step+1,'train_loss':float(loss.detach()),'gradient_norm_before_clip':grad})
                print(json.dumps({'mode':mode,'rank':rank,'seed':seed,**curves[-1],'elapsed':round(time.monotonic()-start,2)}),flush=True)
        model.eval(); vg=torch.Generator().manual_seed(987654)
        losses=[]; acc=[]
        with torch.no_grad():
            for _ in range(20):
                x,y=batch(val,vg); logits=model(x)
                losses.append(float(F.cross_entropy(logits.flatten(0,1),y.flatten())))
                acc.append(float((logits.argmax(-1)==y).float().mean()))
            _,sources,controls=model(x,states=True)
            source_rms=[float(m.square().mean().sqrt()) for m in sources]
            control_rms=[float(a.square().mean().sqrt()) if a is not None else None for a in controls]
        row={'mode':mode,'rank':rank,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
             'cache_scalars_per_token':model.cache_scalars(),'validation_loss':float(np.mean(losses)),
             'validation_accuracy':float(np.mean(acc)),'gradient_max':grad_max,'gradient_clipping_fraction':grad_clips/args.steps,
             'source_rms':source_rms,'control_rms':control_rms,'elapsed_seconds':time.monotonic()-start,'curve':curves}
        results.append(row); out.write_text(json.dumps({'manifest':manifest,'results':results},indent=2)+'\n')
        if args.save:
            torch.save({'state_dict':model.state_dict(),'row':row,'args':vars(args),'alphabet':alphabet},ROOT/'work'/f'{args.tag}_{mode}_r{rank}_s{seed}.pt')
        print('RESULT '+json.dumps({k:v for k,v in row.items() if k!='curve'}),flush=True)

if __name__=='__main__': main()
