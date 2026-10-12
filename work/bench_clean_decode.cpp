// Complete trained-model native CPU decode; all layouts execute the same local
// model, with ordinary RoPE and actual stored precision. No private replay.
#define main clean_reader_test_main
#include "bench_clean_reader.cpp"
#undef main
#include <fstream>

struct BinaryReader {
    std::ifstream stream;
    explicit BinaryReader(const char *path):stream(path,std::ios::binary) {if(!stream)throw std::runtime_error("open fixture");}
    template<class T>void read(T *data,size_t n) {
        stream.read(reinterpret_cast<char*>(data),n*sizeof(T)); if(!stream)throw std::runtime_error("short fixture");
    }
    V floats(size_t n) {V result(n);read(result.data(),n);return result;}
};
struct PrivateWeights {V n1,n2,q,o,up,down,pub;};
struct DecodeWeights {
    int d,L,vocab,p,hq,hkv,width,context,steps,tile,storage,transport,block,anchors,total,c;
    float lam;
    V embed,initial,initialnorm,norm,output,frequency,expected;
    std::vector<PrivateWeights> layers;
    std::vector<V> bank_k,bank_v,dense_k,dense_v;
    std::vector<double> initial_sums;
    std::vector<int32_t> tokens,targets;
    explicit DecodeWeights(const char *path) {
        BinaryReader rd(path);int32_t header[16];rd.read(header,16);
        if(header[0]!=0x46434E32 || header[1]!=2)throw std::runtime_error("fixture version");
        d=header[2];L=header[3];vocab=header[4];p=header[5];hq=header[6];hkv=header[7];width=header[8];
        context=header[9];steps=header[10];tile=header[11];storage=header[12];transport=header[13];block=header[14];anchors=header[15];
        total=context+steps;c=p*hkv;rd.read(&lam,1);
        if(d<1 || L<1 || vocab<1 || p%2 || hq%hkv || width%4 || width>2*c || context<0 || steps<1 ||
            tile<1 || storage<0 || storage>1 || transport<0 || transport>4 || lam<0 || lam>=1 || anchors<1 || anchors>block ||
            (transport==3 && width%(4*hkv)))
            throw std::runtime_error("fixture dimensions");
        embed=rd.floats(size_t(vocab)*d);initial=rd.floats(size_t(2*c)*d);initialnorm=rd.floats(d);
        for(int l=0;l<L;++l) {
            PrivateWeights w;w.n1=rd.floats(d);w.n2=rd.floats(d);w.q=rd.floats(size_t(hq)*p*d);
            w.o=rd.floats(size_t(d)*hq*p);w.up=rd.floats(size_t(8*d)*d);w.down=rd.floats(size_t(d)*4*d);
            if(l)w.pub=rd.floats(size_t(width)*d);layers.push_back(std::move(w));
        }
        norm=rd.floats(d);output=rd.floats(size_t(vocab)*d);
        frequency=rd.floats(p/2);
        for(int l=0;l<L;++l) {int dim=l?width/2:c;bank_k.push_back(rd.floats(size_t(context)*dim));bank_v.push_back(rd.floats(size_t(context)*dim));}
        for(int l=0;l<L;++l) {dense_k.push_back(rd.floats(size_t(context)*c));dense_v.push_back(rd.floats(size_t(context)*c));}
        initial_sums.resize(size_t(L-1)*width);rd.read(initial_sums.data(),initial_sums.size());
        tokens.resize(steps);targets.resize(steps);rd.read(tokens.data(),steps);rd.read(targets.data(),steps);
        expected=rd.floats(size_t(steps)*vocab);
        char byte;if(rd.stream.read(&byte,1))throw std::runtime_error("trailing fixture");
    }
    void release_fixtures() {
        std::vector<V>().swap(bank_k);std::vector<V>().swap(bank_v);
        std::vector<V>().swap(dense_k);std::vector<V>().swap(dense_v);
    }
};
void decode_mv(const V &w,int rows,int cols,const float *x,float *y) {
    cblas_sgemv(CblasRowMajor,CblasNoTrans,rows,cols,1,w.data(),cols,x,1,0,y,1);
}
void decode_rms(const V &x,const V &gamma,V &out) {
    float sum=0;for(float value:x)sum+=value*value;
    float scale=1/std::sqrt(sum/x.size()+1e-5f);
    for(size_t j=0;j<x.size();++j)out[j]=x[j]*scale*gamma[j];
}
struct NativeDecoder {
    const DecodeWeights &w;int mode;
    Fixture f;Reader reader;
    V h,normalized,q,current,physical_k,physical_v,raw,publication,projected,up,gate,logits,cosine,sine;
    std::vector<double> sums;
    NativeDecoder(const DecodeWeights &weights,int which):w(weights),mode(which),
        f(w.total,w.L,w.p,w.hq,w.hkv,w.width,w.storage,w.tile),reader(f,mode),
        h(w.d),normalized(w.d),q(w.hq*w.p),current(2*w.c),physical_k(w.c),physical_v(w.c),raw(2*w.c),
        publication(w.width),projected(w.d),up(8*w.d),gate(4*w.d),logits(size_t(w.steps)*w.vocab),cosine(w.p/2),sine(w.p/2),
        sums((w.transport==1 || w.transport==2 || w.transport==4) ? w.initial_sums : std::vector<double>()) {
        if(w.transport==3 && mode) {
            // Split the base's immutable and mutable planes; no per-token holes
            // are loaded when only the fixed complement is needed at depth.
            int mutable_dim=w.width/2,immutable_dim=w.c-mutable_dim,q=mutable_dim/w.hkv;
            f.keys[0]=Plane(w.total,mutable_dim,w.storage);f.values[0]=Plane(w.total,mutable_dim,w.storage);
            f.keys.emplace_back(w.total,immutable_dim,w.storage);f.values.emplace_back(w.total,immutable_dim,w.storage);
            for(int l=0;l<w.L;++l)for(int head=0;head<w.hkv;++head) {
                f.map[l][head].clear();
                f.map[l][head].push_back({l,head*q,0,q});
                if(q<w.p)f.map[l][head].push_back({w.L,head*(w.p-q),q,w.p-q});
            }
        }
        if(mode==0) {
            for(int l=0;l<w.L;++l)for(size_t i=0;i<size_t(w.context)*w.c;++i) {
                f.dense_k[l].set(i,w.dense_k[l][i]);f.dense_v[l].set(i,w.dense_v[l][i]);
            }
            std::vector<Plane>().swap(f.keys);std::vector<Plane>().swap(f.values);
        } else {
            for(int l=0;l<w.L;++l)for(size_t i=0;i<w.bank_k[l].size();++i) {
                if(w.transport==3 && l==0) {
                    int t=int(i/w.c),global=int(i%w.c),head=global/w.p,j=global%w.p,q=w.width/2/w.hkv;
                    int bank=j<q?0:w.L,dim=j<q?w.width/2:w.c-w.width/2;
                    int field=j<q?head*q+j:head*(w.p-q)+j-q;
                    f.keys[bank].set(size_t(t)*dim+field,w.bank_k[l][i]);
                    f.values[bank].set(size_t(t)*dim+field,w.bank_v[l][i]);
                } else {f.keys[l].set(i,w.bank_k[l][i]);f.values[l].set(i,w.bank_v[l][i]);}
            }
            std::vector<Plane>().swap(f.dense_k);std::vector<Plane>().swap(f.dense_v);
        }
    }
    void reset() {f.n=w.context;if(w.transport==1 || w.transport==2 || w.transport==4)sums=w.initial_sums;}
    float rounded(float x) const {return w.storage?decode(encode(x)):x;}
    int coordinate(int layer,int j) const {
        if(!layer)return j;
        if(w.transport==3) {int q=w.width/2/w.hkv;return (j/q)*w.p+j%q;}
        return ((layer-1)*(w.width/2)+j)%w.c;
    }
    void store_plane(bool key,int layer,int position,int j,float value) {
        int bank=layer,dim=layer?w.width/2:w.c,field=j;
        if(w.transport==3 && layer==0) {
            int head=j/w.p,local=j%w.p,q=w.width/2/w.hkv;
            bank=local<q?0:w.L;dim=local<q?w.width/2:w.c-w.width/2;
            field=local<q?head*q+local:head*(w.p-q)+local-q;
        }
        (key?f.keys[bank]:f.values[bank]).set(size_t(position)*dim+field,value);
    }
    void store(int layer,int position,const float *source) {
        int dim=layer?w.width/2:w.c;
        for(int j=0;j<dim;j+=2) {
            int a=coordinate(layer,j),b=coordinate(layer,j+1),pair=(a%w.p)/2;
            float co=cosine[pair],si=sine[pair];
            float x=source[j],y=source[j+1];
            float ka=rounded(x*co-y*si),kb=rounded(x*si+y*co);
            physical_k[a]=ka;physical_k[b]=kb;
            float norm=co*co+si*si,ic=co/norm,is=-si/norm;
            current[a]=ka*ic-kb*is;current[b]=ka*is+kb*ic;
            if(mode) {store_plane(true,layer,position,j,ka);store_plane(true,layer,position,j+1,kb);}
        }
        for(int j=0;j<dim;++j) {
            int global=coordinate(layer,j);float value=rounded(source[dim+j]);
            current[w.c+global]=value;physical_v[global]=value;
            if(mode)store_plane(false,layer,position,j,value);
        }
        if(mode==0)for(int j=0;j<w.c;++j) {
            f.dense_k[layer].set(size_t(position)*w.c+j,physical_k[j]);
            f.dense_v[layer].set(size_t(position)*w.c+j,physical_v[j]);
        }
    }
    void rotate_query(int position) {
        for(int head=0;head<w.hq;++head)for(int j=0;j<w.p;j+=2) {
            float co=cosine[j/2],si=sine[j/2];
            float a=q[head*w.p+j],b=q[head*w.p+j+1];
            q[head*w.p+j]=a*co-b*si;q[head*w.p+j+1]=a*si+b*co;
        }
    }
    void run() {
        for(int step=0;step<w.steps;++step) {
            int pos=w.context+step;f.n=pos+1;
            for(int j=0;j<w.p/2;++j) {
                float phase=float(pos)*w.frequency[j];cosine[j]=std::cos(phase);sine[j]=std::sin(phase);
            }
            std::copy_n(w.embed.data()+size_t(w.tokens[step])*w.d,w.d,h.data());
            decode_rms(h,w.initialnorm,normalized);decode_mv(w.initial,2*w.c,w.d,normalized.data(),raw.data());store(0,pos,raw.data());
            if(w.transport==2 && pos%w.block==0)std::fill(sums.begin(),sums.end(),0);
            for(int l=0;l<w.L;++l) {
                const PrivateWeights &layer=w.layers[l];decode_rms(h,layer.n1,normalized);
                if(l) {
                    decode_mv(layer.pub,w.width,w.d,normalized.data(),publication.data());
                    if(w.transport!=4)for(int j=0;j<w.width/2;++j) {
                        int global=coordinate(l,j);publication[j]+=current[global];publication[w.width/2+j]+=current[w.c+global];
                    }
                    double *sum=(w.transport==1 || w.transport==2 || w.transport==4)?sums.data()+size_t(l-1)*w.width:nullptr;
                    if(w.transport==4)for(int j=0;j<w.width;++j) {
                        sum[j]+=double(publication[j]);
                        int half=w.width/2,global=coordinate(l,j%half)+(j>=half?w.c:0);
                        publication[j]=current[global]+(1-w.lam)*publication[j]+w.lam*float(sum[j]/double(pos+1));
                    }
                    else if(w.transport==1)for(int j=0;j<w.width;++j) {
                        sum[j]+=double(publication[j]);publication[j]=(1-w.lam)*publication[j]+w.lam*float(sum[j]/double(pos+1));
                    }
                    else if(w.transport==2 && pos%w.block>=w.anchors)for(int j=0;j<w.width;++j)
                        publication[j]=(1-w.lam)*publication[j]+w.lam*float(sum[j]/double(w.anchors));
                    store(l,pos,publication.data());
                    if(w.transport==2 && pos%w.block<w.anchors)for(int j=0;j<w.width/2;++j) {
                        int global=coordinate(l,j);sum[j]+=current[global];sum[w.width/2+j]+=current[w.c+global];
                    }
                }
                decode_mv(layer.q,w.hq*w.p,w.d,normalized.data(),q.data());rotate_query(pos);
                std::copy(q.begin(),q.end(),f.queries.begin()+size_t(l)*w.hq*w.p);reader.run(l);
                decode_mv(layer.o,w.d,w.hq*w.p,reader.out.data(),projected.data());
                for(int j=0;j<w.d;++j)h[j]+=projected[j];
                decode_rms(h,layer.n2,normalized);decode_mv(layer.up,8*w.d,w.d,normalized.data(),up.data());
                for(int j=0;j<4*w.d;++j) {float x=up[j];gate[j]=x/(1+std::exp(-x))*up[4*w.d+j];}
                decode_mv(layer.down,w.d,4*w.d,gate.data(),projected.data());
                for(int j=0;j<w.d;++j)h[j]+=projected[j];
            }
            decode_rms(h,w.norm,normalized);decode_mv(w.output,w.vocab,w.d,normalized.data(),logits.data()+size_t(step)*w.vocab);
        }
    }
    size_t private_scratch() const {
        return 4*(h.size()+normalized.size()+q.size()+current.size()+physical_k.size()+physical_v.size()+raw.size()+
            publication.size()+projected.size()+up.size()+gate.size()+logits.size()+f.queries.size())+reader.scratch();
    }
};
double native_nll(const V &logits,const DecodeWeights &w) {
    double total=0;
    for(int t=0;t<w.steps;++t) {
        const float *row=logits.data()+size_t(t)*w.vocab;double maximum=*std::max_element(row,row+w.vocab),mass=0;
        for(int j=0;j<w.vocab;++j)mass+=std::exp(double(row[j])-maximum);
        total+=std::log(mass)+maximum-row[w.targets[t]];
    } return total/w.steps;
}
int main(int argc,char **argv) {try {
    if(argc!=3)throw std::runtime_error("fixture repeats");
    DecodeWeights w(argv[1]);int repeats=std::stoi(argv[2]);
    NativeDecoder material(w,0),gather(w,1),segmented(w,2);
    std::vector<NativeDecoder*> decoders={&material,&gather,&segmented};w.release_fixtures();
    V errors(3),deltas(3),pair_errors(3);
    double reference_nll=native_nll(w.expected,w);
    for(int mode=0;mode<3;++mode) {
        decoders[mode]->reset();decoders[mode]->run();errors[mode]=error(decoders[mode]->logits,w.expected);
        pair_errors[mode]=error(decoders[mode]->logits,material.logits);
        deltas[mode]=float(native_nll(decoders[mode]->logits,w)-reference_nll);
        if(errors[mode]>(w.storage?.03f:5e-4f) || std::abs(deltas[mode])>(w.storage?.001f:5e-5f))throw std::runtime_error("native trained-logit oracle mismatch");
    }
    std::vector<std::vector<double>> timings(3);
    for(int r=0;r<repeats;++r)for(int order=0;order<3;++order) {
        int mode=(r+order)%3;NativeDecoder &decoder=*decoders[mode];decoder.reset();auto start=Clock::now();decoder.run();
        timings[mode].push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());reader_sink=decoder.logits[r%decoder.logits.size()];
    }
    std::cout<<std::setprecision(10)<<"{\"context\":"<<w.context<<",\"new_tokens\":"<<w.steps<<",\"depth\":"<<w.L
        <<",\"d\":"<<w.d<<",\"head_dim\":"<<w.p<<",\"query_heads\":"<<w.hq<<",\"kv_heads\":"<<w.hkv<<",\"width\":"<<w.width
        <<",\"dtype\":\""<<(w.storage?"bf16":"fp32")<<"\",\"transport\":"<<w.transport<<",\"repeats\":"<<repeats
        <<",\"material_history_bytes\":"<<material.f.history(true)<<",\"compact_history_bytes\":"<<gather.f.history(false)
        <<",\"publication_state_bytes\":"<<8*gather.sums.size()<<",\"max_logits_error_vs_torch\":["<<errors[0]<<","<<errors[1]<<","<<errors[2]
        <<"],\"max_logits_error_vs_material\":["<<pair_errors[0]<<","<<pair_errors[1]<<","<<pair_errors[2]
        <<"],\"delta_nll_vs_torch\":["<<deltas[0]<<","<<deltas[1]<<","<<deltas[2]
        <<"],\"rotary_state_bytes\":"<<4*(w.frequency.size()+gather.cosine.size()+gather.sine.size())
        <<",\"historical_rotary_bytes\":0,\"scratch_bytes\":["<<material.private_scratch()<<","<<gather.private_scratch()<<","<<segmented.private_scratch()<<"],\"milliseconds\":[";
    for(int mode=0;mode<3;++mode) {if(mode)std::cout<<",";std::cout<<(repeats?median(timings[mode]):0);}
    std::cout<<"]}\n";return 0;
} catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 2;} }
