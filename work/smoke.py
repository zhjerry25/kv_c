"""Package smoke test: no training, no downloads, no research imports.

Loads one packaged standalone weight per width, checks batch-vs-cache
consistency on a short slice and full-validation 256 NLL against the recorded
audit values in outputs/clean_full_comparison.json. Finishes in about a minute.
"""
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cyclic_memory import CyclicMemory


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    recorded = json.loads((ROOT / 'outputs/clean_full_comparison.json').read_text())['summary']
    groups = {'none_w24': 'local_none24', 'none_w16': 'local_none16'}
    for tag, audit_group in groups.items():
        path = ROOT / f'weights/{tag}_s0.pt'
        model = CyclicMemory.load(path).eval().requires_grad_(False)
        meta = torch.load(path, map_location='cpu', weights_only=True)['metadata']
        alphabet = meta['alphabet']
        text = (ROOT / 'data/tinyshakespeare.txt').read_text()
        lookup = {ch: i for i, ch in enumerate(alphabet)}
        data = torch.tensor([lookup[ch] for ch in text[int(.9 * len(text)):]])
        tokens = data[:65][None]
        decoder = model.new_cache(65, 1, 7, torch.bfloat16)
        streamed = torch.cat([decoder.append(tokens[:, t]) for t in range(64)], 1)
        parallel = model.forward_stored(tokens)[:, :64]
        max_error = float((parallel - streamed).abs().max())
        assert max_error < .03, max_error
        total, count = 0., 0
        for start in range(0, len(data) - 1, 256):
            n = min(256, len(data) - start - 1)
            total += float(F.cross_entropy(model(data[start:start + n][None])[0],
                                           data[start + 1:start + n + 1], reduction='sum'))
            count += n
        nll = total / count
        expected = recorded[audit_group]['256']['seed_nll'][0]
        assert abs(nll - expected) < 1e-4, (tag, nll, expected)
        memory = model.new_cache(1024).memory_bytes()
        assert memory['publication_state'] == 0
        print(json.dumps(dict(group=tag, max_stream_error=max_error, nll=nll,
                              recorded_nll=expected, history_bytes_per_token=
                              memory['history'] // 1024)), flush=True)
    print(json.dumps(dict(status='smoke_ok')), flush=True)


if __name__ == '__main__':
    main()
