"""Small CPU checks of semantics and the complete data -> training pipeline."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from kvtrain.config import ModelConfig, ROOT, PRESETS, local_path
from kvtrain.data import ByteTokenizer, PackedTokens, prepare, split_for
from kvtrain.model import AttentionKernel, Transformer
from kvtrain.precision import Precision
from kvtrain.train import main as train_main

torch.set_num_threads(2)


def small_config(**kwargs):
    return ModelConfig(vocab_size=64, **PRESETS["tiny"], **kwargs)


class Smoke(unittest.TestCase):
    def test_attention_modes_and_compressed_decode(self):
        tokens = torch.tensor([[1, 4, 7, 2, 8, 3, 6, 9]])
        for attention in ("gqa", "mla", "tpa"):
            for mode in ("standard", "cla", "cyclic"):
                for window in (0, 3):
                    with self.subTest(attention=attention, mode=mode, window=window):
                        torch.manual_seed(1)
                        cfg = small_config(attention=attention, mode=mode, window=window)
                        if mode == "cyclic":
                            cfg = replace(cfg, layers=6)  # Include a wrapped, overwritten sector.
                        model = Transformer(cfg)
                        model.eval()
                        logits = model(tokens)
                        torch.testing.assert_close(logits[:, :4], model(tokens[:, :4]), atol=2e-6, rtol=2e-5)
                        cache = model.new_cache(1, tokens.shape[1])
                        steps = [model.step(tokens[:, :1], cache)]
                        committed = [bank[:, :1].clone() for bank in cache.banks]
                        steps.extend(model.step(tokens[:, i:i + 1], cache) for i in range(1, tokens.shape[1]))
                        sequential = torch.cat(steps, 1)
                        for before, bank in zip(committed, cache.banks):
                            torch.testing.assert_close(before, bank[:, :1], rtol=0, atol=0)
                        torch.testing.assert_close(logits, sequential, atol=2e-6, rtol=2e-5)
                        self.assertEqual(cache.nbytes, tokens.numel() * model.layout.scalars_per_token * 4)
                        model.train()
                        loss = model(tokens[:, :-1], tokens[:, 1:], loss_chunk_size=3)
                        loss.backward()
                        for name, parameter in model.named_parameters():
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_mla_independent_rope_and_checkpoint_gradients(self):
        torch.manual_seed(4)
        cfg = small_config(attention="mla", mode="cyclic", share_rope=False)
        direct = Transformer(replace(cfg, activation_checkpoint=False))
        checkpointed = Transformer(cfg)
        checkpointed.load_state_dict(direct.state_dict())
        x, y = torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8))
        a, b = direct(x, y, 5), checkpointed(x, y, 5)
        a.backward()
        b.backward()
        torch.testing.assert_close(a, b)
        for p, q in zip(direct.parameters(), checkpointed.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-5)
        checkpointed.eval()
        cache = checkpointed.new_cache(2, 8)
        steps = torch.cat([checkpointed.step(x[:, i:i + 1], cache) for i in range(8)], 1)
        torch.testing.assert_close(checkpointed(x), steps, atol=2e-6, rtol=2e-5)
        expected = cfg.kv_latent_dim + (cfg.layers - 1) * (cfg.kv_latent_dim // 4) + cfg.layers * cfg.rope_dim
        self.assertEqual(checkpointed.layout.scalars_per_token, expected)

    def test_window_kernel_and_chunked_loss(self):
        torch.manual_seed(0)
        q, k, v = [torch.randn(1, 2, 7, 16) for _ in range(3)]
        scores = q @ k.transpose(-1, -2) / 4
        i = torch.arange(7)
        allowed = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < 3)
        oracle = scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v
        torch.testing.assert_close(AttentionKernel(3)(q, k, v), oracle, atol=2e-6, rtol=2e-5)
        model = Transformer(small_config())
        x, y = torch.randint(0, 64, (2, 7)), torch.randint(0, 64, (2, 7))
        logits = model(x)
        full_loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        torch.testing.assert_close(model(x, y, 3), full_loss)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        before = model(x, y).item()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            model(x, y).backward()
            optimizer.step()
        self.assertLess(model(x, y).item(), before)

    def test_fineweb_schema_pipeline_and_resume(self):
        (ROOT / "data").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "data", prefix="test-") as temporary:
            directory = Path(temporary)
            docs = [dict(id=f"id-{i}", text=f"Document {i}: education 科学 café. " * 3, score=4) for i in range(100)]
            docs.insert(0, dict(text="  "))
            closed = []

            def stream():
                try:
                    yield from docs
                finally:
                    closed.append(True)

            manifest = prepare(stream(), ByteTokenizer(), directory / "packed", train_tokens=512,
                               val_tokens=192, shard_tokens=37, val_fraction=0.25, encode_batch_size=3)
            self.assertEqual(closed, [True])
            self.assertEqual(manifest["documents"]["empty"], 1)
            for split, budget in (("train", 512), ("val", 192)):
                data = PackedTokens(directory / "packed", split, seq_len=24, verify=True)
                expected = []
                for doc in docs:
                    if doc["text"].strip() and split_for(doc["text"], 0.25, 42) == split:
                        expected.extend(doc["text"].encode("utf-8"))
                        expected.append(256)
                expected = expected[:budget]
                expected[-1] = 256
                np.testing.assert_array_equal(data.read(0, budget), expected)
                x, y = data.batch(3)
                torch.testing.assert_close(x[:, 1:], y[:, :-1])
                state = data.state_dict()
                expected_batch = data.batch(3)
                data.load_state_dict(state)
                for a, b in zip(expected_batch, data.batch(3)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            # The expected streams above are built by document text, independently
            # of document ids and regardless of token shard boundaries.
            args = ["--preset", "tiny", "--attention", "mla", "--mode", "cyclic", "--window", "3",
                    "--data", str(directory / "packed"), "--device", "cpu", "--steps", "2", "--seq-len", "16",
                    "--batch-size", "2", "--accumulation", "2", "--warmup", "0", "--eval-every", "1",
                    "--eval-batches", "1", "--save-every", "1", "--log-every", "1", "--loss-chunk-size", "7"]
            with redirect_stdout(io.StringIO()):
                train_main(args + ["--out", str(directory / "continuous")])
                train_main(args + ["--out", str(directory / "resumed"), "--stop-after", "1"])
                train_main(["--resume", str(directory / "resumed/last.pt"), "--device", "cpu"])
            load = lambda name: torch.load(directory / name / "last.pt", weights_only=True)
            continuous, resumed = load("continuous"), load("resumed")
            self.assertEqual(continuous["step"], resumed["step"])
            for name in continuous["model"]:
                torch.testing.assert_close(continuous["model"][name], resumed["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(continuous["train_sampler"]["generator"], resumed["train_sampler"]["generator"], rtol=0, atol=0)
            portable = torch.load(directory / "resumed/model.pt", weights_only=True)
            Transformer(ModelConfig(**portable["config"])).load_state_dict(portable["model"])
            shard = directory / "packed" / manifest["splits"]["train"]["shards"][0]["file"]
            with shard.open("r+b") as stream:
                stream.write(b"\xff\xff")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                PackedTokens(directory / "packed", seq_len=16, verify=True)
            with self.assertRaisesRegex(ValueError, "Source exhausted"):
                prepare(iter([dict(text="one")]), ByteTokenizer(), directory / "incomplete", train_tokens=100, val_tokens=100)
            self.assertFalse((directory / "incomplete/manifest.json").exists())

    def test_guards(self):
        with self.assertRaisesRegex(RuntimeError, "GPU"):
            Precision("delayed", "cpu")
        with self.assertRaises(ValueError):
            small_config(mode="cyclic", write_fraction="1/3")
        with self.assertRaises(ValueError):
            local_path(ROOT.parent / "data")

    def test_cpu_bf16(self):
        for attention in ("mla", "tpa"):
            model = Transformer(small_config(attention=attention, mode="cyclic"))
            tokens = torch.randint(0, 64, (2, 8))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(tokens[:, :-1], tokens[:, 1:], 4)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
