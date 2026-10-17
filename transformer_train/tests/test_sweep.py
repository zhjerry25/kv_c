from contextlib import redirect_stdout
from dataclasses import replace
from fractions import Fraction
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from kvtrain.config import ModelConfig, PRESETS, ROOT
from kvtrain.data import ByteTokenizer, prepare
from kvtrain.initialization import paired_initialize
from kvtrain.model import Transformer
from kvtrain.sweep import build_plan, parser, main as sweep_main, train_command
from kvtrain.train import main as train_main

torch.set_num_threads(2)


class SweepTests(unittest.TestCase):
    def test_paired_weights_and_publication_rows(self):
        for attention in ("gqa", "mla", "tpa"):
            cfg = ModelConfig(**PRESETS["tiny"], vocab_size=64, attention=attention)
            standard = Transformer(cfg)
            full = Transformer(replace(cfg, mode="cyclic", write_fraction="1"))
            paired_initialize(standard, 0)
            paired_initialize(full, 0)
            torch.testing.assert_close(full.initial.weight, standard.blocks[0].pub.weight, rtol=0, atol=0)
            full_private = dict(full.named_parameters())
            for name, weight in standard.named_parameters():
                if not name.endswith(".pub.weight"):
                    torch.testing.assert_close(weight, full_private[name], rtol=0, atol=0)
            for fraction in ("1/4", "1/2"):
                narrow = Transformer(replace(cfg, mode="cyclic", write_fraction=fraction))
                paired_initialize(narrow, 0)
                reference = dict(full.named_parameters())
                for name, weight in narrow.named_parameters():
                    if not name.endswith(".pub.weight"):
                        torch.testing.assert_close(weight, reference[name], rtol=0, atol=0)
                for layer in range(1, cfg.layers):
                    expected = full.blocks[layer].pub.weight[narrow.layout.indices[layer]]
                    torch.testing.assert_close(narrow.blocks[layer].pub.weight, expected, rtol=0, atol=0)
            changed = Transformer(replace(cfg, mode="cyclic"))
            paired_initialize(changed, 1)
            self.assertFalse(torch.equal(full.embed.weight, changed.embed.weight))

    def test_matrix_dimensions_and_runner_resume(self):
        plan = build_plan(parser().parse_args(["--dry-run"]))
        self.assertEqual(len(plan["runs"]), 24)
        self.assertEqual([row["mode"] for row in plan["runs"][:3]], ["standard"] * 3)
        self.assertEqual(plan["tokens_per_run"], 1_073_741_824)
        for row in plan["runs"]:
            if row["mode"] == "cyclic":
                f = float(Fraction(row["write_fraction"]))
                self.assertAlmostEqual(row["cache_ratio"], 0.1 + 0.9 * f)
                self.assertEqual(row["w"], row["C"] * f)
        (ROOT / "data").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "data", prefix="sweep-test-") as temporary:
            root = Path(temporary)
            docs = [dict(text=f"Education document {i}. " * 10) for i in range(100)]
            prepare(iter(docs), ByteTokenizer(), root / "packed", train_tokens=2048,
                    val_tokens=512, shard_tokens=127, val_fraction=0.25)
            argv = ["--preset", "tiny", "--attention", "gqa", "--fractions", "1/4", "--device", "cpu",
                    "--data", str(root / "packed"), "--out", str(root / "matrix"),
                    "--steps", "2", "--warmup", "0", "--seq-len", "16", "--batch-size", "1",
                    "--accumulation", "1", "--eval-every", "3", "--eval-batches", "1", "--save-every", "1", "--log-every", "1"]
            tiny_plan = build_plan(parser().parse_args(argv))
            case = tiny_plan["runs"][1]
            directory = root / "matrix" / case["name"]
            # Simulate an interrupted first attempt before launching the runner.
            directory.mkdir(parents=True)
            command = train_command(tiny_plan, case, directory)
            with redirect_stdout(io.StringIO()):
                train_main(command[3:] + ["--stop-after", "1"])
            (root / "matrix/matrix.json").write_text(json.dumps(tiny_plan))
            output = io.StringIO()
            with redirect_stdout(output):
                sweep_main(argv)
            self.assertIn("RESUME gqa-cyclic", output.getvalue())
            summary = json.loads((root / "matrix/summary.json").read_text())
            self.assertEqual(len(summary), 2)
            self.assertEqual(summary[0]["delta_nll_vs_standard"], 0)
            self.assertTrue(all(v["steps"] == 2 and v["train_tokens"] == 32 for v in summary))
            checkpoint = torch.load(directory / "last.pt", weights_only=True)
            self.assertTrue(checkpoint["settings"]["paired_init"])
            original_log = (directory / "metrics.jsonl").read_bytes()
            with redirect_stdout(io.StringIO()):
                sweep_main(argv)
            self.assertEqual(original_log, (directory / "metrics.jsonl").read_bytes())
            with self.assertRaisesRegex(ValueError, "plan/data changed"), redirect_stdout(io.StringIO()):
                sweep_main(argv + ["--seed", "1"])


if __name__ == "__main__":
    unittest.main()
