"""FineWeb-Edu streaming -> EOS packed shards -> reproducible mmap batches."""

import argparse
from bisect import bisect_right
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from .config import local_path


def configure_hf_cache():
    # Set these before importing Hub/Datasets: their constants are cached on import.
    os.environ["HF_HOME"] = str(local_path("data/hf_cache"))
    os.environ["HF_HUB_CACHE"] = str(local_path("data/hf_cache/hub"))
    os.environ["HF_DATASETS_CACHE"] = str(local_path("data/hf_cache/datasets"))


def split_for(text, val_fraction, seed):
    # Text hashing sends duplicate documents to the same split, regardless of id.
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=str(seed).encode()[:16]).digest()
    return "val" if int.from_bytes(digest, "big") / 2**64 < val_fraction else "train"


class ByteTokenizer:
    """Offline pipeline tests only. Real runs default to the GPT-2 BPE tokenizer."""
    vocab_size, eos_id = 257, 256

    def encode_batch(self, texts):
        return [list(text.encode("utf-8")) for text in texts]

    def metadata(self):
        return dict(name="byte", vocab_size=self.vocab_size, eos_id=self.eos_id)


class BPETokenizer:
    def __init__(self, name="gpt2", revision="main", tokenizer_file=None):
        configure_hf_cache()
        from tokenizers import Tokenizer
        if tokenizer_file:
            self.path = local_path(tokenizer_file)
        else:
            from huggingface_hub import hf_hub_download
            self.path = Path(hf_hub_download(name, "tokenizer.json", revision=revision,
                                            cache_dir=str(local_path("data/hf_cache/hub"))))
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.name, self.revision = name, revision
        self.eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        if self.eos_id is None:
            raise ValueError("Tokenizer must define <|endoftext|> as an EOS token")
        self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)

    def encode_batch(self, texts):
        return [encoding.ids for encoding in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def metadata(self):
        return dict(name=self.name, revision=self.revision, vocab_size=self.vocab_size,
                    eos_id=self.eos_id, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())


class ShardWriter:
    def __init__(self, directory, split, shard_tokens, dtype):
        self.directory, self.split, self.shard_tokens = directory, split, shard_tokens
        self.dtype, self.shards = np.dtype(dtype), []
        self.file, self.count, self.total, self.digest = None, 0, 0, None

    def append(self, ids):
        while ids:
            if self.file is None:
                self.path = self.directory / f"{self.split}-{len(self.shards):05d}.bin"
                self.file = self.path.open("xb")
                self.count, self.digest = 0, hashlib.sha256()
            n = min(len(ids), self.shard_tokens - self.count)
            payload = np.asarray(ids[:n], dtype=self.dtype).tobytes()
            self.file.write(payload)
            self.digest.update(payload)
            self.count += n
            self.total += n
            ids = ids[n:]
            if self.count == self.shard_tokens:
                self.close()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.shards.append(dict(file=self.path.name, tokens=self.count, sha256=self.digest.hexdigest()))
            self.file = None


def prepare(documents, tokenizer, output, *, train_tokens, val_tokens, shard_tokens=50_000_000,
            val_fraction=0.01, seed=42, encode_batch_size=128, source=None):
    if min(train_tokens, val_tokens, shard_tokens, encode_batch_size) < 1 or not 0 < val_fraction < 1:
        raise ValueError("Positive token budgets/batch size and 0 < val_fraction < 1 required")
    output = local_path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty; use a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dtype = "<u2" if tokenizer.vocab_size <= 65536 else "<u4"
    budgets = dict(train=train_tokens, val=val_tokens)
    writers = {split: ShardWriter(output, split, shard_tokens, dtype) for split in budgets}
    counts = dict(train=0, val=0, empty=0, scanned=0)
    pending = []

    def flush():
        if not pending:
            return
        for (split, text), ids in zip(pending, tokenizer.encode_batch([text for _, text in pending])):
            writer = writers[split]
            remaining = budgets[split] - writer.total
            if not remaining:
                continue
            ids = ids + [tokenizer.eos_id]
            if min(ids) < 0 or max(ids) >= tokenizer.vocab_size:
                raise ValueError("Tokenizer emitted out-of-range IDs")
            ids = ids[:remaining]
            # A budget-truncated final document still ends with EOS.
            ids[-1] = tokenizer.eos_id
            writer.append(ids)
            counts[split] += 1
        pending.clear()

    try:
        for document in documents:
            counts["scanned"] += 1
            text = document.get("text")
            if not isinstance(text, str):
                raise ValueError("Every document needs a string 'text' field")
            if not text.strip():
                counts["empty"] += 1
                continue
            split = split_for(text, val_fraction, seed)
            if writers[split].total < budgets[split]:
                pending.append((split, text))
            if len(pending) >= encode_batch_size:
                flush()
            if all(writers[s].total >= budgets[s] for s in budgets):
                break
        flush()
    finally:
        for writer in writers.values():
            writer.close()
        close = getattr(documents, "close", None)
        if close is not None:
            close()
        # Release cyclic Arrow scanner/file objects while its thread pool and
        # Python are alive, rather than leaving them for native process teardown.
        gc.collect()
    if any(writers[s].total != budgets[s] for s in budgets):
        raise ValueError(f"Source exhausted before budgets were met: { {s: w.total for s, w in writers.items()} }. No manifest published; use smaller budgets or a new output directory.")
    if isinstance(tokenizer, BPETokenizer):
        shutil.copyfile(tokenizer.path, output / "tokenizer.json")
    manifest = dict(version=1, source=source or {}, tokenizer=tokenizer.metadata(), dtype=dtype,
                    seed=seed, val_fraction=val_fraction, documents=counts,
                    splits={s: dict(tokens=w.total, shards=w.shards) for s, w in writers.items()})
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode()
    temporary = output / "manifest.json.tmp"
    temporary.write_bytes(payload)
    temporary.replace(output / "manifest.json")
    return manifest


class PackedTokens:
    def __init__(self, directory, split="train", seq_len=2048, seed=42, verify=False):
        directory = local_path(directory)
        payload = (directory / "manifest.json").read_bytes()
        self.fingerprint = hashlib.sha256(payload).hexdigest()
        self.manifest = json.loads(payload)
        self.seq_len = seq_len
        self.generator = torch.Generator().manual_seed(seed)
        self.arrays, self.ends = [], []
        total = 0
        for shard in self.manifest["splits"][split]["shards"]:
            path = (directory / shard["file"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Shard path escapes dataset directory")
            if path.stat().st_size != shard["tokens"] * np.dtype(self.manifest["dtype"]).itemsize:
                raise ValueError(f"Truncated or incorrect shard: {path}")
            if verify:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != shard["sha256"]:
                    raise ValueError(f"Checksum mismatch: {path}")
            self.arrays.append(np.memmap(path, dtype=self.manifest["dtype"], mode="r"))
            total += shard["tokens"]
            self.ends.append(total)
        self.total = total
        if total != self.manifest["splits"][split]["tokens"] or seq_len < 1 or total <= seq_len:
            raise ValueError("Inconsistent manifest or not enough tokens for seq_len + 1")

    def read(self, start, count):
        if start < 0 or count < 1 or start + count > self.total:
            raise ValueError("Token slice outside dataset")
        output = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            index = bisect_right(self.ends, start)
            previous = self.ends[index - 1] if index else 0
            n = min(count - filled, self.ends[index] - start)
            output[filled:filled + n] = self.arrays[index][start - previous:start - previous + n]
            filled += n
            start += n
        return output

    def batch(self, batch_size, device="cpu"):
        starts = torch.randint(self.total - self.seq_len, (batch_size,), generator=self.generator).tolist()
        ids = torch.from_numpy(np.stack([self.read(start, self.seq_len + 1) for start in starts]))
        if torch.device(device).type == "cuda":
            ids = ids.pin_memory().to(device, non_blocking=True)
        else:
            ids = ids.to(device)
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()

    def state_dict(self):
        return dict(generator=self.generator.get_state(), fingerprint=self.fingerprint, seq_len=self.seq_len)

    def load_state_dict(self, state):
        if state["fingerprint"] != self.fingerprint or state["seq_len"] != self.seq_len:
            raise ValueError("Resume dataset or sequence length differs from checkpoint")
        self.generator.set_state(state["generator"])


def fineweb_documents(subset="sample-10BT", revision="main", seed=42, shuffle_buffer=10000, max_docs=None, source=None):
    # Keep Hugging Face caches within the standalone framework as well.
    configure_hf_cache()
    from huggingface_hub import HfApi
    from datasets import load_dataset
    resolved = HfApi().dataset_info("HuggingFaceFW/fineweb-edu", revision=revision).sha
    if source is not None:
        source["resolved_revision"] = resolved
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name=subset, revision=resolved,
                          split="train", streaming=True, cache_dir=os.environ["HF_DATASETS_CACHE"])
    if shuffle_buffer:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return iter(stream.take(max_docs) if max_docs is not None else stream)


def local_documents(path):
    with local_path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--tokenizer-file")
    parser.add_argument("--train-tokens", type=int, default=10_000_000)
    parser.add_argument("--val-tokens", type=int, default=100_000)
    parser.add_argument("--shard-tokens", type=int, default=50_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-jsonl", help="Offline documents with the same FineWeb-Edu text schema")
    parser.add_argument("--verify", help="Verify an existing dataset's manifest, shard sizes and checksums, then exit")
    args = parser.parse_args()
    if args.verify:
        datasets = {split: PackedTokens(args.verify, split, seq_len=1, verify=True) for split in ("train", "val")}
        print(json.dumps(dict(verified=True, fingerprint=datasets["train"].fingerprint,
                              tokens={s: d.total for s, d in datasets.items()}), indent=2))
        return
    tokenizer = ByteTokenizer() if args.tokenizer == "byte" else BPETokenizer(args.tokenizer, args.tokenizer_revision, args.tokenizer_file)
    if args.local_jsonl:
        documents = local_documents(args.local_jsonl)
        source = dict(local_jsonl=args.local_jsonl)
    else:
        source = dict(dataset="HuggingFaceFW/fineweb-edu", subset=args.subset, revision=args.revision,
                      shuffle_buffer=args.shuffle_buffer, max_docs=args.max_docs)
        documents = fineweb_documents(args.subset, args.revision, args.seed, args.shuffle_buffer, args.max_docs, source)
    manifest = prepare(documents, tokenizer, args.output, train_tokens=args.train_tokens,
                       val_tokens=args.val_tokens, shard_tokens=args.shard_tokens,
                       val_fraction=args.val_fraction, seed=args.seed,
                       encode_batch_size=args.encode_batch_size, source=source)
    print(json.dumps(dict(output=str(local_path(args.output)), tokenizer=manifest["tokenizer"],
                          tokens={s: v["tokens"] for s, v in manifest["splits"].items()},
                          documents=manifest["documents"]), indent=2))


if __name__ == "__main__":
    main()
