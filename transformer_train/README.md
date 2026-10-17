# kvtrain

独立单卡 Transformer 训练框架。只需复制本目录；代码、环境、数据、HF 缓存和训练输出均在本目录内，不引用外部项目文件。模型使用 pre-RMSNorm、RoPE、SwiGLU、共享 embedding/LM head；优化器为 AdamW，支持梯度累积、激活重算、分块交叉熵和完整断点恢复。

## 环境

本机 CPU：在本目录执行 `python3 -m venv .venv`，然后 `.venv/bin/pip install -e .`。以下命令中的 `python` 指已激活环境的解释器。

云端 RTX PRO 6000 Blackwell 建议使用已包含 PyTorch、CUDA 和 Transformer Engine 的 [NGC 26.07 容器](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-07.html)。仅挂载本目录：

```bash
cd transformer_train
docker run --gpus all --ipc=host --rm -it -e NVIDIA_IMEX_CHANNELS=0 \
  -v "$PWD:/workspace" -w /workspace nvcr.io/nvidia/pytorch:26.07-py3 bash
pip install -e .
```

已有兼容 CUDA 12.8+ 的 PyTorch 环境时，执行 `pip install -e .`；FP8 另需 `pip install --no-build-isolation 'transformer_engine[pytorch]>=2.14.1'`。本机 `.venv` 不要上传到 Linux。

## FineWeb-Edu 数据管线

默认流式读取 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 的 `sample-10BT`，使用 GPT-2 BPE。默认准备 10M train / 100K val token；正式训练可扩大预算：

```bash
python -m kvtrain.data --train-tokens 1000000000 --val-tokens 1000000
python -m kvtrain.data --verify data/fineweb-edu
```

逐文档按文本 hash 分离训练/验证，相同文本始终归入同一 split；分词后加 EOS，连续打包为 uint16 分片（大词表自动 uint32）。不做语料全量去重，文档间允许注意力。训练按随机窗口 mmap 读取，窗口可跨分片，输入/标签严格错位一位。分片大小默认 50M token，可用 `--shard-tokens` 修改。

`manifest.json` 保存实际数据集 commit、分词器 SHA256、seed、预算和分片校验和；成功写完两个 split 才发布 manifest。输出目录必须为空，中断的数据准备请使用新目录重新开始。训练断点包含数据 manifest 指纹及采样器状态。

数据准备可设置 `--subset sample-100BT`、`--revision <commit>`、`--shuffle-buffer 10000`、`--encode-batch-size 128`。离线输入使用 `--local-jsonl data/input.jsonl`，每行须有字符串 `text`。`--tokenizer-file data/tokenizer.json` 可使用自有 BPE，要求定义 `<|endoftext|>`；`--tokenizer byte` 仅供离线测试。

## 训练

```bash
# 默认 GQA、标准模式、BF16；模型词表自动取数据词表并对齐至 64。
python -m kvtrain.train --preset 125m --steps 10000 --out runs/gqa-standard

# 循环窄覆盖方法 + MLA + SWA + FP8，RTX PRO 6000 可用的起步配置。
python -m kvtrain.train --preset 125m --attention mla --mode cyclic \
  --write-fraction 1/4 --window 512 --fp8 delayed \
  --seq-len 2048 --batch-size 2 --accumulation 8 \
  --steps 10000 --out runs/mla-cyclic

# 完整恢复模型、优化器、LR 进度、随机数与数据采样位置。
python -m kvtrain.train --resume runs/mla-cyclic/last.pt

# 更大模型，先从较小 micro batch 开始。
python -m kvtrain.train --preset 1b --attention gqa --mode cla \
  --cla-group-size 2 --fp8 delayed --batch-size 1 --accumulation 16 \
  --steps 10000 --out runs/1b-cla
```

| 开关 | 行为 |
| --- | --- |
| `--attention gqa / mla / tpa` | GQA；带 Q/KV latent norm 的 MLA；上下文生成 Q/K/V 两组因子的 TPA |
| `--mode standard` | 每层独立产生完整 KV 表示 |
| `--mode cla --cla-group-size 2` | 每组首层产生 KV，后续层复用；Q、FFN 和 MLA reader 私有，无 detach |
| `--mode cyclic --write-fraction 1/4` | embedding 全宽关键帧 + 每层循环窄覆盖；无残差发布、前缀平均或额外时间状态 |
| `--window N` | 三种注意力及三种模式均可叠加因果 SWA；N 包括当前 token，0 为全因果 |
| `--no-share-rope` | MLA 各层独立完整位置 Key，仅潜码跨层共享/窄覆盖 |
| `--fp8 off / delayed / current / mxfp8` | BF16 基线 / 延迟缩放 FP8 / 当前缩放 FP8 / 数据中心 Blackwell MXFP8 |
| `--compile` | 编译私有 Transformer block；支持 off/current，delayed 与之不兼容 |
| `--no-activation-checkpoint` | 关闭默认激活重算 |

`cyclic` 使用固定最近写入者布局，Key 发布即旋转，只选择完整 RoPE 对。MLA 缓存原始潜码与位置 Key，读取完整潜码后做本层 norm；TPA 缓存四组因子，保留各自 1/rank 缩放与 Xavier 定标。训练临时展开当层 K/V 使用 PyTorch SDPA；GPU SWA 使用编译的 FlexAttention 块稀疏内核，CPU 使用小规模精确 mask。

同精度缓存标量为 `C+(L-1)w`，标准模式为 `L*C`；CLA 为每组一个完整表示。`--no-share-rope` 额外计入每层完整位置 Key。训练中自动微分需要保留/重算激活，不能把推理 KV 压缩率直接当成训练显存节省率。

预设的标准 GQA 参数量（GPT-2 词表）：50m=53.95M、125m=126.74M、350m=344.64M、1b=989.15M；注意力/模式改变会改变实际参数量。启动时输出准确参数量、缓存预算和 FP8 层数。自定义尺寸可用 `--config configs/custom.json`（ModelConfig 字段）；`--dry-run` 用 meta device 检查结构，不分配模型权重。

输出 `metrics.jsonl`、`config.json`、`last.pt` 和便携权重 `model.pt`。验证使用 BF16/FP32 master weights，避免改变 FP8 缩放历史。resume 沿用保存的配置；若需改变模型或训练计划，应新建 run。`--stop-after N` 可提前停止，同时保留原 `--steps` 的 LR 计划。

## FP8 与云端测量

RTX PRO 6000 是 SM120；[TE 当前支持普通 FP8，MXFP8 训练仅支持 SM100/103](https://nvidia.github.io/TransformerEngine/support_matrix.html)。本框架对不支持的 FP8 请求直接报错。FP8 用于对齐的大型 Linear 的前向/反向 GEMM；窄/不对齐投影、norm、RoPE、注意力和 LM head 保持 BF16/FP32。参数和 Adam 状态保留 FP32。FP8 要求 `batch_size * seq_len` 为 16 的倍数；MXFP8 为 32。

```bash
# 同一初始权重，预热后比较真实 forward/backward/AdamW，输出 tokens/s、显存与相对 BF16 加速比。
python -m kvtrain.bench --preset 125m --attention mla --mode cyclic \
  --seq-len 2048 --batch-size 2 --warmup 3 --steps 10
```

本机没有 GPU，CUDA、FlexAttention GPU 内核、FP8 实际吞吐及大模型质量需云端测量；未声称已获得 GPU 加速比。`Transformer.step` 是用于核对语义的参考解码，缓存只保存不可变片段，但读取会临时展开 MLA/TPA 的 K/V，并非专用高性能推理引擎。

## 已运行的必要检查

```bash
python -m unittest discover -s tests -v
# 小量真实数据接入；输出目录须未使用。
python -m kvtrain.data --output data/live-smoke --train-tokens 8192 \
  --val-tokens 2048 --val-fraction 0.25 --shard-tokens 4096 \
  --encode-batch-size 8 --shuffle-buffer 0 --max-docs 128
python -m kvtrain.train --preset tiny --attention mla --mode cyclic \
  --data data/live-smoke --out runs/live-smoke --device cpu --steps 2 \
  --seq-len 32 --batch-size 1 --accumulation 1 --warmup 0 \
  --eval-every 1 --eval-batches 1 --log-every 1
```

CPU 检查覆盖模式组合、跨圈覆盖、缓存不可变与逐 token 等价、MLA 独立 RoPE、梯度重算一致性、BF16 反向、分块 loss、分片边界、校验和损坏检测及完整断点恢复。实际本机验证记录见 `validation.json`。

## 50M 单 seed write fraction 扫描

```bash
# 准备一次，全部实验共享；输出目录须为空。
python -m kvtrain.data --output data/fineweb-edu-1b --seed 0 \
  --train-tokens 1100000000 --val-tokens 2000000
python -m kvtrain.data --verify data/fineweb-edu-1b
python -m kvtrain.sweep --dry-run
python -m kvtrain.sweep --compile
```

默认 24 组：GQA/MLA/TPA 各一组 standard，加 cyclic 的 `1/16, 1/8, 1/4, 3/8, 1/2, 3/4, 1`。固定 d=512、L=10、FFN=1408、完整因果注意力、MLA 共享位置 Key，实际参数约 52–57M；不改变主干来补齐各比例减少的发布参数。10 层的缓存比例依次为 15.625%、21.25%、32.5%、43.75%、55%、77.5%、100%。其中 1/16 只在其余九层覆盖 56.25% 的公共坐标，1/8 已能完整覆盖一轮。

训练 seed=0，2048 context，micro batch=4，accumulation=4，32768 steps，即每组 1,073,741,824 个采样 token，24 组合计约 25.77B。LR=3e-4，warmup=512，cosine；每 1024 步验证相同的 524,288 个采样 token 并保存断点。默认 BF16、FP8 off，避免不同窄宽度的 FP8 对齐资格成为额外变量。`--compile` 可选；需要 FP8 时可用 `--fp8 current --compile` 或 `--fp8 delayed`，该矩阵的结论会同时包含 FP8 策略影响。

脚本自动使用 `--paired-init`：按参数名称独立固定随机流，同架构各比例的私有初始权重一致；窄发布从同一层的全宽随机银行选择对应行。关键帧与 standard 首层发布初始张量一致。保留原方法的初始化幅度：GQA/MLA 的后续 cyclic 发布 std=.0016，standard std=.02；TPA 按完整因子宽度 Xavier 定标。因此 cyclic=1 仍是独立的全宽方法对照，不能直接视作 standard。

单 GPU 逐组执行：先完成三个 standard 和三个 cyclic=1 对照，再按比例轮转三种架构。重跑同一命令自动续训未完成组、跳过已完成组。输出 `runs/50m-write-sweep/summary.csv` 与 `summary.json`，包含最终验证 NLL/PPL、相对各自 standard / cyclic=1 的 ΔNLL、实际参数、绝对发布宽度、缓存比例、吞吐和显存。首要比较同架构的 ΔNLL–缓存比例曲线；单 seed 用于探索拐点，不报告统计显著性。

修改 `--steps`、`--fractions`、`--attention`、`--window` 等配置请使用新 `--out`；矩阵会校验计划、数据指纹与 PyTorch 版本。默认要求训练语料至少达到每组 token 预算，显式 `--allow-data-repeat` 可放开此检查；随机窗口采样本身允许重复。开始前追加 `--include-cla` 可扩展为 27 组（每种注意力增加 CLA group-size=2 对照）。
