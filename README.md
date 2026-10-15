# kv_c

本包是层间 KV cache 压缩方法的实现与证据包。项目未定名，kv_c 为临时代号。

项目从 GQA 出发且已扩展 MLA（DeepSeek-V2）和 TPA（NeurIPS 2025）

对于后二者，在潜码/因子坐标上直接应用相同的循环窄覆盖。

实现为 [factorized_memory.py](factorized_memory.py) 和 [fragment_cache.py](fragment_cache.py)，适配方式见 [方法文档](docs/method.md#mla--tpa-适配)。

MLA 另支持 `share_rope=False`：只压缩 KV 潜码，RoPE Key 每层独立生成、完整缓存。[两种 RoPE 实验结果](docs/results.md#mla-标准实验)。

## 方法

关键帧（embedding 生成的全宽底座 M₀）+ 每层在坐标上的窄发布（$Z = N(H)W$）+ 不可变提交片段 + 最近写入者静态地址直接读取 + 提交即旋转 RoPE。无前缀平均、无锚点、无混合预算、无逐层状态。缓存 $c+(L-1)w$ 标量/token，测试小模型缓存为标准逐层 GQA 的 34.375%（w16）或 45.3125%（w24）。完整构造与数学：[docs/method.md](docs/method.md)。全部已有数据：[docs/results.md](docs/results.md)。

## 证据边界

质量证据为约 188 万参数 GQA 与约 199–210 万参数 MLA/TPA 字符模型；MLA 的 15 组（原生与两种 RoPE 方式）和 TPA 的 9 组完整预算实验已完成，结果与数值限制见 [MLA](docs/results.md#mla-标准实验) 和 [TPA](docs/results.md#tpa-标准实验)。

TinyShakespeare
3 seed n=256

验证集已参与探索；无 GPU、无大模型质量、无延迟结论。容量公式与实现正确性任意尺寸成立，质量模式在更大模型实验前均为小模型假设。

## 包内索引

| 内容                               | 位置                                                                 |
| ---------------------------------- | -------------------------------------------------------------------- |
| 构造与数学                         | [docs/method.md](docs/method.md)                                      |
| 已有数据与淘汰依据                 | [docs/results.md](docs/results.md)                                    |
| 默认模型                           | [cyclic_memory.py](cyclic_memory.py)                                  |
| 前缀对照臂模型                     | [public_memory.py](public_memory.py)                                  |
| 不可变片段缓存（共用）             | [narrow_cache.py](narrow_cache.py)                                    |
| 独立核对（oracle/转换/对表）       | [work/verify_cyclic_memory.py](work/verify_cyclic_memory.py)          |
| 前缀臂核对                         | [work/verify_clean_memory.py](work/verify_clean_memory.py)            |
| 快速冒烟                           | [work/smoke.py](work/smoke.py)                                        |
| 外部文本补评                       | [work/evaluate_ood.py](work/evaluate_ood.py)                          |
| 小模型复现训练壳（none/prefix 臂） | [work/run_clean_direct.py](work/run_clean_direct.py)                  |
| 权重（standalone）                 | [weights/](weights)：none_w24、none_w16、prefix_w24 各 3 种子         |
| 配对初始化/字母表资产              | [weights/pairing/](weights/pairing)（r16 原主线基线，训练壳复现用）   |
| 数据                               | [data/](data)：tinyshakespeare.txt、ood 四份                          |
| 证据 JSON                          | [outputs/](outputs)：完整比较、OOD、锁定记录、精度/核对、本包核对记录 |

## 快速核对

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python work/verify_cyclic_memory.py --full-validation   # 全核对（数分钟）
.venv/bin/python work/smoke.py                                    # 一分钟冒烟
```

核对内容：独立全宽源 oracle、60 个 generic 配置、6 份权重的流式/不可变性/字节断言、全验证 NLL 与审计记录对表（容差 1e-4，实测全为 0.0）。前缀臂：`work/verify_clean_memory.py`（generic 部分可直接跑；trained 模式需外部完整研究 checkpoint）。

外部文本补评（探索性，非锁定记录）：

```bash
.venv/bin/python work/evaluate_ood.py --seed 0 --groups none_w24 none_w16 --threads 2
```

小模型复现训练（配对资产已在 weights/pairing）：

```bash
cd work && ../.venv/bin/python run_clean_direct.py --variant none --width 24 --seed 0 --threads 2 --tag my_run
```

## MLA 本机标准实验

MLA 按 [DeepSeek-V2 官方实现](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/94cf30b/modeling_deepseek.py) 在 Query 降维后、KV 升维前分别使用 RMSNorm（epsilon=1e−6）。缓存原始潜码，读取完整潜码后归一化；训练、prefill 与逐 token 解码使用相同语义。[方法](docs/method.md#mla) / [结果](docs/results.md#mla-标准实验)。

在项目根目录，一条命令完成核对和全部 MLA 实验：

```bash
.venv/bin/python work/run_mla_suite.py --all --jobs 2 --tag my_mla
```

原生、共享 RoPE 1/4 与 3/8、独立 RoPE 1/4 与 3/8，各种子 0/1/2，共 15 组，全部从头训练。每组 1600 步、B=16、n=256；自动完成 256/512/1024 全验证、既有 OOD 的 FP32/BF16 评分、保存加载和训练权重解码核对。最多两组并行，阻塞等待进程完成，不轮询日志或结果。每 100 步保存断点，成功后清理优化器断点；同一命令可续跑。

结果与曲线位于 `outputs/<tag>/`，standalone 权重位于 `weights/<tag>/`，按五种构造分目录。正式记录：[汇总](outputs/mla_standard/summary.json)、[核对](outputs/mla_standard/verification.json)、[初始化配对](outputs/mla_standard/pairing.json)、[权重](weights/mla_standard/)。

TPA 的原生与两档窄写共 9 组记录见 [结果](docs/results.md#tpa-标准实验) / [汇总](outputs/tpa_standard/summary.json)。两种注意力的通用核对可单独运行：

```bash
.venv/bin/python work/verify_attention_adaptation.py --output outputs/attention_verification.json
```

## 转换与来源

standalone 权重由 `verify_cyclic_memory.py --convert <研究 checkpoint 目录>` 从完整训练 checkpoint 精确转换（无适配训练）；源 checkpoint 名称与 SHA256 记录在各权重的 metadata 及 [outputs/cyclic_verification.json](outputs/cyclic_verification.json) 中。原研究树的完整研究 checkpoint（22.6MB/份）未随包携带，原件未动。

迁移时从原研究树适配的细节：训练壳与核对脚本的数据/资产路径改指包内（`data/`、`weights/pairing/`），逻辑未改；OOD 评估器仅修剪模型加载路径，语料处理/排除/chunk 口径原样（对 prefix 复算与锁定记录一致，差 0.0）。
