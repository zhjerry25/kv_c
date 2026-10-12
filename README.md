# kv_c

2026-10-12 本包是层间 KV cache 压缩方法的实现与证据包。项目未定名，kv_c 为临时代号。

## 方法

关键帧（embedding 生成的全宽底座 M₀）+ 每层在循环坐标上的残差窄增量（$Z = \text{旧值} + N(H)W$，只写本层坐标）+ 不可变提交片段 + 最近写入者静态地址直接读取 + 提交即旋转 RoPE。无前缀平均、无锚点、无混合预算、无逐层状态。缓存 $c+(L-1)w$ 标量/token，测试小模型缓存为标准逐层 GQA 的 34.375%（w16）或 45.3125%（w24）。完整构造与数学：[docs/method.md](docs/method.md)。全部已有数据与淘汰依据：[docs/results.md](docs/results.md)。

## 证据边界

质量证据为约 188 万参数字符模型（TinyShakespeare，三种子，训练长度 256）；验证集已参与探索；无 GPU、无大模型质量、无延迟结论。容量公式与实现正确性任意尺寸成立，质量模式在 135M 复现前均为小模型假设。

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
| 数据                               | [data/](data)：tinyshakespeare.txt（SHA 86c4e6aa…）、ood 四份        |
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
cd work && ../../.venv/bin/python run_clean_direct.py --variant none --width 24 --seed 0 --threads 2 --tag my_run
```

## 转换与来源

standalone 权重由 `verify_cyclic_memory.py --convert <研究 checkpoint 目录>` 从完整训练 checkpoint 精确转换（无适配训练）；源 checkpoint 名称与 SHA256 记录在各权重的 metadata 及 [outputs/cyclic_verification.json](outputs/cyclic_verification.json) 中。原研究树的完整研究 checkpoint（22.6MB/份）未随包携带，原件未动。

迁移时从原研究树适配的细节：训练壳与核对脚本的数据/资产路径改指包内（`data/`、`weights/pairing/`），逻辑未改；OOD 评估器仅修剪模型加载路径，语料处理/排除/chunk 口径原样（对 prefix 复算与锁定记录一致，差 0.0）。
