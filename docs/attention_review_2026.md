# 主流大语言模型 Attention 架构研究综述（截至 2026 年 8 月）

> 资料口径：本文优先采用官方 Hugging Face 模型卡、官方技术报告、官方仓库和 arXiv 论文。闭源模型若官方未披露实现细节，统一标注“官方未完全披露”，不把社区推测写成事实。
> 量化口径：不同论文和团队使用不同 GPU、序列长度、批大小与量化方式，所有性能数字只在各自来源口径内成立，不能直接横向比较。
> 核验记录：本次修订已根据官方 HF 配置修正 GLM-5.1 上下文为 202,752；新增 Phi、DBRX、Nemotron、InternLM、Baichuan、Step、MiMo、Zamba、Arctic 系列的 Attention 配置均来自官方模型卡/配置或可复核的公开配置镜像。

---

## 一、执行摘要

1. **层内 KV 压缩已成为主流**：MHA 到 MQA/GQA 再到 MLA 的演进主线非常清晰；截至 2026 年，DeepSeek、GLM、Kimi、MiniMax M3、Mistral Small 4 等均在不同程度使用 MLA 类或共享 KV 方案，Phi、DBRX、InternLM、Nemotron、Step、MiMo、Zamba、Arctic 与 Hunyuan 则继续验证 GQA、SWA、SSM 与混合架构。
2. **长上下文瓶颈从“能不能训练”转向“能不能低成本推理”**：2024 年主流是 128K，2025 年出现 256K 至 1M 开源模型，2026 年 DeepSeek-V4、GLM-5.2、Kimi K3、MiniMax M3 都把 1M 上下文作为生产目标。
3. **稀疏与压缩注意力重新成为主线**：DeepSeek-V4 使用 CSA + HCA，GLM-5 使用 DSA，MiniMax M3 使用 MSA，三者都在 KV 数量维度做压缩或选择，而不是只依赖窗口注意力。
4. **线性注意力在超长上下文中重新崛起**：Qwen3-Next、Qwen3.5/3.6/3.8、Kimi Linear/K3 都采用 Gated DeltaNet 或 KDA 与 GQA/MLA 混合，形成“3:1 线性到全量”的常见结构。
5. **FlashAttention 和 PagedAttention 是系统级加速，不是新的 Attention 数学形式**：训练常用 FlashAttention 系列，推理服务常用 PagedAttention、FlashMLA、FP4 indexer cache 等，它们可与任何架构组合。
6. **GQA 仍是最稳妥的默认选择**：Qwen2.5、Llama 3/4、MiniMax M2/M2.7、GLM-4.7 等继续使用 GQA；MLA 则更适合超大 MoE 和长上下文高并发场景。
7. **闭源模型透明性显著低于开源模型**：GPT-5.6 Sol、Claude Fable 5/Opus 4.8、Gemini 3.x 的架构细节均未披露，只能根据官方声明和公开基准确认产品能力。
8. **位置编码与 Attention 强耦合**：RoPE、YaRN、NTK-aware、ALiBi、partial RoPE、p-RoPE 会直接影响长上下文效果，不能把“上下文长度提升”单独归因于 Attention 结构升级。
9. **未来 1 到 2 年的关键方向是“少量全量注意力 + 压缩/线性/稀疏/SSM 混合”**：纯 MHA 会继续退出大模型主力，但完全线性化仍需要解决长距离召回和训练稳定性问题。
10. **代码交付已覆盖教学实现**：本仓库新增 SWA、Block Sparse、Linear Attention、PagedAttention 教学接口，并提供单元测试与统一接口说明。

---

## 快速总览（截至 2026-08）

| 系列 | 最新代表 | Attention 核心 | 上下文 | 说明 |
|------|----------|----------------|--------|------|
| Qwen | Qwen3.8-2.4T-A95B | Gated DeltaNet + Gated Attention | 262K，扩展 1M | 512 专家 Top-10 |
| DeepSeek | V4-Pro-0813 | MLA + CSA + HCA | 1M | 1M 下 KV 约为 V3.2 的 10% |
| GLM | GLM-5.2 | MLA + DSA + IndexShare | 1M | IndexShare 降 2.9x FLOPs |
| Kimi | Kimi K3 | 69 KDA + 24 Gated MLA | 1M | 2.8T/104B Active |
| MiniMax | M3 | GQA + MSA | 1M | 相对 M2 的 prefill 9x、decode 15x |
| Llama | Llama 4 Scout/Maverick | GQA + chunked local + NoPE 间隔 | 10M/1M | MoE |
| Mistral | Mistral-Small-4 | MLA 类 | 1M | `kv_lora_rank=256` |
| Gemma | Gemma 4 | GQA + local/global | 128K-256K | p-RoPE，Unified K/V |
| GPT | GPT-5.6 Sol | 官方未披露 | 官方未披露 | 闭源 |
| Gemini | Gemini 3.x | 官方未披露 | 官方未披露 | 闭源 |
| Claude | Fable 5 / Mythos 5 | 官方未披露 | 官方未披露 | 闭源 |
| Phi | Phi-4-mini | GQA | 128K | 3.8B Dense，LongRope |
| DBRX | DBRX-Instruct | GQA 48Q/8KV | 32K | 132B/36B Active，16 专家 Top-4 |
| Nemotron | Nemotron-3-Super | Mamba-2 + MoE + GQA | 256K，可扩展 1M | 120B/12B Active，LatentMoE |
| InternLM | InternLM3-8B | GQA 32Q/2KV | 32K | Dynamic RoPE |
| Baichuan | Baichuan-M3-235B | 继承 Qwen3 的 GQA | 40K | 基于 Qwen3-235B-A22B |
| Step | Step-3.7-Flash | GQA 类 + SWA + MoE | 256K | 198B/11B Active，多模态 |
| Xiaomi MiMo | MiMo-V2.5-Pro | SWA + Global Attention 6:1 | 1M | 1.02T/42B Active，KV 减少约 7x |
| Zamba | Zamba2-7B | Mamba2 + Shared Attention | 4K，可扩展 16K | SSM/Transformer 混合 |
| Snowflake Arctic | Arctic-Instruct | GQA + Dense-MoE Hybrid | 4K | 480B，128 专家 Top-2 |
| Hunyuan | Hy3 | GQA 64Q/8KV | 256K | 295B/21B Active，192 专家 Top-8 |

**Attention 演进主线**

- 2023 年：GQA 成为开源模型默认选择。
- 2024 年：DeepSeek-V2 提出 MLA，KV Cache 相对 MHA 减少约 93.3%。
- 2025 年：Qwen3-Next 和 Kimi Linear 验证 Gated DeltaNet/KDA 线性注意力；Llama 4 和 Gemma 3 验证 MoE + 局部注意力。
- 2026 年：DeepSeek-V4、GLM-5.2、MiniMax M3 把稀疏/压缩注意力推到 1M 上下文生产场景；Kimi K3 达到 2.8T 参数。

## 信息核验记录

- GLM-5.1 上下文已根据官方配置从 1M 修正为 202,752。
- Zamba-7B 的 Attention 核心修正为 Mamba + Shared Attention，避免与 Zamba2 的 Mamba2 混淆。
- Hunyuan-A13B、Hy3、Hy-MT2、Step-3.7、MiMo-V2.5-Pro、Zamba2、Arctic 均以官方 HF 配置或官方模型卡核验。
- Step-3.7-Flash 的 KV head 数量未在顶层配置完整公开，本文按“GQA 类”表述，不写成确定值。
- GPT、Gemini、Claude 等闭源模型的 Attention 细节继续标注“官方未完全披露”，不将推测写成事实。
- 文中量化数字均来自对应技术报告、官方模型卡或论文；不同测试口径不直接横向比较。

---

## 二、模型系列逐一分析

### 2.1 Qwen 系列（阿里巴巴）

Qwen 的演进路线可以概括为：GQA 基线 -> Gated DeltaNet + Gated Attention 混合 -> 更大 MoE 与多模态扩展。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Qwen2 / 2.5 | 2024-06 / 2024-09 | 是 | Dense 0.5B-72B | GQA | 32K，YaRN 128K | RoPE base 1M，SwiGLU，RMSNorm |
| Qwen3 | 2025-04 | 是 | Dense/MoE | GQA + QK-Norm 类优化 | 32K-128K 级 | 混合思考模式，统一 GQA |
| Qwen3-Next-80B-A3B | 2025-09 | 是 | MoE 80B/3B Active | Gated DeltaNet + Gated Attention 3:1 | 256K，YaRN 扩展 1,010,000 | 12 个隐藏块，每块 3 层线性 + 1 层全量 |
| Qwen3.5 / 3.6 | 2025-2026 | 是 | Dense/MoE | Gated DeltaNet + Gated Attention | 256K，可扩展 1M | 9B、27B、35B-A3B 等尺寸 |
| Qwen3.8-27B | 2026-08-05 | 是 | 多模态 Dense | Gated DeltaNet + Gated Attention | 262,144 原生，托管 1M | 16 个隐藏块，Gated Attention 24Q/4KV，head_dim 256 |
| Qwen3.8-2.4T-A95B | 2026-08-12 | 是 | MoE 2.4T/95B Active | Gated DeltaNet + Gated Attention | 262,144 原生，扩展 1M | 512 专家 Top-10，Gated Attention 64Q/4KV |

**关键事实**

- Qwen3-Next 官方模型卡明确给出 Hidden Layout：`12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))`，其中 Gated DeltaNet 使用 32 个 V head、16 个 QK head，head_dim 128；Gated Attention 使用 16Q/2KV，head_dim 256。
- Qwen3.8-27B 官方模型卡给出：`16 * (3 * (Gated DeltaNet -> FFN) -> 1 * (Gated Attention -> FFN))`；Gated Attention 为 24Q/4KV，Gated DeltaNet 为 48 V head / 16 QK head。
- Qwen3.8-2.4T 官方模型卡给出 23 个类似隐藏块，512 个专家、每 Token 激活 10 个专家。
- 从 Qwen3-Next 开始，Qwen 系列从“GQA 全量注意力”转向“少量全量注意力 + 线性注意力”的混合路线，目标是在 256K 到 1M 上下文下降低解码成本。

### 2.2 DeepSeek 系列（深度求索）

DeepSeek 是 MLA 的提出者和主要推动者，2026 年进一步进入“MLA + 压缩稀疏注意力”阶段。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| DeepSeek LLM 7B/67B | 2024-01 | 是 | Dense | MHA | 4K 级 | 标准稠密 Transformer |
| DeepSeek-V2 / V2.5 | 2024-05 / 2024-09 | 是 | MoE 236B/21B Active | MLA | 128K | KV Cache 相对 MHA 对照减少约 93.3% |
| DeepSeek-V3 | 2024-12 | 是 | MoE 671B/37B Active | MLA | 128K | MLA + DeepSeekMoE + 多 Token 预测 |
| DeepSeek-R1 | 2025-01 | 是 | MoE 671B/37B Active | MLA | 128K | 基于 V3-Base 的强化学习推理 |
| DeepSeek-V3.2 | 2025-09 | 是 | MoE | MLA + DSA | 128K | CSA + HCA，长上下文稀疏化 |
| DeepSeek-V4-Pro | 2026-04 预览，2026-08-13 正式 | 是 | MoE 1.6T/49B Active | MLA 类 + CSA + HCA + SWA 分支 | 1,048,576 | 1M 下约 27% 推理 FLOPs、10% KV Cache（相对 V3.2） |
| DeepSeek-V4-Flash | 2026-04 预览，2026-07-31 更新 | 是 | MoE 284B/13B Active | MLA 类 + CSA + HCA + SWA 分支 | 1,048,576 | 1M 下约 10% 推理 FLOPs、7% KV Cache（相对 V3.2） |

**DeepSeek-V4 技术要点**

- 官方技术报告标题为《DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence》。
- V4 使用混合注意力：CSA 先把每 `m` 个 Token 的 KV 压缩为一个 entry，再执行 DeepSeek Sparse Attention；HCA 使用更大压缩比例 `m' >> m` 并把压缩后的 KV 保持为稠密注意力。
- 配置中出现 `q_lora_rank`、`qk_rope_head_dim`、`sliding_window=128` 和 1 个共享 KV head，说明 MLA 的低秩潜向量、解耦 RoPE 和局部窗口分支在 V4 中继续保留。
- 推理侧还包含 FP4 indexer cache、DSpark 投机解码、异构 KV cache 与 on-disk KV cache，这些是系统层优化而非模型数学本身。

### 2.3 GLM 系列（智谱 AI）

GLM 的公开演进路线是：MHA -> MQA -> GQA -> MLA + DSA。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| GLM-130B | 2022 | 是 | Dense 130B | MHA | 2K 级 | DeepNorm + RoPE + GLU |
| ChatGLM2/3 | 2023 | 是 | Dense 6B | MQA | 8K，可扩展 32K | 小模型 KV Cache 压缩 |
| GLM-4 | 2024-01 API，2024-06 开源 9B | 是 | Dense 9B | GQA | 128K | 从 MQA 转向 GQA |
| GLM-4.7 | 2025-12 | 是 | MoE | GQA 96Q/8KV | 202,752 | 大 MoE 使用 GQA |
| GLM-4.7-Flash | 2026-01 | 是 | MoE Lite | MLA 类 | 202,752 | `kv_lora_rank=512` |
| GLM-5 | 2026-02 | 是 | MoE 744B/40B Active | MLA + DSA | 202,752 | 引入 DeepSeek Sparse Attention |
| GLM-5.1 | 2026-04 | 是 | MoE | MLA + DSA | 202,752 | 长时任务能力升级 |
| GLM-5.2 | 2026-06 | 是 | MoE | MLA + DSA + IndexShare | 1,048,576 | IndexShare 在 1M 上下文降低 2.9x 每 Token FLOPs |

**关键事实**

- GLM-5 官方模型卡确认从 355B/32B Active 扩展到 744B/40B Active，并集成 DSA 以降低部署成本。
- GLM-5.2 官方模型卡提出 IndexShare：每 4 个稀疏注意力层共享同一个 indexer，在 1M 上下文降低每 Token FLOPs 约 2.9 倍。
- GLM-4.7 的配置仍为纯 GQA（96Q/8KV），GLM-4.7-Flash 已出现 `kv_lora_rank=512`，说明智谱在轻量模型上先验证 MLA，再在 GLM-5 系列大规模启用。

### 2.4 Kimi 系列（月之暗面）

Kimi 的演进路线是：长上下文系统工程 -> MLA/MoE -> KDA 线性注意力混合 -> 3T 级混合架构。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Kimi 初代 | 2023-10 | 否 | Dense Transformer | MHA 类 + Ring Attention | 技术报告 128K | 分布式长序列注意力 |
| Moonlight-16B-A3B | 2025-02 | 是 | MoE 16B/3B Active | DeepSeek-V3 类 MLA | 技术报告口径 | 小 MoE 训练效率 |
| Kimi K2 | 2025-07 | 是 | MoE 1T/32B Active | MLA | 256K | 1T 级 MoE + MLA |
| Kimi K2.5 / K2.6 / K2.7 | 2026 | 是 | MoE | MLA | 256K | 多模态扩展，64Q，`kv_lora_rank=512` |
| Kimi Linear | 2025-10 | 是 | MoE 48B/3B Active | KDA + Gated MLA 3:1 | 1M | KV Cache 最高减少 75%，解码吞吐最高提升约 6x |
| Kimi K3 | 2026-06 | 是 | MoE 2.8T/104B Active | 69 KDA + 24 Gated MLA | 1,048,576 | Attention Residuals，Stable LatentMoE 16/896 专家 |

**关键事实**

- Kimi Linear 官方模型卡明确：KDA 是 Gated DeltaNet 的精细化版本，使用 3:1 的 KDA 与全局 MLA 混合；在 128K 上下文 RULER 上报告 84.3 分和 3.98x 速度提升，在 1M 上下文中相对 MLA 报告约 6.3x 更快的 TPOT。
- Kimi K3 官方模型卡明确：2.8T 总参、104B 激活、93 层、1 个 Dense 层、69 KDA + 24 Gated MLA、896 专家、每 Token 激活 16 个专家、2 个共享专家、1M 上下文。
- K3 使用 Attention Residuals（AttnRes）替代普通残差连接，这是 Kimi 对传统 Pre-Norm/Post-Norm 之外跨层信息路径的尝试。

### 2.5 MiniMax 系列

MiniMax 的路线从“线性注意力 + 全量注意力混合”转向“GQA 基线 + 学习式稀疏注意力”。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| MiniMax-Text-01 / MiniMax-01 | 2025-01 | 是 | MoE 456B/45.9B Active | Lightning Attention + Softmax Attention | 1M | Tiling + Intra-block 线性注意力 |
| MiniMax-M1 | 2025-06/07 | 是 | MoE | 混合架构延续 | 40K/80K | 轻量档位 |
| MiniMax-M2 | 2025-10 | 是 | MoE | GQA 48Q/8KV | 196,608 | 回归全量注意力 |
| MiniMax-M2.5 / M2.7 | 2026 | 是 | MoE | GQA 48Q/8KV | 204,800 | 256 专家 Top-8，RoPE theta 5M |
| MiniMax-M3 | 2026-06 | 是 | MoE 428B/23B Active | GQA + MiniMax Sparse Attention | 1,048,576 | 稀疏选择 + 专用 GPU kernel |

**关键事实**

- MSA 论文明确：MSA 是建立在 GQA 上的 blockwise sparse attention，Index Branch 为每个 GQA group 独立选择 Top-k KV block，Main Branch 只对选中 block 做精确 block-sparse attention。
- MSA 论文在 109B 模型上报告：1M 上下文每 Token 注意力计算降低 28.4x，H800 上 prefill 14.2x、decode 7.6x 墙钟加速。
- MiniMax-M3 模型卡报告：相对 M2，在 1M 上下文下 prefill 9x、decode 15x 加速，每 Token 计算降至 1/20。两份数字测试口径不同，均保留。

### 2.6 Llama 系列（Meta）

Llama 系列是开源模型从 MHA 走向 GQA，并进一步走向 MoE + 局部注意力的代表性路径。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Llama 1 | 2023-02 | 是 | Dense 7B-65B | MHA | 2K | RoPE + SwiGLU + RMSNorm |
| Llama 2 | 2023-07 | 是 | Dense 7B/13B/70B | 小模型 MHA，70B GQA | 4K | 70B 使用 8 KV Head |
| Llama 3 / 3.1 / 3.2 / 3.3 | 2024 | 是 | Dense 8B-405B | 全尺寸 GQA | 8K-128K | RoPE base 500K，128K 训练 |
| Llama 4 Scout / Maverick | 2025-04 | 是 | MoE | GQA + chunked local attention + NoPE 间隔层 | Scout 10M，Maverick 1M | MoE，QK-Norm，scaled RoPE，attention temperature tuning |

**关键事实**

- Meta 官方 Llama 4 模型卡确认 Scout 为 109B 总参/17B 激活、16 专家、上下文 10M；Maverick 为 400B 总参/17B 激活、128 专家、上下文 1M。
- 官方源码 `args.py` 和 `model.py` 显示 Llama 4 使用 `n_kv_heads`（GQA）、`nope_layer_interval`（无位置编码间隔层）、`attention_chunk_size`（局部 chunked attention）和 `use_qk_norm`。
- Llama 4 的“10M 上下文”不等于所有层都执行全量注意力；源码中的 chunked local attention 和 NoPE 层说明长上下文依赖局部窗口、稀疏层间设计和位置编码缩放。

### 2.7 GPT 系列（OpenAI）

GPT 系列早期有公开架构，2023 年后闭源。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| GPT-1 | 2018 | 是 | Dense 117M | MHA | 512 | decoder-only Transformer |
| GPT-2 | 2019 | 是 | Dense 1.5B | MHA | 1,024 | 分层 decoder |
| GPT-3 | 2020 | 是 | Dense 175B | MHA | 2,048 | 标准 MHA，无公开稀疏注意力 |
| GPT-3.5 / GPT-4 / GPT-4o / o1 | 2022-2024 | 否 | 官方未完全披露 | 官方未完全披露 | 4K-128K 产品档位 | 产品级长上下文，架构细节未公开 |
| GPT-5.5 / GPT-5.6 Sol | 2025-2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 从多家官方基准看是 2026 年主力模型，但未披露 Attention |

**不确定性声明**：OpenAI 官方页面当前无法直接抓取正文，GPT-5.6 Sol 的公开名称来自 Kimi K3、Qwen3.8 等官方模型卡与 DeepSeek-V4 模型卡引用的 OpenAI 页面；其 Attention 架构无公开证据，不应推断为 GQA 或 MLA。

### 2.8 Gemini 系列（Google）

Gemini 是闭源多模态系列，公开细节远少于 Gemma。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Gemini 1.0 | 2023-12 | 否 | 官方未完全披露 | 官方未完全披露 | API 初期 32K 级 | 多模态 |
| Gemini 1.5 Pro / Flash | 2024 | 否 | 官方未完全披露，含 MoE | 官方未完全披露 | 1M，后 2M，研究 10M | 长上下文基础设施 |
| Gemini 2.0 / 3.x | 2024-2026 | 否 | 官方未完全披露 | 官方未完全披露 | 产品档位持续扩大 | Gemini 3.1 Pro 等名称出现在多个官方基准中 |

**开源对照**：Google 的 Gemma 系列公开了 Attention 细节，是研究 Gemini 架构的重要替代资料。

### 2.9 Claude 系列（Anthropic）

Claude 全程闭源，Anthropic 未公开 MHA/GQA/MLA 等实现。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Claude 1 / 2 / 2.1 | 2023 | 否 | 官方未完全披露 | 官方未完全披露 | 早期 9K 到 200K | 长会话与系统级上下文处理 |
| Claude 3 / 3.5 / 4.x | 2024-2026 | 否 | 官方未完全披露 | 官方未完全披露 | 产品级 200K 到更大档位 | 多模态与 agentic 能力 |
| Claude Opus 4.8 / Fable 5 / Mythos 5 | 2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | Anthropic 官方页面确认 Fable 5 与 Mythos 5 于 2026-06 发布 |

**不确定性声明**：Claude 的“上下文压缩”是系统级能力，不能等同于模型层新 Attention。

### 2.10 Mistral 系列（Mistral AI）

Mistral 的演进路线是：GQA + SWA -> GQA 全注意力 -> 2026 年 MLA 类小模型。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Mistral 7B | 2023-09 | 是 | Dense 7B | GQA + SWA | 8K 级，窗口 4096 | Rolling Buffer KV Cache |
| Mixtral 8x7B | 2023-12 | 是 | MoE 46.7B/12.9B Active | GQA 全注意力 | 32K | 8 专家 Top-2 |
| Mistral Large | 2024-02 | 部分 | 123B 级 | 官方未完整披露 | 32K | 闭源/API 混合发布 |
| Codestral / NeMo / Small | 2024 | 是 | Dense/MoE | GQA 系 | 32K-128K | 代码、边缘与部署档位 |
| Mistral-Large-3 | 2025-12 | 是 | MoE 673B/39B Active | MHA 128Q/128KV | 256K | 128 专家 Top-4 + 1 共享专家 |
| Ministral-3 | 2025-12 | 是 | Dense/MoE | GQA/MLA 类视尺寸而定 | 256K | 3B/8B/14B 推理档位 |
| Mistral-Small-4 | 2026-01 | 是 | MoE | MLA 类 | 1M | 32Q/32KV，`kv_lora_rank=256`，`q_lora_rank=1024` |
| Mistral-Medium-3.5 | 2026-03 | 是 | Dense 128B | GQA 96Q/8KV | 256K | 长上下文 Dense 档位 |

**关键事实**：Mistral-Large-3 的 `params.json` 显示 128 个 Query/KV head，属于 MHA；Mistral-Small-4 的 text config 显示 `kv_lora_rank` 和 `q_lora_rank`，属于 MLA 类实现。

### 2.11 Mixtral 系列

Mixtral 可作为“MoE 化 GQA”的早期验证。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Mixtral 8x7B | 2023-12 | 是 | MoE 46.7B/12.9B Active | GQA | 32K | 8 专家 Top-2 |
| Mixtral 8x22B | 2024-04 | 是 | MoE 141B/39B Active | GQA | 64K 级 | 更大 MoE，长上下文 |

### 2.12 Yi 系列（01.AI）

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Yi-6B / Yi-34B | 2023-11 | 是 | Dense | GQA | 4K | 32Q/4KV 与 56Q/8KV |
| Yi-34B-200K | 2023-12 | 是 | Dense | GQA + RoPE 外推 | 200K | Dynamic NTK / LongLoRA 类扩展 |
| Yi-1.5 | 2024-06 | 是 | Dense | GQA | 16K 级 | 指令微调升级 |

### 2.13 Gemma 系列（Google）

Gemma 是 Google 提供公开 Attention 细节的开放系列。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Gemma 1 | 2024-02 | 是 | Dense 2B/7B | MHA | 8K | RoPE + RMSNorm + SwiGLU |
| Gemma 2 | 2024-06 | 是 | Dense 2B/9B/27B | GQA + 局部/全局交替 | 8K | 4096 滑动窗口局部层 |
| Gemma 3 | 2025-03 | 是 | Dense/MoE | GQA + 窗口注意力 | 128K | 小窗口 + QK-Norm |
| Gemma 4 | 2026-05 | 是 | Dense/MoE | GQA + local sliding + global attention | E2B/E4B 128K，12B/26B-A4B/31B 256K | p-RoPE，global 层 Unified K/V，1024 滑动窗口 |

**Gemma 4 官方模型卡原文要点**：local sliding window attention 与 full global attention 交错，且最后一层始终是 global；小模型窗口 512，中大型模型窗口 1024。

### 2.14 Falcon 系列（TII）

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Falcon 7B / 40B | 2023 | 是 | Dense | MQA + ALiBi | 2K 级 | 单 KV Head |
| Falcon-180B | 2023-09 | 是 | Dense | MQA 系 | 2K 级 | 大规模 MQA 验证 |
| Falcon-Mamba | 2024-10 | 是 | SSM | Mamba-2 固定状态 | 32K | 无传统 KV Cache |

### 2.15 PaLM 系列（Google）

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| PaLM 540B | 2022 | 否 | Dense | MQA | 2,048 | 并行层 + SwiGLU |
| PaLM 2 | 2023 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 闭源 |

### 2.16 MiniCPM（面壁智能）

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| MiniCPM3-4B | 2024-09 | 是 | Dense 4B | MLA | 32K | 小模型 MLA 验证 |

### 2.17 Grok 系列（xAI）

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Grok-1 | 2024-03 | 是 | MoE 314B | 25% 层 Attention + 8 KV Head | 8K | 层间稀疏 + 8 专家 Top-2 |
| Grok-2 / Grok-3 / Grok-4.x | 2024-2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 闭源 |

---

### 2.18 Phi 系列（Microsoft）

Phi 系列的公开配置显示：早期小模型使用 MHA，Phi-4 起明确转向 GQA。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Phi-3-mini-128k | 2024 | 是 | Dense 3.8B | MHA 32Q/32KV | 128K | LongRope 长上下文 |
| Phi-4 | 2024-12 | 是 | Dense 14B | GQA 40Q/10KV | 16K 配置 | RoPE theta 250K |
| Phi-4-mini | 2025-02 | 是 | Dense 3.8B | GQA 24Q/8KV | 128K | LongRope，共享输入/输出 embedding |

**核验说明**：Phi-4-mini 官方 README 明确写有 grouped-query attention 和 128K context；配置中的 `sliding_window` 字段不应被单独理解为 Mistral 式 SWA，必须以官方 README 为准。

### 2.19 DBRX（Databricks）

DBRX 是 2024 年少数公开 GQA + MoE 配置的开源大模型之一。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| DBRX-Base / Instruct | 2024-03 | 是 | MoE 132B/36B Active | GQA 48Q/8KV | 32K | 40 层，16 专家 Top-4，RoPE theta 500K |

**核验说明**：当前 Databricks 官方 HF 页面无法直接读取配置，本文使用社区镜像中的公开 config 字段进行核验：`n_heads=48`、`attn_config.kv_n_heads=8`、`max_seq_len=32768`。

### 2.20 Nemotron 系列（NVIDIA）

Nemotron 3 是“Mamba-2 + MoE + 少量 GQA”的代表性混合架构。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Nemotron-3-Nano | 2025-12 | 是 | MoE 30B/3.5B Active | 23 Mamba-2 + 23 MoE + 6 GQA | 256K 默认，可扩展 1M | 128+1 专家，6 专家激活 |
| Nemotron-3-Super | 2026-03 | 是 | LatentMoE 120B/12B Active | Mamba-2 + MoE + select Attention | 256K 默认，可扩展 1M | MTP，NVFP4 预训练 |

**核验说明**：Nano 官方 README 明确 52 层中 6 层使用 GQA，Super 官方 README 明确为 LatentMoE + Mamba-2 + Attention hybrid 且上下文最高 1M。

### 2.21 InternLM 系列（上海 AI Lab）

InternLM 的公开路线是 GQA + Dynamic RoPE 长上下文。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| InternLM2.5-7B | 2024 | 是 | Dense 7B | GQA 32Q/8KV | 262,144 | Dynamic RoPE factor 2 |
| InternLM3-8B-Instruct | 2025 | 是 | Dense 8B | GQA 32Q/2KV | 32K | Dynamic RoPE factor 6，RoPE theta 50M |

### 2.22 Baichuan 系列（百川智能）

Baichuan 早期使用独立 MHA 架构，2025 年后的 M 系列直接基于 Qwen 基座。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Baichuan2-7B-Base | 2023 | 是 | Dense 7B | MHA 32Q | 4K | 原生中文预训练 |
| Baichuan-M2-32B | 2025 | 是 | 基于 Qwen2.5-32B | GQA 40Q/8KV | 131,072 | 领域强化 |
| Baichuan-M3-235B | 2026-01 | 是 | 基于 Qwen3-235B-A22B | GQA 64Q/4KV | 40,960 | 128 专家 Top-8 |

**核验说明**：Baichuan M2/M3 的 HF 配置分别为 `qwen2` 与 `qwen3_moe`，Attention 直接继承 Qwen 架构，不属于全新 Attention 设计。

---

### 2.23 Step 系列（阶跃星辰）

Step 系列早期闭源，2026 年起开始公开 MoE 权重。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Step-1 / Step-2 / Step-3 API | 2024-2025 | 否 | 官方未完全披露 | 官方未完全披露 | 产品档位 | 闭源商业模型 |
| Step-3.5-Flash | 2026-02 | 是 | MoE | GQA 类 + SWA | 256K | 开源 MoE 推理档位 |
| Step-3.7-Flash | 2026-05 | 是 | MoE 198B/11B Active | GQA 类 + 512 SWA | 256K | 多模态，1.8B Vision Encoder，FP8/FP4 支持 |

**核验说明**：Step-3.7-Flash 官方模型卡确认 198B 总参、约 11B 激活、256K 上下文；官方配置中 text 部分为 64 heads、512 sliding window、45 层、MoE 中间维度 1280。

### 2.24 Xiaomi MiMo 系列（小米）

MiMo 是 SWA + Global Attention 混合的 1M 上下文 MoE 系列。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| MiMo-Audio-7B | 2025-09 | 是 | 基于 Qwen2 | GQA | 官方未完整披露 | 任意模态语音模型 |
| MiMo-V2-Flash | 2025-12 | 是 | MoE | SWA + Global Attention | 1M | 混合注意力 + MTP |
| MiMo-V2.5-Pro | 2026-04 | 是 | MoE 1.02T/42B Active | SWA + GA 6:1，GQA 128Q/8KV | 1M | KV Cache 减少约 7x，3 层 MTP，27T Token 预训练 |

**核验说明**：MiMo-V2.5-Pro 官方模型卡明确给出 70 层、10 个 Full Attention 层、SWA 窗口 128、QK head dim 192、V head dim 128。

### 2.25 Zamba 系列（Zyphra）

Zamba 是 Mamba2 与共享 Attention 混合的代表性开源系列。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Zamba-7B | 2024 | 是 | Mamba + Attention Hybrid | Mamba + Shared Attention | 4K | 共享 Attention 权重 |
| Zamba2-7B-Instruct-v2 | 2025 | 是 | Mamba2 + Attention Hybrid | Mamba2 + Shared Attention | 4K，可扩展 16K | LoRA 投影差异化共享块 |

**核验说明**：Zamba2-7B 官方配置为 `zamba2`，81 层，32 heads；模型卡明确是 Mamba2 + transformer blocks 混合，并支持通过 `use_long_context=True` 扩展到 16K。

### 2.26 Snowflake Arctic 系列

Arctic 是“Dense 主干 + 大规模 MoE 残差”的代表。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Snowflake Arctic-Instruct | 2024-04 | 是 | Dense-MoE Hybrid 480B | GQA 56Q/8KV | 4K | 10B Dense + 128x3.66B MoE，Top-2 |

**核验说明**：官方配置显示 `model_type=arctic`、35 层、56 heads、8 KV heads、128 experts、Top-2。

---

### 2.27 Hunyuan 系列（腾讯）

Hunyuan 的公开路线是：Dense/MoE 商用模型 -> Hunyuan-A13B 开源 MoE -> Hy3 大规模 MoE。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Hunyuan 商用 API | 2023-2025 | 否 | 官方未完全披露 | 官方未完全披露 | 产品档位 | 闭源 |
| Hunyuan-A13B-Instruct | 2025-06 | 是 | MoE 80B/13B Active | GQA 32Q/8KV | 256K，默认配置 32K | 64 专家，动态 RoPE，量化部署 |
| Hy3 | 2026-07 | 是 | MoE 295B/21B Active | GQA 64Q/8KV | 256K | 192 专家 Top-8，MTP 3.8B 参数 |
| Hy-MT2-30B-A3B | 2026-05 | 是 | MoE 30B/3B Active | GQA 32Q/4KV | 256K | 128 专家 Top-8，33 语言翻译 |

**核验说明**：Hunyuan-A13B 官方模型卡确认 80B 总参/13B 激活、GQA、256K 上下文；Hy3 官方模型卡确认 295B 总参/21B 激活、GQA 64Q/8KV、256K 上下文、192 专家。

---

### 2.28 遗漏分析：已覆盖 / 未覆盖 / 待补充

**已覆盖模型系列**

- 必选系列：Qwen、DeepSeek、GLM、Kimi、MiniMax、Step、Xiaomi MiMo、Llama。
- 额外系列：GPT、Gemini、Claude、Mistral、Mixtral、Yi、Gemma、Falcon、PaLM、MiniCPM、Grok、Phi、DBRX、Nemotron、InternLM、Baichuan、Zamba、Snowflake Arctic、Hunyuan。

**已覆盖核心模块**

- Attention：MHA、MQA、GQA、MLA、SWA、Block Sparse、Linear、Hybrid、Gated DeltaNet、Lightning Attention、Ring Attention、Compressed Sparse Attention、ALiBi Attention、PagedAttention、FlashAttention v1-v4。
- 位置编码：RoPE、YaRN、Dynamic NTK、ALiBi、Partial RoPE、Position Interpolation、LongRoPE、2D Position。
- MoE：ExpertFFN、TopKRouter、MixtureOfExperts、DeepSeekMoE、LatentMoE、ExpertParallelMoE、load-balance loss。
- 系统/工程接口：FlashMLA、SpeculativeDecoder、EagleSpeculator、OnDiskKVStore。
- Transformer 基础：RMSNorm、SwiGLU FFN、FeedForward、TransformerBlock、CausalLMModel、BlockSparseIndexer、AttentionResidual、MultiTokenPredictionHead。

**未覆盖或仍需待补充**

- 真实生产级 FlashMLA / CSA / DSA CUDA kernel。
- 分布式 Ring Attention 多设备通信与调度。
- ALiBi 已接入 CausalLMModel，但仍需生产级 kernel 优化。
- LongRoPE 与 2D Position 已支持 CausalLMModel 配置，仍需官方精确系数与大规模验证。
- Mamba-2 精确选择性扫描 / 并行扫描。
- DSpark / EAGLE 真实投机解码调度。
- PagedAttention 与 KV offload 的生产级内存调度、copy-on-write。
- MoE group GEMM / 真实 multi-node all-reduce，当前为 ExpertParallelMoE 教学接口。
- Step、MiMo、Zamba、Arctic 的完整技术报告级细节仍受公开资料限制，部分参数以官方配置为准。

---

## 三、Attention 机制专题解析

### 3.1 MHA

- 核心原理：每个 Query Head 独立拥有 Key/Value Head，注意力矩阵为 `Q K^T / sqrt(d_k)`。
- 关键参数：`hidden_size`、`num_heads`、`head_dim`。
- 优化目标：表达力最大化，作为所有变体的数学基线。
- 适用场景：中小模型、短上下文、训练资源充足。
- 主要收益：实现简单，训练稳定，多头能捕捉不同关系。
- 主要代价：KV Cache 最大，解码带宽和显存开销随上下文线性增长。

### 3.2 MQA

- 核心原理：所有 Query Head 共享一组 Key/Value，KV Cache 相对 MHA 约为 `1 / num_heads`。
- 代表模型：PaLM、Falcon、ChatGLM2/3。
- 收益：解码吞吐提升明显，KV 显存大幅下降。
- 代价：K/V 信息单一，质量损失在长上下文和复杂任务上更明显。
- 量化口径：PaLM 540B 用 MQA 降低超大模型解码带宽；不同论文未提供统一横评。

### 3.3 GQA

- 核心原理：把 Query Head 分成 `g` 组，每组共享一组 Key/Value；`g=1` 为 MQA，`g=h` 为 MHA。
- 关键参数：`num_heads`、`num_kv_groups`。
- 代表模型：Llama 2/3/4、Qwen2/2.5/3、GLM-4/4.7、Mistral 7B、MiniMax M2/M2.7。
- 收益：KV Cache 降到 MHA 的 `1/g`，质量损失远小于 MQA。
- 代价：仍需要完整注意力计算，长上下文显存仍随序列增长。

### 3.4 MLA

- 核心原理：把 Key/Value 联合压缩到低维潜向量，推理只缓存潜向量和解耦 RoPE Key。
- 关键参数：`kv_lora_rank`、`q_lora_rank`、`qk_rope_head_dim`。
- 代表模型：DeepSeek-V2/V3/V4、GLM-5/5.2、Kimi K2/K3、MiniCPM3、Mistral Small 4。
- 收益：DeepSeek-V2 报告 KV Cache 相对 MHA 对照减少约 93.3%；V4 在 1M 上下文下相对 V3.2 将 KV Cache 降到约 10%（Pro）。
- 代价：实现复杂度高于 GQA，需要矩阵吸收和自定义 kernel；潜空间压缩可能损失极端长距离细节。

### 3.5 SWA

- 核心原理：每个 Query 只关注固定窗口 `w` 内的 Key/Value。
- 代表模型：Mistral 7B、Gemma 2/4 局部层、DeepSeek-V4 的局部分支。
- 收益：计算从 O(n^2) 降为 O(nw)，解码缓存可复用滚动窗口。
- 代价：窗口外长距离依赖丢失，通常需要全局层或压缩记忆补充。

### 3.6 Sparse Attention / Block Sparse Attention

- 核心原理：以 block 为单位选择需要计算的 Key/Value，而不是逐 Token 稀疏。
- 代表模型：MiniMax M3 的 MSA、DeepSeek-V4 的 CSA/HCA、GLM-5 的 DSA。
- 收益：MSA 论文在 1M 上下文报告每 Token 注意力计算降低 28.4x。
- 代价：需要 indexer、block table、Top-k 选择与 kernel 协同设计；稀疏模式错误会直接损失召回。

### 3.7 Linear Attention 与 Hybrid Attention

- 核心原理：用可分解核函数替代 Softmax，使 KV 信息进入固定大小状态。
- 代表模型：Kimi Linear/K3 的 KDA、Qwen3-Next/3.5/3.6/3.8 的 Gated DeltaNet、MiniMax-01 的 Lightning Attention。
- 收益：推理状态 O(1)，不随上下文线性增长；Kimi Linear 报告 KV Cache 最高减少 75%，解码吞吐最高提升约 6x。
- 代价：纯线性注意力表达力弱，通常需要 3:1 或类似比例混合全量注意力层。

### 3.8 FlashAttention v1/v2/v3/v4

- 核心原理：IO-aware 分块注意力，在线 Softmax，不物化 n x n 注意力矩阵。
- 代表用途：所有主流训练框架的基础 kernel。
- 量化：FA2 论文报告相对 FA1 最高约 2x，相对 PyTorch 注意力在 A100 上最高约 9x；FA3 面向 Hopper 异步与低精度，FA4 在本仓库中是教学化 v4 路径。
- 澄清：训练或推理使用 FlashAttention 不等于模型架构本身使用新 Attention 类型。

### 3.9 PagedAttention

- 核心原理：把 KV Cache 切成固定大小物理 block，通过 block table 管理序列。
- 代表用途：vLLM 推理引擎。
- 量化：vLLM 论文报告相对 FasterTransformer、Orca 等系统吞吐提升约 2-4x，KV 显存接近零浪费。
- 澄清：PagedAttention 更多是系统层调度优化，不是模型数学上的新 Attention 变种。

### 3.10 RoPE、YaRN、NTK、ALiBi 与位置编码扩展

- RoPE：把相对位置编码进 Q/K，是 Llama、Qwen、GLM、Mistral 的主流选择。
- YaRN：调整 RoPE base 与温度，Qwen3-Next 用它把 256K 扩展到约 1M。
- NTK-aware / Dynamic NTK：按序列长度动态调整频率，Yi-34B-200K 等使用。
- ALiBi：用线性距离偏置替代位置嵌入，Falcon 使用，适合外推但长距离精度有限。
- Partial / p-RoPE：只旋转部分维度，Gemma 4 和 DeepSeek-V4 用于长上下文稳定。

---

## 四、跨模型横向对比

### 4.1 MHA vs MQA vs GQA

| 维度 | MHA | MQA | GQA |
|------|-----|-----|-----|
| KV Head 数 | `h` | 1 | `g` |
| KV Cache 相对大小 | 1x | `1/h` | `1/g` |
| 表达力 | 最高 | 最低 | 居中 |
| 主流代表 | GPT-3、Llama 1 | PaLM、Falcon | Llama、Qwen、GLM |
| 取舍逻辑 | 训练与质量优先 | 极低解码成本 | 质量和成本平衡 |

### 4.2 GQA 实现差异

- Llama 3/4 和 Qwen 更强调生态统一：全尺寸使用 GQA。
- Mistral 7B 在 GQA 上叠加 4096 SWA，Gemma 2/4 在 GQA 上叠加局部/全局交替。
- MiniMax M2/M2.7 在 GQA 上使用 48Q/8KV 与 RoPE theta 5M，重点服务 200K 上下文。
- GLM-4.7 使用 96Q/8KV，证明大 MoE 也可以继续用 GQA。

### 4.3 MLA 实现差异

- DeepSeek 是 MLA 的“原生产品化”路线，V2/V3/V4 持续复用并加入 DSA/CSA。
- GLM 在轻量模型先用 `kv_lora_rank` 验证，再在 GLM-5/5.2 大规模启用。
- Kimi 把 MLA 与 KDA 线性注意力按 3:1 混合，K3 达到 69 KDA + 24 Gated MLA。
- Mistral Small 4 在 1M 上下文的 MoE 小模型上使用 MLA 类结构，证明该方案不限于超大模型。

### 4.4 FlashAttention 的使用边界

- 训练优化：FlashAttention v1/v2/v3/v4 是主流 kernel，不改变模型权重语义。
- 推理优化：FlashMLA、FP4 indexer cache 等与 MLA/稀疏注意力协同。
- 工程加速：PagedAttention 管理 KV Cache 分配，适合并发服务。

### 4.5 长上下文路线对比

| 路线 | 代表 | 优势 | 代价 |
|------|------|------|------|
| RoPE 缩放 + 长上下文训练 | Llama 3.1、Qwen2.5 | 实现简单，质量稳 | KV 显存高 |
| MLA/KV 压缩 | DeepSeek、Kimi K2 | 显存下降明显 | 需要潜空间与 kernel |
| SWA/局部窗口 | Mistral、Gemma | 计算成本低 | 长距离依赖受限 |
| Block Sparse / DSA / MSA | DeepSeek-V4、GLM-5.2、MiniMax M3 | 长上下文推理成本低 | 选择器质量影响召回 |
| Linear/Hybrid | Qwen3-Next、Kimi K3 | 解码状态固定 | 训练与稳定性复杂 |
| Paged/系统级 | vLLM | 提升吞吐和显存利用率 | 不改变模型数学 |

---

## 五、结构化汇总表

### 表 1：模型系列/版本级汇总表

| 系列 | 版本 | 发布时间 | 开源 | 基础架构 | Attention 核心 | 位置编码/长上下文 | 上下文窗口 | KV/内存优化 | 关键优化 | 来源与可信度 |
|------|------|----------|------|----------|----------------|--------------------|------------|--------------|----------|--------------|
| Qwen | Qwen3.8-2.4T-A95B | 2026-08 | 是 | MoE 2.4T/95B Active | Gated DeltaNet + Gated Attention | RoPE/YaRN | 262K，扩展 1M | 线性状态 + GQA 4 KV | 512 专家 Top-10 | 官方 HF 模型卡，高 |
| Qwen | Qwen3.8-27B | 2026-08 | 是 | Dense 27B | Gated DeltaNet + Gated Attention | RoPE/YaRN | 262K，托管 1M | 线性状态 + GQA 4 KV | 多模态、Agent | 官方 HF 模型卡，高 |
| Qwen | Qwen3-Next | 2025-09 | 是 | MoE 80B/3B Active | Gated DeltaNet + Gated Attention | YaRN | 256K，扩展 1,010,000 | KDA 状态 + GQA 2 KV | 3:1 混合 | 官方 HF 模型卡，高 |
| DeepSeek | V4-Pro-0813 | 2026-08 | 是 | MoE 1.6T/49B Active | MLA + CSA + HCA | Partial RoPE + YaRN | 1M | KV 相对 V3.2 约 10% | DSpark、FP4 indexer | 官方报告/模型卡，高 |
| DeepSeek | V4-Flash-0731 | 2026-07 | 是 | MoE 284B/13B Active | MLA + CSA + HCA | Partial RoPE + YaRN | 1M | KV 相对 V3.2 约 7% | 1M 推理 FLOPs 约 10% | 官方报告/模型卡，高 |
| DeepSeek | V3 | 2024-12 | 是 | MoE 671B/37B Active | MLA | RoPE | 128K | 低秩 KV | DeepSeekMoE | 官方技术报告，高 |
| GLM | GLM-5.2 | 2026-06 | 是 | MoE | MLA + DSA + IndexShare | 位置编码未完整公开 | 1M | `kv_lora_rank=512` | 1M FLOPs 降 2.9x | 官方 HF 模型卡，高 |
| GLM | GLM-5 | 2026-02 | 是 | MoE 744B/40B Active | MLA + DSA | 位置编码未完整公开 | 202K | `kv_lora_rank=512` | DSA | 官方 HF 模型卡，高 |
| GLM | GLM-4.7 | 2025-12 | 是 | MoE | GQA 96Q/8KV | RoPE | 202K | GQA | 大 MoE | 官方 HF 配置，高 |
| Kimi | K3 | 2026-06 | 是 | MoE 2.8T/104B Active | 69 KDA + 24 Gated MLA | 官方技术报告补充 | 1M | KDA 状态 + MLA | AttnRes、Stable LatentMoE | 官方 HF 模型卡，高 |
| Kimi | Linear | 2025-10 | 是 | MoE 48B/3B Active | KDA + Gated MLA 3:1 | RoPE | 1M | KV Cache 最高减少 75% | 解码吞吐最高约 6x | 官方 HF 模型卡，高 |
| Kimi | K2 | 2025-07 | 是 | MoE 1T/32B Active | MLA | RoPE | 256K | MLA | 1T MoE | 官方 HF 模型卡，高 |
| MiniMax | M3 | 2026-06 | 是 | MoE 428B/23B Active | GQA + MSA | RoPE theta 5M | 1M | 稀疏 block 选择 | 1M 下 M2 的 prefill 9x、decode 15x | 官方模型卡 + 论文，高 |
| MiniMax | M2.7 | 2026-04 | 是 | MoE | GQA 48Q/8KV | RoPE theta 5M | 204,800 | GQA | 256 专家 Top-8 | 官方 HF 配置，高 |
| MiniMax | M2 | 2025-10 | 是 | MoE | GQA 48Q/8KV | RoPE theta 5M | 196,608 | GQA | 回归全量注意力 | 官方 HF 配置，高 |
| MiniMax | Text-01 | 2025-01 | 是 | MoE 456B/45.9B Active | Lightning + Softmax Attention | 位置编码随实现 | 1M | 线性状态 | 混合注意力 | 官方技术报告，高 |
| Llama | Llama 4 Maverick | 2025-04 | 是 | MoE 400B/17B Active | GQA + chunked local + NoPE 间隔 | scaled RoPE | 1M | GQA | 128 专家 | Meta 官方模型卡/源码，高 |
| Llama | Llama 4 Scout | 2025-04 | 是 | MoE 109B/17B Active | GQA + chunked local + NoPE 间隔 | scaled RoPE | 10M | GQA | 16 专家 | Meta 官方模型卡/源码，高 |
| Llama | Llama 3.1 | 2024-07 | 是 | Dense 8B-405B | GQA | RoPE base 500K | 128K | GQA | 128K 训练 | 官方模型卡，高 |
| GPT | GPT-5.6 Sol | 2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方基准引用 | 第三方官方模型卡引用，低-中 |
| Gemini | Gemini 3.1 Pro Preview | 2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方基准引用 | 官方模型卡引用，低-中 |
| Claude | Fable 5 / Mythos 5 | 2026-06 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 安全护栏 | Anthropic 官方页面，中 |
| Mistral | Mistral-Small-4 | 2026-01 | 是 | MoE | MLA 类 | RoPE | 1M | `kv_lora_rank=256` | 1M 上下文 | 官方 HF 配置，高 |
| Mistral | Mistral-Large-3 | 2025-12 | 是 | MoE 673B/39B Active | MHA 128Q/128KV | RoPE | 256K | MHA | 128 专家 Top-4 | 官方 params.json，高 |
| Mistral | Mixtral 8x7B | 2023-12 | 是 | MoE 46.7B/12.9B Active | GQA | RoPE | 32K | GQA | 8 专家 Top-2 | 官方论文/模型卡，高 |
| Gemma | Gemma 4 12B | 2026-05 | 是 | Dense 12B | GQA + local/global | p-RoPE | 256K | Unified K/V | 1024 窗口 | 官方 HF 模型卡，高 |
| Gemma | Gemma 2 | 2024-06 | 是 | Dense 2B/9B/27B | GQA + local/global | RoPE | 8K | GQA | 4096 窗口 | 官方技术报告，高 |
| Yi | Yi-34B-200K | 2023-12 | 是 | Dense 34B | GQA | Dynamic NTK | 200K | GQA | RoPE 外推 | 官方模型卡，中-高 |
| Falcon | Falcon 40B | 2023-05 | 是 | Dense 40B | MQA + ALiBi | ALiBi | 2K 级 | MQA | 单 KV Head | 官方模型卡，高 |
| Grok | Grok-1 | 2024-03 | 是 | MoE 314B | 25% 层 Attention | 官方未完整披露 | 8K | 8 KV Head | 8 专家 Top-2 | 官方仓库，中-高 |
| Phi | Phi-4-mini | 2025-02 | 是 | Dense 3.8B | GQA 24Q/8KV | LongRope | 128K | GQA | 128K 上下文 | 官方 HF 模型卡/配置，高 |
| Phi | Phi-4 | 2024-12 | 是 | Dense 14B | GQA 40Q/10KV | RoPE theta 250K | 16K | GQA | 高质量推理数据 | 官方 HF 配置，高 |
| DBRX | DBRX-Instruct | 2024-03 | 是 | MoE 132B/36B Active | GQA 48Q/8KV | RoPE theta 500K | 32K | GQA | 16 专家 Top-4 | 官方开源+社区镜像配置，中-高 |
| Nemotron | Nemotron-3-Super | 2026-03 | 是 | LatentMoE 120B/12B Active | Mamba-2 + MoE + GQA | 默认 256K，可扩展 1M | 256K-1M | Mamba-2 状态 + GQA | MTP、NVFP4 | 官方 HF 模型卡/配置，高 |
| Nemotron | Nemotron-3-Nano | 2025-12 | 是 | MoE 30B/3.5B Active | 23 Mamba-2 + 23 MoE + 6 GQA | 默认 256K，可扩展 1M | 256K-1M | Mamba-2 状态 + GQA | 128+1 专家 | 官方 HF 模型卡/配置，高 |
| InternLM | InternLM3-8B-Instruct | 2025 | 是 | Dense 8B | GQA 32Q/2KV | Dynamic RoPE | 32K | GQA | Dynamic RoPE factor 6 | 官方 HF 配置，高 |
| InternLM | InternLM2.5-7B | 2024 | 是 | Dense 7B | GQA 32Q/8KV | Dynamic RoPE | 262K | GQA | Dynamic RoPE factor 2 | 官方 HF 配置，高 |
| Baichuan | Baichuan-M3-235B | 2026-01 | 是 | 基于 Qwen3-235B-A22B | GQA 64Q/4KV | RoPE theta 5M | 40,960 | GQA | 128 专家 Top-8 | 官方 HF 配置，高 |
| Baichuan | Baichuan-M2-32B | 2025 | 是 | 基于 Qwen2.5-32B | GQA 40Q/8KV | RoPE theta 1M | 131,072 | GQA | 领域强化 | 官方 HF 配置，高 |
| Step | Step-3.7-Flash | 2026-05 | 是 | MoE 198B/11B Active | GQA 类 + SWA | Llama3-style RoPE scaling | 262,144 | GQA + SWA | 多模态，FP8/FP4 | 官方 HF 模型卡/配置，高 |
| Xiaomi MiMo | MiMo-V2.5-Pro | 2026-04 | 是 | MoE 1.02T/42B Active | SWA + GA 6:1 | RoPE theta 10M | 1M | GQA + SWA，KV 减少约 7x | 3 层 MTP | 官方 HF 模型卡/配置，高 |
| Zamba | Zamba2-7B-Instruct-v2 | 2025 | 是 | Mamba2 + Attention Hybrid | Mamba2 + Shared Attention | RoPE 4K，扩展 16K | 4K-16K | SSM 状态 | 共享 Attention + LoRA | 官方 HF 模型卡/配置，高 |
| Snowflake Arctic | Arctic-Instruct | 2024-04 | 是 | Dense-MoE Hybrid 480B | GQA 56Q/8KV | RoPE | 4,096 | GQA | 128 专家 Top-2 | 官方 HF 配置/模型卡，高 |
| Hunyuan | Hy3 | 2026-07 | 是 | MoE 295B/21B Active | GQA 64Q/8KV | RoPE | 262,144 | GQA | 192 专家 Top-8，MTP | 官方 HF 模型卡/配置，高 |
| Hunyuan | Hunyuan-A13B-Instruct | 2025-06 | 是 | MoE 80B/13B Active | GQA 32Q/8KV | Dynamic RoPE | 256K，默认 32K | GQA | 64 专家 | 官方 HF 模型卡/配置，高 |

### 表 2：Attention 类型能力对比表

| Attention 类型 | 代表模型 | 时间/显存特征 | 典型优势 | 典型局限 | 适用场景 |
|----------------|----------|--------------|----------|----------|----------|
| MHA | GPT-3、Llama 1 | 训练 O(n^2)，KV 最大 | 表达力强、实现简单 | 长上下文成本高 | 基线、中小模型 |
| MQA | PaLM、Falcon | KV 约 1/h | 解码带宽最低 | 表达力弱 | 极端解码成本约束 |
| GQA | Llama、Qwen、GLM | KV 约 1/g | 质量与成本平衡 | 注意力仍是 O(n^2) | 主流 Dense/MoE |
| MLA | DeepSeek、Kimi K2、GLM-5 | KV 低秩压缩 | 显存大幅下降 | 实现复杂、需 kernel | 超大 MoE、长上下文 |
| SWA | Mistral 7B、Gemma | 计算 O(nw) | 局部高效 | 长距离受限 | 局部依赖、边缘部署 |
| Block Sparse | MiniMax M3、DeepSeek-V4 | 只计算选中 block | 长上下文成本可控 | 选择器影响召回 | 1M 上下文服务 |
| Linear/Hybrid | Qwen3-Next、Kimi K3 | 解码状态固定 | 解码成本低 | 训练与精度复杂 | 超长上下文、Agent |
| FlashAttention | 主流训练框架 | IO-aware，不物化矩阵 | 训练/推理 kernel 加速 | 不改变模型数学 | 所有 Transformer |
| PagedAttention | vLLM | block 管理 KV | 吞吐提升、显存浪费少 | 系统层优化 | 高并发推理服务 |
| RoPE/YaRN/NTK/ALiBi | Llama、Qwen、Falcon | 位置编码扩展 | 提升外推能力 | 不等同 Attention 升级 | 长上下文训练/推理 |

### 表 3：模型与模块覆盖缺口表

| 模型/模块 | 当前文档是否覆盖 | 当前代码是否覆盖 | 优先级 | 建议实现方式 |
|-----------|------------------|------------------|--------|--------------|
| Step 系列 | 已覆盖 | 不适用 | 中 | 文档继续维护官方配置 |
| Xiaomi MiMo | 已覆盖 | 不适用 | 中 | 文档继续维护官方配置 |
| Zamba 系列 | 已覆盖 | 不适用 | 中 | 文档继续维护官方配置 |
| Snowflake Arctic | 已覆盖 | 不适用 | 中 | 文档继续维护官方配置 |
| Ring Attention | 已覆盖 | 教学实现 | 高 | 增加多设备通信模拟与生产级说明 |
| CSA/HCA/DSA | 已覆盖 | CSA 教学实现 | 高 | 增加学习式 Indexer 与 kernel 接口 |
| FlashMLA | 已覆盖 | 接口模拟 | 高 | 增加真实 CUDA kernel 与缓存布局 |
| 投机解码 | 已覆盖 | 简化接口 | 中 | 增加 EAGLE/DSpark 草稿模型接口 |
| LongRoPE/2D Position | 已覆盖 | 已接入 CausalLMModel | 中 | 官方精确系数与大规模验证 |
| ALiBi 集成 | 已覆盖 | 已接入 CausalLMModel | 中 | 生产级 additive bias kernel 优化 |
| Mamba-2/SSM | 已覆盖 | 简化 SSM | 高 | 增加选择性扫描与并行扫描 |
| KV offload | 已覆盖 | 教学接口 | 中 | 增加分页缓存调度与磁盘缓存策略 |
| MoE 负载均衡 | 已覆盖 | 损失已实现 | 中 | 增加 expert parallelism/group GEMM |

---

## 六、行业趋势与未来判断

### 6.1 为什么越来越多模型从 MHA 迁移到 GQA/MQA/MLA

- 解码阶段是服务成本大头：MHA 的 KV Cache 随层数和上下文线性增长，GQA 直接把 KV Head 压缩到 `1/g`，MLA 进一步压缩潜空间。
- 模型规模越大，KV Cache 占比越显著：1M 上下文下，MHA 即使有 FlashAttention 也无法解决缓存和服务吞吐问题。
- MLA 在 DeepSeek-V2/V3 上经过生产验证后，2026 年被 GLM、Kimi、Mistral 等团队跟进，说明“低秩 KV + 解耦 RoPE”已经从论文变成工程共识。

### 6.2 长上下文时代 Attention 的核心瓶颈

- 训练瓶颈：注意力矩阵计算和上下文并行通信，而不是单纯显存。
- 推理瓶颈：KV Cache 容量、解码带宽、稀疏选择器 kernel 效率。
- 产品瓶颈：1M 上下文能否稳定检索、保持长程推理和降低每 Token 成本。

### 6.3 训练优化与推理优化的重心

- 训练：FlashAttention 系列、上下文并行、长序列数据、位置编码缩放、线性/稀疏注意力训练稳定性。
- 推理：MLA/稀疏/线性压缩、PagedAttention、FP4/FP8、投机解码、on-disk KV、共享 prefix。

### 6.4 未来 1 到 2 年判断

- GQA 仍会是中小模型和部分 MoE 的默认方案，但 MLA 类方案在 1M 上下文和超大 MoE 中会更主流。
- 纯 Softmax Attention 不会消失，但会从“所有层”变成“少量全局层”。
- Sparse Attention 和 Linear Attention 会在超长上下文中重新崛起，但需要更可靠的 indexer、门控状态和 kernel 生态。
- Attention、SSM、线性状态、稀疏检索会进一步融合，边界会越来越模糊。
- 闭源模型将继续不披露架构，但开源模型的官方配置和 kernel 会成为 Attention 研究的主要证据来源。

---

## 七、代码实现方案

### 7.1 新增文件

- `attentionfactory/sliding_window_attention.py`
  - `SlidingWindowAttention`，继承 `BaseAttention`，支持 `window_size`、`num_kv_groups`、`causal`。
- `attentionfactory/block_sparse_attention.py`
  - `BlockSparseAttention`，继承 `BaseAttention`，支持 `block_size`、`top_k`、显式 `block_indices`。
- `attentionfactory/linear_attention.py`
  - `LinearAttention`，继承 `BaseAttention`，支持 `elu/relu/linear` kernel 与因果状态累积。
- `attentionfactory/paged_attention.py`
  - `PagedKVBlockAllocator`、`PagedAttentionCache` 和 `paged_attention`，模拟 block table、稠密 gather 与序列克隆。
- `attentionfactory/positional.py`
  - `RotaryPositionEmbedding`、`YaRNScaledRotaryEmbedding`、`DynamicNTKRotaryEmbedding`、`ALiBiBias` 和工厂函数。
- `attentionfactory/moe.py`
  - `ExpertFFN`、`TopKRouter`、`MixtureOfExperts`、`DeepSeekMoE`、`ExpertParallelMoE`，覆盖 Top-k 路由、共享专家、DeepSeek 风格 MoE 与专家并行模拟。
- `attentionfactory/hybrid_attention.py`
  - `HybridAttention`，按层索引在线性注意力和 GQA 全注意力之间路由，复现 Qwen3-Next/Kimi 的 3:1 混合模式。
- `attentionfactory/gated_delta_net.py`
  - `GatedDeltaNet`，门控 Delta 规则线性注意力的教学实现。
- `attentionfactory/lightning_attention.py`
  - `LightningAttention`，分块线性注意力与 intra-block softmax 的 MiniMax 风格实现。
- `attentionfactory/sparse_indexer.py`
  - `BlockSparseIndexer`，学习式 KV block Top-k 选择器，可与 `BlockSparseAttention` 组合。
- `attentionfactory/latent_moe.py`
  - `LatentMoE`，Nemotron-3 风格的潜空间路由 MoE。
- `attentionfactory/attention_residual.py`
  - `AttentionResidual`，Kimi K3 Attention Residual 的简化教学版本。
- `attentionfactory/multi_token_prediction.py`
  - `MultiTokenPredictionHead`，DeepSeek/Nemotron 风格多 Token 预测头。
- `attentionfactory/ring_attention.py`
  - `ring_attention` 与 `RingAttention`，分块在线 Softmax 精确注意力。
- `attentionfactory/compressed_sparse_attention.py`
  - `CompressedSparseAttention`，CSA 风格 KV 压缩 + block sparse 选择。
- `attentionfactory/alibi_attention.py`
  - `AlibiAttention`，GQA 与 ALiBi additive bias 集成。
- `attentionfactory/flash_mla.py`
  - `FlashMLA`，MLA 推理接口模拟。
- `attentionfactory/speculative.py`
  - `SpeculativeDecoder` 与 `EagleSpeculator`，draft-target 与 hidden-state drafting 投机解码教学接口。
- `attentionfactory/ssm.py`
  - `Mamba2Layer`，简化固定状态 SSM 层。
- `attentionfactory/kv_offload.py`
  - `OnDiskKVStore`，on-disk KV cache 接口模拟。
- `attentionfactory/positional.py` 扩展
  - `LongRoPEScaledRotaryEmbedding`、`TwoDimensionalPositionEmbedding`。
- `attentionfactory/moe.py` 扩展
  - `load_balance_loss`，Top-k MoE 辅助负载均衡损失。
- `attentionfactory/norm.py`
  - `RMSNorm`，Llama/Qwen/Mistral 等模型使用的归一化层。
- `attentionfactory/ffn.py`
  - `SwiGLUFFN` 与 `FeedForward`，覆盖主流 Dense 和 MoE 专家网络。
- `attentionfactory/transformer.py`
  - `TransformerBlock`，组合可插拔 Attention、RMSNorm 和 FFN/MoE。
- `attentionfactory/model.py`
  - `CausalLMModel`，组合 Embedding、位置编码、Transformer Block、RMSNorm 与 LM Head；支持 `alibi`、`longrope`、`2d` 位置配置。
- `attentionfactory/registry.py`
  - `build_attention`、`build_positional_encoding`、`list_attentions`，统一模块选择入口。
- `tests/test_extended_attention.py`
  - 覆盖 shape、梯度、确定性、窗口/块稀疏掩码、PagedAttention 与稠密注意力一致性。
- `tests/test_positional_and_moe.py`
  - 覆盖 RoPE 范数保持、YaRN/NTK 有限性、ALiBi 掩码、路由权重归一化、MoE 与 DeepSeekMoE 梯度。
- `tests/test_blocks_and_hybrid.py`
  - 覆盖 Hybrid Attention 路由、Partial RoPE、Position Interpolation、RMSNorm、SwiGLU、Transformer Block 与 MoE FFN。
- `tests/test_model_registry_gated.py`
  - 覆盖 Gated DeltaNet、注册表、CausalLMModel 的 Dense/MoE/Hybrid 组合与梯度。
- `tests/test_extra_modules.py`
  - 覆盖 Lightning Attention、LatentMoE、Attention Residual、Block Sparse Indexer 与 MTP。
- `tests/test_gap_modules.py`
  - 覆盖 Ring Attention、CSA、ALiBi Attention、FlashMLA、SpeculativeDecoder、EagleSpeculator、Mamba2Layer、LongRoPE/2D、On-Disk KV、ExpertParallelMoE、Paged clone 与 load-balance loss。
- 修改 `attentionfactory/__init__.py` 导出新模块。

### 7.2 设计说明

- Attention 模块继承现有 `BaseAttention`，保持 `forward(hidden_state, attention_mask, return_attention_weights)` 接口；位置编码和 MoE 模块不继承 `BaseAttention`，因为它们不是 Attention 算子本身。
- `BlockSparseAttention` 和 `SlidingWindowAttention` 仍物化 score 矩阵，因此定位是“教学/算法接口版”；真实生产版应由 CUDA kernel 直接跳过未选中 block。
- `PagedAttentionCache` 不实现真实 GPU 内存分页、copy-on-write 或调度策略，只模拟逻辑 block table；生产系统应使用 vLLM 的 PagedAttention。
- `LinearAttention` 使用 `cumsum` 计算因果状态，复杂度为 O(n) 状态更新，但仍是 PyTorch 教学实现，不包含 Gated DeltaNet 的门控与 chunkwise kernel。
- `MixtureOfExperts` 按专家循环执行，便于阅读和测试；生产版本应使用 group GEMM、expert parallelism 和负载均衡调度。
- `PartialRotaryPositionEmbedding`、`PositionInterpolation`、`RMSNorm` 和 `SwiGLUFFN` 是可直接组合的基础模块；YaRN 和 Dynamic NTK 的精确数值仍需与 Transformers 官方实现对比。

### 7.3 运行与测试

```bash
python -m pytest -p no:capture -q
python -m ruff check attentionfactory tests
```

说明：当前机器默认 pytest 9.0.1 在 capture 初始化阶段会触发 macOS readline 相关段错误，使用 `-p no:capture` 可正常运行；这不是测试本身失败。

### 7.4 当前覆盖与待补充清单

**已覆盖代码模块**

- Attention：MHA、MQA、GQA、MLA、SWA、Block Sparse、Linear、Hybrid、Gated DeltaNet、Lightning Attention、Ring Attention、Compressed Sparse Attention、ALiBi Attention、PagedAttention、FlashAttention v1-v4。
- 位置编码：RoPE、YaRN、Dynamic NTK、ALiBi、Partial RoPE、Position Interpolation、LongRoPE、2D Position。
- MoE：ExpertFFN、TopKRouter、MixtureOfExperts、DeepSeekMoE、LatentMoE、load-balance loss。
- 系统/工程接口：FlashMLA、SpeculativeDecoder、OnDiskKVStore。
- Transformer 基础：RMSNorm、SwiGLU FFN、FeedForward、TransformerBlock、CausalLMModel、BlockSparseIndexer、AttentionResidual、MultiTokenPredictionHead。
- 注册表：`build_attention`、`build_positional_encoding`、`list_attentions`。

**已覆盖模型级组合**

- `CausalLMModel` 支持 Dense、MoE、Hybrid Attention、ALiBi、LongRoPE、2D Position、padding/causal mask、tie embeddings。

**未覆盖或待补充**

- 真实生产级 FlashMLA / CSA / DSA CUDA kernel。
- 分布式 Ring Attention 多设备通信。
- Mamba-2 精确选择性扫描 / 并行扫描。
- DSpark / EAGLE 真实投机解码调度。
- PagedAttention / KV offload 生产级内存调度与 copy-on-write。
- MoE expert parallelism / group GEMM。
- LongRoPE 官方精确系数与大规模验证。

**测试基线**

```bash
python -m pytest -p no:capture -q
# 171 passed

python -m ruff check attentionfactory tests
# All checks passed
```

---

## 八、参考资料

### Qwen

- Qwen2 Technical Report: https://arxiv.org/abs/2407.10671
- Qwen2.5 Technical Report: https://arxiv.org/abs/2412.15115
- Qwen3-Next 官方模型卡: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3.8-27B 官方模型卡: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.8-2.4T-A95B 官方模型卡: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B

### DeepSeek

- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeek-V4 Technical Report: https://arxiv.org/abs/2606.19348
- DeepSeek-V4-Pro-0813 官方模型卡: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813

### GLM

- GLM-130B: https://arxiv.org/abs/2210.02414
- GLM-5 官方模型卡: https://huggingface.co/zai-org/GLM-5
- GLM-5.2 官方模型卡: https://huggingface.co/zai-org/GLM-5.2
- IndexShare: https://arxiv.org/abs/2603.12201

### Kimi

- Kimi 初代技术报告: https://arxiv.org/abs/2310.08588
- Kimi Linear 官方模型卡: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- Kimi K3 官方模型卡: https://huggingface.co/moonshotai/Kimi-K3

### MiniMax

- MiniMax-01: https://arxiv.org/abs/2501.08313
- MiniMax Sparse Attention: https://arxiv.org/abs/2606.13392
- MiniMax MSA 仓库: https://github.com/MiniMax-AI/MSA
- MiniMax-M3 官方模型卡: https://huggingface.co/MiniMaxAI/MiniMax-M3

### Llama

- Llama 1: https://arxiv.org/abs/2302.13971
- Llama 2: https://arxiv.org/abs/2307.09288
- Llama 3 Herd: https://arxiv.org/abs/2503.24095
- Llama 4 官方模型卡/源码: https://github.com/meta-llama/llama-models/tree/main/models/llama4

### Mistral / Gemma / 其他

- Mistral 7B: https://arxiv.org/abs/2310.06825
- Mixtral of Experts: https://arxiv.org/abs/2401.04088
- Gemma 2: https://arxiv.org/abs/2408.00118
- Gemma 4 官方模型卡: https://huggingface.co/google/gemma-4-12B-it
- Falcon 模型卡: https://huggingface.co/tiiuae/falcon-40b
- Grok-1 仓库: https://github.com/xai-org/grok-1
- Claude Fable 5 / Mythos 5 官方页面: https://www.anthropic.com/news/claude-fable-5-mythos-5

### Phi / DBRX / Nemotron / InternLM / Baichuan

- Phi-4 官方配置: https://huggingface.co/microsoft/Phi-4
- Phi-4-mini 官方模型卡: https://huggingface.co/microsoft/Phi-4-mini-instruct
- DBRX 官方开源说明: https://www.databricks.com/blog/dbrx-open-source-llm
- DBRX 公开配置镜像: https://huggingface.co/alpindale/dbrx-instruct
- Nemotron-3-Nano 官方模型卡: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- Nemotron-3-Super 官方模型卡: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
- InternLM2.5-7B 官方配置: https://huggingface.co/internlm/internlm2_5-7b
- InternLM3-8B-Instruct 官方配置: https://huggingface.co/internlm/internlm3-8b-instruct
- Baichuan-M2-32B 官方配置: https://huggingface.co/baichuan-inc/Baichuan-M2-32B
- Baichuan-M3-235B 官方配置: https://huggingface.co/baichuan-inc/Baichuan-M3-235B

### Step / MiMo / Zamba / Arctic

- Step-3.7-Flash 官方模型卡: https://huggingface.co/stepfun-ai/Step-3.7-Flash
- Xiaomi MiMo-V2.5-Pro 官方模型卡: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
- Zamba2-7B-Instruct-v2 官方模型卡: https://huggingface.co/Zyphra/Zamba2-7B-Instruct-v2
- Snowflake Arctic-Instruct 官方模型卡: https://huggingface.co/Snowflake/snowflake-arctic-instruct

### Hunyuan

- Hunyuan-A13B-Instruct 官方模型卡: https://huggingface.co/tencent/Hunyuan-A13B-Instruct
- Hy3 官方模型卡: https://huggingface.co/tencent/Hy3
- Hy-MT2-30B-A3B 官方模型卡: https://huggingface.co/tencent/Hy-MT2-30B-A3B

### Attention 机制论文

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- MQA: https://arxiv.org/abs/1911.02150
- GQA: https://arxiv.org/abs/2305.13245
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- PagedAttention: https://arxiv.org/abs/2309.06180
- Ring Attention: https://arxiv.org/abs/2310.01889
- YaRN: https://arxiv.org/abs/2309.00071
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
