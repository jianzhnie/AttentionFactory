# 主流大语言模型 Attention 架构研究综述（截至 2026 年 8 月）

> 资料口径：本文优先采用官方 Hugging Face 模型卡、官方技术报告、官方仓库和 arXiv 论文。闭源模型若官方未披露实现细节，统一标注“官方未完全披露”，不把社区推测写成事实。
> 量化口径：不同论文和团队使用不同 GPU、序列长度、批大小与量化方式，所有性能数字只在各自来源口径内成立，不能直接横向比较。

---

## 一、执行摘要

1. **层内 KV 压缩已成为主流**：MHA 到 MQA/GQA 再到 MLA 的演进主线非常清晰；截至 2026 年，DeepSeek、GLM、Kimi、MiniMax M3、Mistral Small 4 等均在不同程度使用 MLA 类或共享 KV 方案。
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
| GLM-5.1 | 2026-04 | 是 | MoE | MLA + DSA | 1M | 长时任务能力升级 |
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

<!-- CONTINUE_SERIES -->
