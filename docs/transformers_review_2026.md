# 主流大语言模型 Transformer 架构研究综述（截至 2026 年 8 月）

> 范围说明：本文以 Attention 机制为主线，但覆盖完整 Transformer 模型架构的演变——归一化、FFN 与激活、MoE 结构、残差与层间设计、Embedding 与输出头、量化感知训练与多模态融合均在讨论范围内（见第三章）。

> 资料口径：本文优先采用官方 Hugging Face 模型卡、官方技术报告、官方仓库和 arXiv 论文。闭源模型若官方未披露实现细节，统一标注“官方未完全披露”，不把社区推测写成事实。
>
> 量化口径：不同论文和团队使用不同 GPU、序列长度、批大小与量化方式，所有性能数字只在各自来源口径内成立，不能直接横向比较。



## 一、执行摘要

1. **层内 KV 压缩已成为 Attention 主线**：MHA → MQA → GQA → MLA 的演进清晰；截至 2026 年，DeepSeek、GLM、Kimi K3、MiniMax M3、Mistral Small 4 均采用 MLA 或共享 KV 方案；Phi、DBRX、InternLM、Nemotron、Step、MiMo、Zamba、Arctic、Hunyuan 继续验证 GQA、SWA、SSM 与混合架构。
2. **长上下文瓶颈从"能不能训练"转向"能不能低成本推理"**：2024 年主流 128K，2025 年 256K–1M 开源，2026 年 DeepSeek-V4、GLM-5.2、Kimi K3、MiniMax M3 将 1M 上下文推向生产。
3. **稀疏/压缩注意力在本文样本中重新增多**：DeepSeek-V4（CSA+HCA）、GLM-5.2（DSA+IndexShare）、MiniMax M3（MSA）从 KV 数量维度压缩或选择，而非仅依赖窗口注意力；GLM-5.2 模型卡声明 IndexShare 在 1M 上下文降低约 2.9× FLOPs [INDEXSHARE]，独立 IndexCache 论文则报告其 30B/H100 实验的端到端加速 [INDEXCACHE]。
4. **线性注意力在超长上下文中重新受到采用**：Qwen3-Next/3.8、Kimi Linear/K3 的来源记录采用 Gated DeltaNet/KDA 与 GQA/MLA 混合；其中 3:1 是两个系列采用的代表配比，尚不能视为行业统一值。Kimi Linear 来源报告 KV Cache 最高减少约 75%；模型卡/报告最高约 6.3×，但论文 Figure 7 的 batch=1 decode 对照约为 2.2–2.3×，不能把 6.3× 迁移为所有服务场景的端到端加速 [KIMI-LINEAR]。
5. **FlashAttention/PagedAttention 是系统级加速而非新数学形式**：训练默认 FlashAttention 系列，推理服务组合 PagedAttention、FlashMLA、FP4 indexer cache，可与任意架构正交叠加。
6. **GQA 仍是最稳妥的默认基线**：Qwen2.5、Llama 3/4、MiniMax M2/M2.7、GLM-4.7 继续使用 GQA；MLA 更适合超大 MoE 与 1M 上下文高并发场景。
7. **闭源模型透明性显著低于开源**：GPT-5.6 Sol、Claude Fable 5/Mythos 5、Gemini 3.x 架构细节未披露；本文仅将闭源模型列为「官方未完全披露」，不做架构推断。
8. **位置编码与 Attention 强耦合**：RoPE、YaRN、NTK-aware、ALiBi、Partial RoPE（0.25–0.5）、mRope、NoPE 间隔层直接影响长上下文效果，不能把上下文长度单独归因于 Attention 升级。
9. **归一化与 FFN 形成稳定基线**：在已公开配置的 decoder-only 模型中，「RMSNorm + Pre-Norm + SwiGLU」是最常见组合；但 DeepNorm、Sandwich-LN、并行块和 GELU/GeGLU 仍在特定架构中有价值，不应表述为“完全取代”。
10. **MoE 进入细粒度与系统协同阶段**：Switch Top-1 → Mixtral Top-2+辅助损失 → DeepSeekMoE 的细粒度路由、共享专家与无辅助损失偏置。本文样本中 2026 年代表模型覆盖 128–896 路由专家、Top-4–Top-16 激活；这是样本分布，不是行业统一标准。
11. **层间配比成为核心超参**：纯 Attention 时代的固定堆叠已经结束，2026 年主要设计空间是"线性:全量 3:1"、"SWA:全局 6:1"、"1 个稀疏 indexer + 3 个共享"、"首 1–3 层稠密"等层类型配比；DeepSeek-V4 主干末三层实际为 `4,128,4`，`compress_ratios` 末三项 `0,0,0` 属于 3 个 DSpark/MTP 层，不能解释为主干末三层全量注意力。
12. **量化从部署选项前置到训练设计**：FP8 在若干超大模型训练中已常见，FP4/MXFP4 QAT 也开始进入旗舰模型；但 BF16 仍广泛使用，尚不能把 FP8/FP4 称为所有模型的默认。
13. **多模态从外挂走向原生**：Kimi K3（MoonViT-V2 401M）、MiniMax-M3（CLIP + patch merge）、Qwen3.8-27B（27 层视觉塔 + mrope 交错 `mrope_section=[11,11,10]`）均在 2026 年旗舰中内置多模态塔与专属位置编码。
14. **残差/层间路径重新成为设计变量**：DeepSeek-V4 的 Manifold-Constrained Hyper-Connections（mHC，`hc_mult=4`）与 Kimi K3 的 Attention Residuals（AttnRes，`block_size=12`）都显示，除 Pre/Post-Norm 外，跨层信息通路也可成为旗舰模型的试验方向；两者的可复现性与泛化收益仍需更多公开证据。
15. **未来 1–2 年值得跟踪的方向：混合架构 + 量化前置 + 层配比搜索**：GQA 在中小模型/部分 MoE 继续稳固，MLA + 稀疏/线性/SSM 混合有望继续服务 1M 上下文和超大 MoE；AttnRes 类跨层通路、层配比搜索和原生多模态仍需更多可复现结果验证。



## 快速总览（截至 2026-08）

| 系列 | 最新代表 | Attention 核心 | 上下文 | 说明 |
|------|----------|----------------|--------|------|
| Qwen | Qwen3.8-2.4T-A95B | Gated DeltaNet + Gated Attention | 262K，扩展 1M | 512 专家 Top-10 |
| DeepSeek | V4-Pro-0813 | MLA + CSA + HCA | 1M | 来源报告：1M 下 KV 约为 V3.2 的 10% [DEEPSEEK-V4] |
| GLM | GLM-5.2 | MLA + DSA + IndexShare | 1M | 来源报告：IndexShare 降 2.9× FLOPs [INDEXSHARE] |
| Kimi | Kimi K3 | 69 KDA + 24 Gated MLA | 1M | 2.8T/104B Active |
| MiniMax | M3 | GQA + MSA | 1M | 来源报告：相对 M2 的 prefill 9×、decode 15× [MINIMAX-M3] |
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
| Step | Step-3.7-Flash | Full GQA 64Q/8KV + SWA 96Q/8KV（1:3） | 256K | 198B/11B Active，多模态 |
| Xiaomi MiMo | MiMo-V2.5-Pro | SWA + Global Attention 6:1 | 1M | 1.02T/42B Active；来源报告 KV 减少约 7× [MIMO-V25] |
| Zamba | Zamba2-7B | Mamba2 + Shared Attention | 4K，可扩展 16K | SSM/Transformer 混合 |
| Snowflake Arctic | Arctic-Instruct | GQA + Dense-MoE Hybrid | 4K | 480B，128 专家 Top-2 |
| Hunyuan | Hy3 | GQA 64Q/8KV | 256K | 295B/21B Active，192 专家 Top-8 |

**Attention 演进主线**

- 2023 年：GQA 成为开源模型默认选择。
- 2024 年：DeepSeek-V2 提出 MLA，报告中相对 MHA 对照的 KV Cache 约减少 93.3% [DEEPSEEK-V2]。
- 2025 年：Qwen3-Next 和 Kimi Linear 验证 Gated DeltaNet/KDA 线性注意力；Llama 4 和 Gemma 3 验证 MoE + 局部注意力。
- 2026 年：DeepSeek-V4、GLM-5.2、MiniMax M3 把稀疏/压缩注意力推到 1M 上下文生产场景；Kimi K3 达到 2.8T 参数。

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
| Qwen3.8-2.4T-A95B | 2026-08（HF 仓库 08-08，08-12 更新） | 是 | MoE 2.4T/95B Active | Gated DeltaNet + Gated Attention | 262,144 原生，扩展 1M | 512 专家 Top-10，Gated Attention 64Q/4KV |

### 2.2 DeepSeek 系列（深度求索）

DeepSeek 是 MLA 的提出者和主要推动者，2026 年进一步进入“MLA + 压缩稀疏注意力”阶段。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| DeepSeek LLM 7B/67B | 2024-01 | 是 | Dense | MHA | 4K 级 | 标准稠密 Transformer |
| DeepSeek-V2 / V2.5 | 2024-05 / 2024-09 | 是 | MoE 236B/21B Active | MLA | 128K | KV Cache 相对 MHA 对照减少约 93.3% [DEEPSEEK-V2] |
| DeepSeek-V3 | 2024-12 | 是 | MoE 671B/37B Active | MLA | 128K | MLA + DeepSeekMoE + 多 Token 预测 |
| DeepSeek-R1 | 2025-01 | 是 | MoE 671B/37B Active | MLA | 128K | 基于 V3-Base 的强化学习推理 |
| DeepSeek-V3.2 | 2025-09 | 是 | MoE | MLA + DSA | 128K | CSA + HCA，长上下文稀疏化 |
| DeepSeek-V4-Pro | 2026-04 预览，2026-08-13 正式 | 是 | MoE 1.6T/49B Active | MLA 类 + CSA + HCA + SWA 分支 | 1,048,576 | 来源报告：1M 下约 27% 推理 FLOPs、10% KV Cache（相对 V3.2）[DEEPSEEK-V4] |
| DeepSeek-V4-Flash | 2026-04 预览，2026-07-31 更新 | 是 | MoE 284B/13B Active | MLA 类 + CSA + HCA + SWA 分支 | 1,048,576 | 来源报告：1M 下约 10% 推理 FLOPs、7% KV Cache（相对 V3.2）[DEEPSEEK-V4] |

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
| GLM-5.2 | 2026-06 | 是 | MoE | MLA + DSA + IndexShare | 1,048,576 | 来源报告：IndexShare 在 1M 上下文降低 2.9× 每 Token FLOPs [INDEXSHARE] |


### 2.4 Kimi 系列（月之暗面）

Kimi 的演进路线是：长上下文系统工程 -> MLA/MoE -> KDA 线性注意力混合 -> 3T 级混合架构。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Kimi 初代 | 2023-10 | 否 | Dense Transformer | MHA 类 + Ring Attention | 技术报告 128K | 分布式长序列注意力 |
| Moonlight-16B-A3B | 2025-02 | 是 | MoE 16B/3B Active | DeepSeek-V3 类 MLA | 技术报告口径 | 小 MoE 训练效率 |
| Kimi K2 | 2025-07 | 是 | MoE 1T/32B Active | MLA | 256K | 1T 级 MoE + MLA |
| Kimi K2.5 / K2.6 / K2.7 | 2026 | 是 | MoE | MLA | 256K | 多模态扩展，64Q，`kv_lora_rank=512` |
| Kimi Linear | 2025-10 | 是 | MoE 48B/3B Active | KDA + Gated MLA 3:1 | 1M | 来源报告：KV Cache 最高减少 75%，最高约 6.3×；Figure 7 batch=1 decode 约 2.2–2.3× [KIMI-LINEAR] |
| Kimi K3 | 2026-06 | 是 | MoE 2.8T/104B Active | 69 KDA + 24 Gated MLA | 1,048,576 | Attention Residuals，Stable LatentMoE 16/896 专家 |

### 2.5 MiniMax 系列

MiniMax 的路线从“线性注意力 + 全量注意力混合”转向“GQA 基线 + 学习式稀疏注意力”。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| MiniMax-Text-01 / MiniMax-01 | 2025-01 | 是 | MoE 456B/45.9B Active | Lightning Attention + Softmax Attention | 1M | Tiling + Intra-block 线性注意力 |
| MiniMax-M1 | 2025-06/07 | 是 | MoE | 混合架构延续 | 40K/80K | 轻量档位 |
| MiniMax-M2 | 2025-10 | 是 | MoE | GQA 48Q/8KV | 196,608 | 回归全量注意力 |
| MiniMax-M2.5 / M2.7 | 2026 | 是 | MoE | GQA 48Q/8KV | 204,800 | 256 专家 Top-8，RoPE theta 5M |
| MiniMax-M3 | 2026-06 | 是 | MoE 428B/23B Active | GQA + MiniMax Sparse Attention | 1,048,576 | 稀疏选择 + 专用 GPU kernel |


### 2.6 Llama 系列（Meta）

Llama 系列是开源模型从 MHA 走向 GQA，并进一步走向 MoE + 局部注意力的代表性路径。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Llama 1 | 2023-02 | 是 | Dense 7B-65B | MHA | 2K | RoPE + SwiGLU + RMSNorm |
| Llama 2 | 2023-07 | 是 | Dense 7B/13B/70B | 小模型 MHA，70B GQA | 4K | 70B 使用 8 KV Head |
| Llama 3 / 3.1 / 3.2 / 3.3 | 2024 | 是 | Dense 8B-405B | 全尺寸 GQA | 8K-128K | RoPE base 500K，128K 训练 |
| Llama 4 Scout / Maverick | 2025-04 | 是 | MoE | GQA + chunked local attention + NoPE 间隔层 | Scout 10M，Maverick 1M | MoE，QK-Norm，scaled RoPE，attention temperature tuning |

### 2.7 GPT 系列（OpenAI）

GPT 系列早期有公开架构，2023 年后闭源。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| GPT-1 | 2018 | 是 | Dense 117M | MHA | 512 | decoder-only Transformer |
| GPT-2 | 2019 | 是 | Dense 1.5B | MHA | 1,024 | 分层 decoder |
| GPT-3 | 2020 | 是 | Dense 175B | MHA | 2,048 | 标准 MHA，无公开稀疏注意力 |
| GPT-3.5 / GPT-4 / GPT-4o / o1 | 2022-2024 | 否 | 官方未完全披露 | 官方未完全披露 | 4K-128K 产品档位 | 产品级长上下文，架构细节未公开 |
| GPT-5.5 / GPT-5.6 Sol | 2025-2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 从多家官方基准看是 2026 年主力模型，但未披露 Attention |

### 2.8 Gemini 系列（Google）

Gemini 是闭源多模态系列，公开细节远少于 Gemma。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Gemini 1.0 | 2023-12 | 否 | 官方未完全披露 | 官方未完全披露 | API 初期 32K 级 | 多模态 |
| Gemini 1.5 Pro / Flash | 2024 | 否 | 官方未完全披露，含 MoE | 官方未完全披露 | 1M，后 2M，研究 10M | 长上下文基础设施 |
| Gemini 2.0 / 3.x | 2024-2026 | 否 | 官方未完全披露 | 官方未完全披露 | 产品档位持续扩大 | Gemini 3.1 Pro 等名称出现在多个官方基准中 |


### 2.9 Claude 系列（Anthropic）

Claude 全程闭源，Anthropic 未公开 MHA/GQA/MLA 等实现。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Claude 1 / 2 / 2.1 | 2023 | 否 | 官方未完全披露 | 官方未完全披露 | 早期 9K 到 200K | 长会话与系统级上下文处理 |
| Claude 3 / 3.5 / 4.x | 2024-2026 | 否 | 官方未完全披露 | 官方未完全披露 | 产品级 200K 到更大档位 | 多模态与 agentic 能力 |
| Claude Opus 4.8 / Fable 5 / Mythos 5 | 2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | Anthropic 官方页面确认 Fable 5 与 Mythos 5 于 2026-06 发布 |


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

### 2.18 Phi 系列（Microsoft）

Phi 系列的公开配置显示：早期小模型使用 MHA，Phi-4 起明确转向 GQA。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Phi-3-mini-128k | 2024 | 是 | Dense 3.8B | MHA 32Q/32KV | 128K | LongRope 长上下文 |
| Phi-4 | 2024-12 | 是 | Dense 14B | GQA 40Q/10KV | 16K 配置 | RoPE theta 250K |
| Phi-4-mini | 2025-02 | 是 | Dense 3.8B | GQA 24Q/8KV | 128K | LongRope，共享输入/输出 embedding |


### 2.19 DBRX（Databricks）

DBRX 是 2024 年少数公开 GQA + MoE 配置的开源大模型之一。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| DBRX-Base / Instruct | 2024-03 | 是 | MoE 132B/36B Active | GQA 48Q/8KV | 32K | 40 层，16 专家 Top-4，RoPE theta 500K |


### 2.20 Nemotron 系列（NVIDIA）

Nemotron 3 是“Mamba-2 + MoE + 少量 GQA”的代表性混合架构。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Nemotron-3-Nano | 2025-12 | 是 | MoE 30B/3.5B Active | 23 Mamba-2 + 23 MoE + 6 GQA | 256K 默认，可扩展 1M | 128+1 专家，6 专家激活 |
| Nemotron-3-Super | 2026-03 | 是 | LatentMoE 120B/12B Active | Mamba-2 + MoE + select Attention | 256K 默认，可扩展 1M | MTP，NVFP4 预训练 |


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


### 2.23 Step 系列（阶跃星辰）

Step 系列早期闭源，2026 年起开始公开 MoE 权重。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Step-1 / Step-2 / Step-3 API | 2024-2025 | 否 | 官方未完全披露 | 官方未完全披露 | 产品档位 | 闭源商业模型 |
| Step-3.5-Flash | 2026-02 | 是 | MoE | GQA 类 + SWA | 256K | 开源 MoE 推理档位 |
| Step-3.7-Flash | 2026-05 | 是 | MoE 198B/11B Active | 12 Full GQA + 33 SWA，周期 1:3 | 256K | 1.8B Vision Encoder，288 专家 Top-8，3 层 MTP |

### 2.24 Xiaomi MiMo 系列（小米）

MiMo 是 SWA + Global Attention 混合的 1M 上下文 MoE 系列。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| MiMo-Audio-7B | 2025-09 | 是 | 基于 Qwen2 | GQA | 官方未完整披露 | 任意模态语音模型 |
| MiMo-V2-Flash | 2025-12 | 是 | MoE | SWA + Global Attention | 1M | 混合注意力 + MTP |
| MiMo-V2.5-Pro | 2026-04 | 是 | MoE 1.02T/42B Active | SWA + GA 6:1，GQA 128Q/8KV | 1M | KV Cache 减少约 7×，3 层 MTP，27T Token 预训练 |


### 2.25 Zamba 系列（Zyphra）

Zamba 是 Mamba2 与共享 Attention 混合的代表性开源系列。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Zamba-7B | 2024 | 是 | Mamba + Attention Hybrid | Mamba + Shared Attention | 4K | 共享 Attention 权重 |
| Zamba2-7B-Instruct-v2 | 2025 | 是 | Mamba2 + Attention Hybrid | Mamba2 + Shared Attention | 4K，可扩展 16K | LoRA 投影差异化共享块 |


### 2.26 Snowflake Arctic 系列

Arctic 是“Dense 主干 + 大规模 MoE 残差”的代表。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Snowflake Arctic-Instruct | 2024-04 | 是 | Dense-MoE Hybrid 480B | GQA 56Q/8KV | 4K | 10B Dense + 128x3.66B MoE，Top-2 |



### 2.27 Hunyuan 系列（腾讯）

Hunyuan 的公开路线是：Dense/MoE 商用模型 -> Hunyuan-A13B 开源 MoE -> Hy3 大规模 MoE。

| 版本 | 时间 | 开源 | 基础架构 | Attention 核心 | 上下文 | 关键优化 |
|------|------|------|----------|----------------|--------|----------|
| Hunyuan 商用 API | 2023-2025 | 否 | 官方未完全披露 | 官方未完全披露 | 产品档位 | 闭源 |
| Hunyuan-A13B-Instruct | 2025-06 | 是 | MoE 80B/13B Active | GQA 32Q/8KV | 256K，默认配置 32K | 64 专家，动态 RoPE，量化部署 |
| Hy3 | 2026-07 | 是 | MoE 295B/21B Active | GQA 64Q/8KV | 256K | 192 专家 Top-8，MTP 3.8B 参数 |
| Hy-MT2-30B-A3B | 2026-05 | 是 | MoE 30B/3B Active | GQA 32Q/4KV | 256K | 128 专家 Top-8，33 语言翻译 |


### 2.28 遗漏分析：已覆盖 / 未覆盖 / 待补充

| 类别 | 文档状态 | 当前代码状态 | 主要缺口 | 优先级 |
|------|----------|------------------|----------|--------|
| A. Attention | 已覆盖 | 已覆盖 MHA/MQA/GQA/MLA/SWA/BlockSparse/Linear/Hybrid/GDN/KDA/Lightning/Ring/CSA/DSA/MSA/HCA/ALiBi/FlashMLA/FA1-4；Ring 有 distributed reference path | FlashMLA/DSA/MSA/CSA/HCA/KDA 无生产 kernel；FA1-4 与 Ring collective 仍是算法/教学实现 | 高 |
| B. 位置编码 | 已覆盖 | 已覆盖 Learned Absolute/Sinusoidal/T5 Bias/NoPE/RoPE/YaRN/NTK/ALiBi/Partial RoPE/PI/LongRoPE registry/2D/mRoPE | 官方逐频率 LongRoPE preset 与 checkpoint-compatible head-wise mRoPE 仍待核验 | 中 |
| C. FFN/MLP | 已覆盖 | FeedForward/SwiGLU/GeGLU/ReGLU/Clamp-SwiGLU/FFN factory/QATWrapper 已实现 | QAT observer/calibration、KV 专用量化与生产低精度 kernel 仍待补 | 中 |
| D. 归一化 | 已覆盖 | RMSNorm/LayerNorm/DeepNorm/LayerScale/QK-Norm 已实现并接入 MHA/MQA/GQA/MLA 与 TransformerBlock | 生产 fused norm 和大深度数值验证仍待补 | 中 |
| E. 激活函数 | 已覆盖 | 精确 GELU/erf GELU/tanh GELU/ReLU/Squared ReLU/SiLU/Swish/Clipped SiLU 已实现 | 量化数值边界测试仍可补充 | 低 |
| F. 残差/层序 | 已覆盖 | Pre/Post/Sandwich/DeepNorm、Parallel Block、LayerScale、AttnRes 已集成 `TransformerBlock` | AttnRes 仍是逐维门控参考版，未复现跨 block 缓冲/路由 | 中 |
| G. MoE | 已覆盖 | Top-k/共享专家/LatentMoE/Z-Loss/无辅助偏置/Expert Choice/Expert Dropout/Gumbel/EP all-to-all reference 已实现 | Group GEMM、全局 capacity/drop 和真实多节点吞吐仍待补 | 高 |
| H. SSM/Hybrid | 已覆盖 | `Mamba2Layer`、`HybridSSMBlock`、`HybridLayerStack` 已实现 per-channel state、causal conv、norm/residual/FFN 和 layer map | 官方 SSD/fused selective-scan kernel、checkpoint-compatible Zamba 配置仍待补 | 高 |
| I. Embedding/输出头 | 已覆盖 | tied embeddings、`MultiTokenPredictionHead`、`mtp_loss`、序列/Token 分类、奖励、Embedding 头和结构化 `CausalLMOutput` 已实现 | chained MTP 与大规模参数/质量对齐仍待补 | 中 |
| J. 整体架构 | 已覆盖 | Decoder-only、Encoder-only、Encoder-Decoder/CrossAttn、Prefix LM、`MultimodalCausalLM` early/cross fusion 已实现 | 真实 ViT/patch merge、视频变长 batch 和生产多模态 checkpoint 仍待补 | 中-高 |
| K. 训练/推理系统 | 已覆盖 | Paged KV/COW、Tiered HBM/CPU/NVMe reference、Speculative/EAGLE/DSpark、Medusa 并行预测头、FA1-4、QAT 已实现 | 生产 allocator、树验证、异步 DMA、KV reuse、CUDA graph 和量化导出仍待补 | 高 |

## 三、Transformer 整体架构的演变进程与关键组件

Attention 是 2026 年架构差异化的主线，但模型的最终形态同样由非 Attention 组件决定。本章先按时间线梳理整体演变进程（3.1），再按组件逐一分析（3.2-3.9）。配置字段来自第九章列出的来源记录；本轮已核验的 2026 条目见顶部联网记录，其他条目仍保留“待联网复核”。

### 3.1 演变进程：五个阶段（2017-2026）

**阶段一（2017-2019）：原始 Transformer 与预训练范式确立。** 2017 年原始 Transformer 的形态是 Post-LN + LayerNorm + ReLU FFN + 正弦位置编码 + MHA。GPT-2 把 LayerNorm 移到子层输入（Pre-Norm 雏形）并换用 GELU，BERT/GPT 系列确立预训练范式。这一阶段规模在 117M-1.5B，架构问题的主题还是"能不能训练深、能不能 scale"。

**阶段二（2020-2022）：规模化倒逼第一批效率组件。** GPT-3（175B，MHA）证明规模定律后，瓶颈转向推理成本与训练稳定性：PaLM 540B 引入 MQA（KV cache 首次成为架构变量）、SwiGLU 与并行层；RMSNorm 提出；GShard/Switch 把 MoE 带回主线；GLM-130B 用 DeepNorm 支撑深层训练。位置编码侧，RoPE（2021）与 ALiBi（2021）相继提出，为后续长上下文埋下伏笔。

**阶段三（2023-2024）：Llama 范式收敛，GQA 与 MoE 普及。** Llama 把"Pre-Norm RMSNorm + SwiGLU + RoPE"打包成事实标准，Llama 2 70B 与 Llama 3 全尺寸铺开 GQA；Mixtral 8x7B 让开源 MoE 可用；DeepSeek-V2 提出 MLA 与"细粒度专家 + 共享专家"的 DeepSeekMoE——KV 压缩和低秩化正式进入主线。长上下文以 RoPE 缩放（YaRN/NTK/插值）实现，FlashAttention-2 成为训练默认 kernel。2024 年末 DeepSeek-V3 集大成：671B MoE、FP8 训练、MTP、无辅助损失路由（noaux_tc）。

**阶段四（2025）：混合架构与线性注意力复兴。** 纯 Softmax 全量注意力在长上下文下的成本迫使架构分化：Qwen3-Next 与 Kimi Linear 用 Gated DeltaNet/KDA + 3:1 全量混合验证线性注意力的生产可用性；MiniMax-01 验证 Lightning Attention；Llama 4（chunked local + NoPE 间隔层）与 Gemma 3（局部/全局交替 + QK-Norm）探索"少量全局层"路线；MTP 从训练信号变成推理配置项；GPT-OSS 把带 clamp 的 SwiGLU 变体带入开源配置。

**阶段五（2026）：稀疏/压缩生产化、量化前置、原生多模态。** MLA 从 DeepSeek 扩散到 GLM、Kimi、Mistral Small 4；DSA/MSA/CSA 三类学习式稀疏注意力把 1M 上下文推入生产（DeepSeek-V4、GLM-5.2、MiniMax M3）；Qwen3.8 全系列转向 Gated DeltaNet 混合，DeepSeek-V4 以 mHC（`hc_mult=4`）改造残差，Kimi K3 以 2.8T + AttnRes 探索层间路径；FP8 训练成为常见选项、FP4/MXFP4 QAT 前置到训练流程；词表进入 130K-250K 区间；旗舰开源模型普遍原生多模态。闭源模型（GPT-5.6、Claude Fable 5、Gemini 3.x）继续不披露架构。

**驱动力总结。** 五个阶段的更替由四股力量交替主导：训练稳定性（Pre-Norm、QK-Norm、clamp SwiGLU）、训练算力效率（MoE、FP8/FP4）、推理 KV 与带宽成本（MQA -> GQA -> MLA -> 稀疏/线性）、上下文长度（位置编码缩放 -> 窗口 -> 稀疏/压缩）。值得注意的是顺序：每一次组件创新几乎都是先解决上一代主导矛盾的副产物——例如 MLA 解决 KV 成本，却催生了 indexer 与低秩 kernel 的新工程复杂度，进而推动 2026 年的 IndexShare 与 FP4 indexer cache。

### 3.2 归一化：LayerNorm -> RMSNorm -> QK-Norm

- **RMSNorm 在公开 decoder-only 样本中占主导**：自 Llama 普及后，Qwen、DeepSeek、GLM、Kimi、MiniMax、MiMo、Hunyuan 等公开配置记录多采用 RMSNorm；这不等于所有模型或所有子模块都已放弃 LayerNorm。RMSNorm 省去了均值平移，算子更简洁。
- **Pre-Norm 是默认结构**：训练稳定性优于 Post-Norm；GLM-130B 时代曾用 DeepNorm 支撑深层 Post-Norm，2026 年已很少见。
- **QK-Norm 从技巧变成标配**：对 Q/K 投影做归一化以稳定 logits 尺度，Gemma 3、Llama 4 使用；2026 年配置中 MiniMax-M3（`use_qk_norm`，per-head 粒度）与 Hy3（`qk_norm: true`）均默认开启。
- **归一化变体仍在分化**：MiniMax-M3 使用 Gemma 风格的 `(1 + weight)` 缩放（`use_gemma_norm`），说明归一化细节仍在被当作调优空间。

### 3.3 FFN 与激活函数：SwiGLU 一统天下，变体微调

- **SwiGLU 是公开 decoder-only 样本中的常见基线**：继 PaLM/Llama 之后，本文可核对的多数开源配置记录使用 SwiGLU/GLU 族；Gemma、DBRX、Falcon 等系列仍需按具体版本配置区分，不能外推为“所有模型”。中间维度通常在 hidden 的约 2.7-4 倍范围内，但 MoE 专家会按粒度缩小。
- **带 clamp 的 SwiGLU 变体出现**：为抑制极端激活值，GPT-OSS 风格的受限 SwiGLU 进入主流配置——MiniMax-M3 使用 `swiglu_limit=7.0, alpha=1.702`（`hidden_act=swigluoai`），DeepSeek-V4 使用 `swiglu_limit=10.0`。
- **输出门控**：Qwen3.8 系列在注意力输出端加 Swish 门控（`attn_output_gate`），Kimi K3 使用 SiTU-GLU 激活，门控思想正从 FFN 扩散到注意力分支。

### 3.4 MoE 架构演变：细粒度、共享专家与无辅助损失路由

- **规模主线**：Switch/GShard（top-1/2）-> Mixtral 8x7B（8 专家 top-2）-> DeepSeekMoE 确立"细粒度路由专家 + 共享专家"范式。2026 年主流为 256-512 个路由专家、每 Token 激活 4-10 个、1-2 个共享专家（DeepSeek-V4 384/Top-6+1、GLM-5.2 256/Top-8+1、Qwen3.8-2.4T 512/Top-10、MiniMax-M3 128/Top-4+1、MiMo-V2.5 384/Top-8、Hy3 192/Top-8+1）。
- **路由算法换代**：softmax + 辅助均衡损失 -> sigmoid 评分 + `noaux_tc`（无辅助损失的偏置均衡，DeepSeek-V3 提出）+ `routed_scaling_factor`（V4-Pro 2.5、GLM-5.2 2.5、Hy3 2.826）。V4-Pro 使用 `sqrtsoftplus` 评分；Qwen3.8 仍保留 `router_aux_loss_coef=0.001`，说明两条路线并存。
- **结构变体**：Nemotron-3-Super 的 LatentMoE 在潜空间做路由与专家计算；DeepSeek-V4-Pro 的专家权重直接以 FP4 存储（`expert_dtype=fp4`），MoE 与量化深度耦合。
- **首层保持稠密**：`first_k_dense_replace`（前 1-3 层不用 MoE）成为通行做法（DeepSeek、GLM、Hy3、MiniMax-M3 均为 1-3 层），保护底层表征学习。

### 3.5 残差与层间结构：MTP 层与 Attention Residual

- **多 Token 预测（MTP）从训练技巧变成推理架构组件**：DeepSeek-V3 用于训练信号，2026 年配置中 `num_nextn_predict_layers` 普遍出现（DeepSeek-V4-Pro=1、GLM-5.2=1、Hy3=1），MiMo-V2.5-Pro 为 3 层、MiniMax-M3 达 7 个模块；DeepSeek 的 DSpark 则把草稿模块直接并入主模型（目标层 58-60），投机解码与主架构合流。
- **Attention Residual（AttnRes）**：Kimi K3 用学习化的注意力残差替代普通残差相加（`attn_res_block_size=12`），是 Pre-Norm/Post-Norm 之争后层间路径的实验性改动，值得持续观察。
- **Manifold-Constrained Hyper-Connections（mHC）**：DeepSeek-V4 用 `hc_mult=4` 的多副本残差流，在每个 Attention/FFN 子层前后进行受约束的混合；与 AttnRes 都属于残差拓扑变化，但 mHC 的多副本/约束矩阵不能用普通逐维 residual gate 代替。

### 3.6 Embedding 与输出头

- **词表持续膨胀**：多语言与 Agent 场景推动词表从 32K/50K 时代进入 130K-250K 区间——DeepSeek-V4 129,280、MiMo 152,576、GLM-5.2 154,880、Kimi K3 160K、Qwen3.8 达 248,320。
- **权重共享呈现规模分化**：`tie_word_embeddings` 在 Gemma、Phi-4-mini 等小中型模型中仍常见；若干超大模型选择不共享，用更多参数与显存换取输入表示和输出分类器的独立容量。
- **输出头精度保护**：Hy3 开启 `enable_lm_head_fp32`，在低精度存储时代保留输出头的高精度计算。

### 3.7 量化感知训练（QAT）进入架构定义

- **FP8 block-wise**：DeepSeek-V3 之后，FP8 训练在若干超大模型中快速普及（如 V4-Pro 的 FP8 权重与 `ue8m0` 缩放格式、MiMo 的 block-wise FP8）；同期 BF16 配置仍很常见。
- **FP4 推进**：Nemotron-3-Super 使用 NVFP4 预训练；Kimi K3 从 SFT 阶段起做 MXFP4 权重 + MXFP8 激活的量化感知训练；DeepSeek-V4-Pro 专家权重 FP4 存储。量化从"部署压缩"前置为"架构属性"。

### 3.8 多模态从外挂走向原生

- 2026 年的旗舰开源模型多为**原生多模态**：Kimi K3（MoonViT-V2，401M）、MiniMax-M3（CLIP 视觉塔 + patch merge 压缩）、Qwen3.8-27B（27 层视觉塔 + mrope 交错位置编码，`mrope_section=[11,11,10]`）。
- 位置编码随之演化出多模态分支（mrope），与文本侧的 Partial RoPE（0.25-0.5 区间，Qwen3.8/MiniMax-M3/MiMo 配置均为部分旋转）并存。

### 3.9 组件演变速查表

| 组件 | 2017-2022 | 2023-2024 | 2025-2026 主流 |
|------|-----------|-----------|----------------|
| 归一化 | LayerNorm | RMSNorm（Llama 普及） | RMSNorm 为主，QK-Norm 在长训练/大 head_dim 场景增多 |
| FFN 激活 | ReLU/GELU | SwiGLU | SwiGLU + clamp 变体、输出门控 |
| MoE 路由 | Switch top-1 | Mixtral top-2、辅助损失 | sigmoid/sqrtsoftplus + noaux_tc、共享专家、首层稠密 |
| 残差/层间 | Post-Norm | Pre-Norm | Pre-Norm + AttnRes 实验 + MTP 层 |
| Embedding | 32K 词表、共享权重 | 128K 词表 | 130K-250K 词表、不共享、fp32 输出头 |
| 量化 | 训练后 PTQ | FP8 训练（V3） | BF16/FP8 并存，FP4/MXFP4 QAT 起步 |
| 多模态 | 无/外挂 CLIP | ViT 投影外挂 | 原生视觉塔 + mrope |

## 四、Attention 机制专题解析

### 4.0 阅读前的五项强制澄清

1. **FlashAttention 是 IO-aware kernel 优化，不是新的 Attention 数学形式**；它可与 MHA/MQA/GQA/MLA 等组合，是否能直接组合取决于 kernel 对布局和 mask 的支持。
2. **PagedAttention 是推理系统对 KV 内存的分页与调度**，并非模型权重定义的 Attention 变体。
3. **长上下文能力不等于 Attention 升级**；位置编码缩放、长序列训练数据、curriculum、上下文并行和评测方法都可能是决定因素。
4. **FFN、归一化、激活与残差对训练稳定性和质量的影响不亚于 Attention**；只比较 Attention 会遗漏大量计算与优化差异。
5. **MoE 不能只看专家总数**；Top-k、路由分数、capacity/drop 策略、共享专家、负载均衡、EP 通信与 Group GEMM 共同决定质量和墙钟性能。

### 4.1 MHA

- 核心原理：每个 Query Head 独立拥有 Key/Value Head，注意力矩阵为 `Q K^T / sqrt(d_k)`。
- 关键参数：`hidden_size`、`num_heads`、`head_dim`。
- 优化目标：表达力最大化，作为所有变体的数学基线。
- 适用场景：中小模型、短上下文、训练资源充足。
- 主要收益：实现简单，训练稳定，多头能捕捉不同关系。
- 主要代价：KV Cache 最大，解码带宽和显存开销随上下文线性增长。

### 4.2 MQA

- 核心原理：所有 Query Head 共享一组 Key/Value，KV Cache 相对 MHA 约为 `1 / num_heads`。
- 代表模型：PaLM、Falcon、ChatGLM2/3。
- 收益：解码吞吐提升明显，KV 显存大幅下降。
- 代价：K/V 信息单一，质量损失在长上下文和复杂任务上更明显。
- 量化口径：PaLM 540B 用 MQA 降低超大模型解码带宽；不同论文未提供统一横评。

### 4.3 GQA

- 核心原理：设 Query Head 数为 `h`、KV Head 数为 `h_kv`，每 `r=h/h_kv` 个 Query Head 共享一组 K/V；`h_kv=1` 退化为 MQA，`h_kv=h` 退化为 MHA。
- 关键参数：`num_heads`、`num_key_value_heads`、`head_dim`，且通常要求 `h % h_kv == 0`。
- 代表模型：Llama 2/3/4、Qwen2/2.5/3、GLM-4/4.7、Mistral 7B、MiniMax M2/M2.7。
- 收益：不计其他缓存时，KV Cache 约降到 MHA 的 `h_kv/h = 1/r`；质量通常介于 MHA 与 MQA 之间。
- 代价：仍需要完整注意力计算，长上下文显存仍随序列增长。

### 4.4 MLA

- 核心原理：把 Key/Value 联合压缩到低维潜向量，推理只缓存潜向量和解耦 RoPE Key。
- 关键参数：`kv_lora_rank`、`q_lora_rank`、`qk_rope_head_dim`。
- 代表模型：DeepSeek-V2/V3/V4、GLM-5/5.2、Kimi K2/K3、MiniCPM3、Mistral Small 4。
- 收益：DeepSeek-V2 报告 KV Cache 相对 MHA 对照减少约 93.3% [DEEPSEEK-V2]；DeepSeek-V4 论文第 5 页称在 1M 上下文下 Pro 变体相对 V3.2 将 KV Cache 降到约 10% [DEEPSEEK-V4]。
- 代价：实现复杂度高于 GQA，需要矩阵吸收和自定义 kernel；潜空间压缩可能损失极端长距离细节。

### 4.5 SWA

- 核心原理：每个 Query 只关注固定窗口 `w` 内的 Key/Value。
- 代表模型：Mistral 7B、Gemma 2/4 局部层、DeepSeek-V4 的局部分支。
- 收益：计算从 O(n^2) 降为 O(nw)，解码缓存可复用滚动窗口。
- 代价：窗口外长距离依赖丢失，通常需要全局层或压缩记忆补充。

### 4.6 Sparse Attention / Block Sparse Attention

- 核心原理：以 block 为单位选择需要计算的 Key/Value，而不是逐 Token 稀疏。
- 代表模型：MiniMax M3 的 MSA、DeepSeek-V4 的 CSA/HCA、GLM-5 的 DSA。
- 收益：MSA 论文第 12 页在 1M 上下文报告每 Token 注意力计算降低 28.4×；论文第 8–12 页另给出 H800 prefill 14.2×、decode 7.6×，这些是 109B/6B-active 实验模型的 attention/服务口径 [MSA]。DeepSeek-V4 的 CSA/HCA 数字见论文第 9–13 页，不能与 MSA 的 kernel 实验直接横比 [DEEPSEEK-V4]。
- 代价：需要 indexer、block table、Top-k 选择与 kernel 协同设计；稀疏模式错误会直接损失召回。

### 4.7 Linear Attention 与 Hybrid Attention

- 核心原理：用可分解核函数替代 Softmax，使 KV 信息进入固定大小状态。
- 代表模型：Kimi Linear/K3 的 KDA、Qwen3-Next/3.5/3.6/3.8 的 Gated DeltaNet、MiniMax-01 的 Lightning Attention。
- 收益：递推状态对序列长度 `n` 为 O(1)（状态本身仍受 head/feature 维度影响）；Kimi Linear 报告 KV Cache 最高减少 75%，模型卡/报告最高约 6.3×，而论文 Figure 7 的 batch=1 decode 对照约为 2.2–2.3× [KIMI-LINEAR]。
- 代价：纯线性注意力表达力弱，通常需要 3:1 或类似比例混合全量注意力层。

### 4.8 FlashAttention v1/v2/v3/v4

- 核心原理：IO-aware 分块注意力，在线 Softmax，不物化 n x n 注意力矩阵。
- 代表用途：所有主流训练框架的基础 kernel。
- 量化：FA2 论文报告相对 FA1 最高约 2×、相对 PyTorch 注意力在 A100 上最高约 9× [FA2]；FA3 面向 Hopper 异步与低精度，FA4 在本仓库中是教学化 v4 路径。
- 澄清：训练或推理使用 FlashAttention 不等于模型架构本身使用新 Attention 类型。

### 4.9 PagedAttention

- 核心原理：把 KV Cache 切成固定大小物理 block，通过 block table 管理序列。
- 代表用途：vLLM 推理引擎。
- 量化：vLLM 论文报告相对 FasterTransformer、Orca 等对照系统吞吐提升约 2–4×，并将 KV 内存浪费控制在 4% 以下 [PAGEDATTN]；这些数字只在论文的硬件、工作负载与对照设置下成立。
- 澄清：PagedAttention 更多是系统层调度优化，不是模型数学上的新 Attention 变种。

### 4.10 RoPE、YaRN、NTK、ALiBi 与位置编码扩展

- RoPE：把相对位置编码进 Q/K，是 Llama、Qwen、GLM、Mistral 的主流选择。
- YaRN：调整 RoPE base 与温度，Qwen3-Next 用它把 256K 扩展到约 1M。
- NTK-aware / Dynamic NTK：按序列长度动态调整频率，Yi-34B-200K 等使用。
- ALiBi：用线性距离偏置替代位置嵌入，Falcon 使用，适合外推但长距离精度有限。
- Partial / p-RoPE：只旋转部分维度，Gemma 4 和 DeepSeek-V4 用于长上下文稳定。


## 六、结构化汇总表

### 表 1a：模型系列/版本级汇总表

| 系列 | 版本 | 发布时间 | 开源 | 基础架构 | Attention 核心 | 位置编码/长上下文 | 上下文窗口 | KV/内存优化 | 关键优化 | 来源、证据等级与复核状态 |
|------|------|----------|------|----------|----------------|--------------------|------------|--------------|----------|--------------|
| Qwen | Qwen3.8-2.4T-A95B | 2026-08 | 是 | MoE 2.4T/95B Active | Gated DeltaNet + Gated Attention | Partial RoPE (0.25)，theta 1e7 | 262K，扩展 1M | 线性状态 + GQA 4 KV | 512 专家 Top-10 + 1 shared | [QWEN38-24T]；公开确认；已复核 2026-08-18 |
| Qwen | Qwen3.8-27B | 2026-08 | 是 | Dense 27B | Gated DeltaNet + Gated Attention | Partial RoPE (0.25) + mRoPE | 262K，托管 1M | 线性状态 + GQA 4 KV | 27 层视觉塔、多模态 | [QWEN38-27B]；公开确认；已复核 2026-08-18 |
| Qwen | Qwen3-Next | 2025-09 | 是 | MoE 80B/3B Active | Gated DeltaNet + Gated Attention | Partial RoPE (0.25)，YaRN 扩展 | 256K，扩展 1,010,000 | 线性状态 + GQA 2 KV | 3:1 混合 | [QWEN-NEXT]；公开确认 |
| DeepSeek | V4-Pro-0813 | 2026-08 | 是 | MoE 1.6T/49B Active | MLA + CSA + HCA | Partial RoPE + YaRN | 1M | 来源报告：KV 为 V3.2 约 10% [DEEPSEEK-V4] | mHC、DSpark、FP4 indexer | [DEEPSEEK-V4]；论文/报告 + 公开确认；已复核 2026-08-18 |
| DeepSeek | V4-Flash-0731 | 2026-07 | 是 | MoE 284B/13B Active | MLA + CSA + HCA | Partial RoPE + YaRN | 1M | 来源报告：KV 为 V3.2 约 7% | 来源报告：推理 FLOPs 约 10% | [DEEPSEEK-V4]；论文/报告；报告已复核，Flash 发布配置未单独复核 |
| DeepSeek | V3 | 2024-12 | 是 | MoE 671B/37B Active | MLA | RoPE | 128K | 低秩 KV | DeepSeekMoE | [DEEPSEEK-V3]；论文/报告 |
| GLM | GLM-5.2 | 2026-06 | 是 | MoE（总/激活未披露） | MLA + DSA + IndexShare | RoPE theta 8M；indexer interleave | 1M | `kv_lora_rank=512` | 模型卡：FLOPs 降 2.9×，MTP acceptance +20% [INDEXSHARE] | [GLM52]；公开确认；已复核 2026-08-18 |
| GLM | GLM-5 | 2026-02 | 是 | MoE 744B/40B Active | MLA + DSA | RoPE theta 1M | 202,752 | `kv_lora_rank=512` | DSA、前 3 层 dense | [GLM5]；公开确认；已复核 2026-08-18 |
| GLM | GLM-4.7 | 2025-12 | 是 | MoE | GQA 96Q/8KV | RoPE | 202K | GQA | 大 MoE | 官方 HF 配置；公开确认 |
| Kimi | K3 | 2026-06 | 是 | MoE 2.8T/104B Active | 69 KDA + 24 Gated MLA | KDA + MLA NoPE | 1M | KDA 状态 + MLA | AttnRes、Stable LatentMoE | [KIMI-K3]；公开确认；已复核 2026-08-18 |
| Kimi | Linear | 2025-10 | 是 | MoE 48B/3B Active | KDA + Gated MLA 3:1 | RoPE | 1M | 来源报告：KV Cache 最高减少 75% | 最高约 6.3×；Figure 7 batch=1 decode 约 2.2–2.3× | [KIMI-LINEAR]；公开确认 + 论文/报告；已复核 2026-08-18 |
| Kimi | K2 | 2025-07 | 是 | MoE 1T/32B Active | MLA | RoPE | 256K | MLA | 1T MoE | 官方 HF 模型卡；公开确认 |
| MiniMax | M3 | 2026-06 | 是 | MoE 428B/23B Active | GQA + MSA | Partial RoPE 0.5，theta 5M | 1M | 稀疏 block 选择（Top-16×128） | 模型卡：相对 M2 prefill 9×、decode 15× | [MINIMAX-M3] + [MSA]；公开确认 + 论文/报告；已复核 2026-08-18 |
| MiniMax | M2.7 | 2026-04 | 是 | MoE | GQA 48Q/8KV | RoPE theta 5M | 204,800 | GQA | 256 专家 Top-8 | 官方 HF 配置；公开确认；待联网复核 |
| MiniMax | M2 | 2025-10 | 是 | MoE | GQA 48Q/8KV | RoPE theta 5M | 196,608 | GQA | 回归全量注意力 | 官方 HF 配置；公开确认 |
| MiniMax | Text-01 | 2025-01 | 是 | MoE 456B/45.9B Active | Lightning + Softmax Attention | 位置编码随实现 | 1M | 线性状态 | 混合注意力 | [MINIMAX-01]；论文/报告 |
| Llama | Llama 4 Maverick | 2025-04 | 是 | MoE 400B/17B Active | GQA + chunked local + NoPE 间隔 | scaled RoPE | 1M | GQA | 128 专家 | [LLAMA4]；公开确认 |
| Llama | Llama 4 Scout | 2025-04 | 是 | MoE 109B/17B Active | GQA + chunked local + NoPE 间隔 | scaled RoPE | 10M | GQA | 16 专家 | [LLAMA4]；公开确认 |
| Llama | Llama 3.1 | 2024-07 | 是 | Dense 8B-405B | GQA | RoPE base 500K | 128K | GQA | 128K 训练 | 官方模型卡；公开确认 |
| GPT | GPT-5.6 Sol | 2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 第三方基准引用 | 间接引用；合理推断；待联网复核 |
| Gemini | Gemini 3.1 Pro Preview | 2026 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 第三方基准引用 | 间接引用；合理推断；待联网复核 |
| Claude | Fable 5 / Mythos 5 | 2026-06 | 否 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 产品层安全能力 | [CLAUDE-2026]；名称公开确认、架构未披露；待联网复核 |
| Mistral | Mistral-Small-4 | 2026-01 | 是 | MoE | MLA 类 | RoPE | 1M | `kv_lora_rank=256` | 1M 上下文 | 官方 HF 配置；公开确认；待联网复核 |
| Mistral | Mistral-Large-3 | 2025-12 | 是 | MoE 673B/39B Active | MHA 128Q/128KV | RoPE | 256K | MHA | 128 专家 Top-4 | 官方 `params.json`；公开确认 |
| Mistral | Mixtral 8x7B | 2023-12 | 是 | MoE 46.7B/12.9B Active | GQA | RoPE | 32K | GQA | 8 专家 Top-2 | [MIXTRAL]；论文/报告 + 公开确认 |
| Gemma | Gemma 4 12B | 2026-05 | 是 | Dense 12B | GQA + local/global | p-RoPE | 256K | Unified K/V | 1024 窗口 | [GEMMA4]；公开确认；原记录待本轮复核 |
| Gemma | Gemma 2 | 2024-06 | 是 | Dense 2B/9B/27B | GQA + local/global | RoPE | 8K | GQA | 4096 窗口 | [GEMMA2]；论文/报告 |
| Yi | Yi-34B-200K | 2023-12 | 是 | Dense 34B | GQA | Dynamic NTK | 200K | GQA | RoPE 外推 | 官方模型卡；公开确认 |
| Falcon | Falcon 40B | 2023-05 | 是 | Dense 40B | MQA + ALiBi | ALiBi | 2K 级 | MQA | 单 KV Head | [FALCON40B]；公开确认 |
| Grok | Grok-1 | 2024-03 | 是 | MoE 314B | 25% 层 Attention | 官方未完整披露 | 8K | 8 KV Head | 8 专家 Top-2 | [GROK1]；公开确认 |
| Phi | Phi-4-mini | 2025-02 | 是 | Dense 3.8B | GQA 24Q/8KV | LongRope | 128K | GQA | 128K 上下文 | [PHI4-MINI]；公开确认 |
| Phi | Phi-4 | 2024-12 | 是 | Dense 14B | GQA 40Q/10KV | RoPE theta 250K | 16K | GQA | 高质量推理数据 | [PHI4]；公开确认 |
| DBRX | DBRX-Instruct | 2024-03 | 是 | MoE 132B/36B Active | GQA 48Q/8KV | RoPE theta 500K | 32K | GQA | 16 专家 Top-4 | [DBRX] 公开说明 + [DBRX-CONFIG] 社区镜像；配置细节为合理推断 |
| Nemotron | Nemotron-3-Super | 2026-03 | 是 | LatentMoE 120B/12B Active | Mamba-2 + MoE + GQA | 默认 256K，可扩展 1M | 256K-1M | Mamba-2 状态 + GQA | MTP、NVFP4 | [NEMOTRON-SUPER]；公开确认；本轮未复核 |
| Nemotron | Nemotron-3-Nano | 2025-12 | 是 | MoE 30B/3.5B Active | 23 Mamba-2 + 23 MoE + 6 GQA | 默认 256K，可扩展 1M | 256K-1M | Mamba-2 状态 + GQA | 128+1 专家 | [NEMOTRON-NANO]；公开确认 |
| InternLM | InternLM3-8B-Instruct | 2025 | 是 | Dense 8B | GQA 32Q/2KV | Dynamic RoPE | 32K | GQA | Dynamic RoPE factor 6 | [INTERNLM3]；公开确认 |
| InternLM | InternLM2.5-7B | 2024 | 是 | Dense 7B | GQA 32Q/8KV | Dynamic RoPE | 262K | GQA | Dynamic RoPE factor 2 | [INTERNLM25]；公开确认 |
| Baichuan | Baichuan-M3-235B | 2026-01 | 是 | 基于 Qwen3-235B-A22B | GQA 64Q/4KV | RoPE theta 5M | 40,960 | GQA | 128 专家 Top-8 | [BAICHUAN-M3]；公开确认；本轮未复核 |
| Baichuan | Baichuan-M2-32B | 2025 | 是 | 基于 Qwen2.5-32B | GQA 40Q/8KV | RoPE theta 1M | 131,072 | GQA | 领域强化 | [BAICHUAN-M2]；公开确认 |
| Step | Step-3.7-Flash | 2026-05 | 是 | MoE 198B/11B Active | Full GQA 64Q/8KV + SWA 96Q/8KV | Llama3-style scaling | 262,144 | GQA + SWA window 512 | 288 专家 Top-8，3 MTP，多模态 | [STEP37]；公开确认；已复核 2026-08-18 |
| Xiaomi MiMo | MiMo-V2.5-Pro | 2026-04 | 是 | MoE 1.02T/42B Active | SWA + GA 6:1 | Partial RoPE 0.334，theta 10M | 1M | 模型卡：KV 近 7×；128Q/8KV | 3 层 MTP、27T 预训练 | [MIMO-V25]；公开确认；已复核 2026-08-18 |
| Zamba | Zamba2-7B-Instruct-v2 | 2025 | 是 | Mamba2 + Attention Hybrid | Mamba2 + Shared Attention | RoPE 4K，扩展 16K | 4K-16K | SSM 状态 | 共享 Attention + LoRA | [ZAMBA2]；公开确认 |
| Snowflake Arctic | Arctic-Instruct | 2024-04 | 是 | Dense-MoE Hybrid 480B | GQA 56Q/8KV | RoPE | 4,096 | GQA | 128 专家 Top-2 | [ARCTIC]；公开确认 |
| Hunyuan | Hy3 | 2026-07 | 是 | MoE 295B/21B Active | GQA 64Q/8KV | RoPE theta 11.16M | 262,144 | GQA + QK-Norm | 192 专家 Top-8 + 1 shared，MTP | [HY3]；公开确认；已复核 2026-08-18 |
| Hunyuan | Hunyuan-A13B-Instruct | 2025-06 | 是 | MoE 80B/13B Active | GQA 32Q/8KV | Dynamic RoPE | 256K，默认 32K | GQA | 64 专家 | [HUNYUAN-A13B]；公开确认 |

### 表 1b：代表版本 A-K 全模块主表

> 表 1a 负责版本、参数、A/B 与上下文演进；本表是 C-K 的规范化伴随表，并再次列出 A/B 以便独立阅读。只对每个系列的代表版本给出已披露字段，`未披露` 是有效结果，不能从同系列或同规模模型继承。旧版本差异见第二章各系列表。

| 模型 | A Attention | B 位置编码 | C/E FFN 与激活 | D/F 归一化与残差 | G MoE | H SSM/Hybrid | I Embedding/输出头 | J 整体范式 | K 训练/推理系统 | 证据等级/复核 |
|------|-------------|------------|------------------|----------------------|-------|--------------|----------------------|------------|-------------------|---------------|
| Qwen3.8-2.4T | Gated DeltaNet + Gated Attention，3:1 | Partial RoPE 0.25，theta 1e7 | SwiGLU；Attention 输出门控 | RMSNorm；Pre-Norm | 512 / Top-10 / 1；aux loss 0.001 | 线性状态 + 全量 Attention 混合，非 SSM | vocab 248,320；MTP×1 | Decoder-only MoE | BF16；1M 扩展方案 | [QWEN38-24T]；公开确认；已复核 |
| DeepSeek-V4-Pro | MLA 128Q/1KV + CSA/HCA + SWA(128) | YaRN factor 16（64K→1M） | Clamp-SwiGLU（limit 10） | RMSNorm + mHC（`hc_mult=4`，Sinkhorn×20） | 384 / Top-6 / 1；sqrtsoftplus + noaux_tc | 压缩稀疏/稠密 + 局部混合，非 SSM | vocab 129,280；主 MTP×1 + DSpark×3 | Decoder-only MoE | FP8/FP4；DSpark、分层/磁盘 KV | [DEEPSEEK-V4]；论文/报告 + 公开确认；已复核 |
| GLM-5.2 | MLA + DSA；IndexShare 1:3 | RoPE theta 8M | SwiGLU | RMSNorm；前 3 层 dense FFN | 256 / Top-8 / 1；sigmoid + noaux_tc | 稀疏 Attention + 跨层 index 复用，非 SSM | vocab 154,880；MTP×1 | Decoder-only MoE | BF16；共享 indexer；MTP acceptance +20%（卡） | [GLM52] + [INDEXSHARE] + [INDEXCACHE]；已复核 |
| Kimi K3 | 69 KDA + 24 Gated MLA | MLA 分支 NoPE；KDA 自带门控位置动态 | SiTU-GLU（beta 4.0） | RMSNorm；AttnRes block 12 | 896 / Top-16 / 2；LatentMoE、noaux_tc | KDA 线性状态 + MLA，非 SSM | vocab 163,840；无 MTP；401M 视觉塔 | Decoder-only 原生多模态 MoE | MXFP4/MXFP8 QAT | [KIMI-K3]；公开确认；已复核 |
| MiniMax-M3 | GQA 64Q/4KV + MSA（Top-16×128） | Partial RoPE 0.5，theta 5M | Clamp-SwiGLU（limit 7.0） | Gemma 式 RMSNorm + QK-Norm；Pre-Norm | 128 / Top-4 / 1；sigmoid | 学习式稀疏 + 前 3 层全量，非 SSM | vocab 200,064；MTP modules×7；32 层 CLIP | Decoder-only 原生多模态 MoE | BF16；专用 sparse kernel | [MINIMAX-M3] + [MSA]；公开确认 + 论文/报告；已复核 |
| Llama 4 Maverick | GQA + chunked local；NoPE 间隔层 | scaled RoPE + NoPE | GLU 族；精确变体待配置复核 | RMSNorm + QK-Norm；Pre-Norm | 128 专家；路由细节见源码 | Local + Global Attention，非 SSM | 文本/视觉输出细节按版本；未完全披露 | Decoder-only 原生多模态 MoE | 服务 kernel/量化方案非架构固定项 | [LLAMA4]；公开确认 |
| Step-3.7-Flash | 12 Full 64Q/8KV + 33 SWA 96Q/8KV（window 512） | Llama3-style factor 2；Full partial 0.5 | SwiGLU clamp 配置；前 3 层 dense FFN | RMSNorm；head-wise attention gate | 288 / Top-8 / shared dim 1280；sigmoid + bias | SWA:Full 3:1，非 SSM | vocab 128,896；MTP×3；1.8B Vision Encoder | Decoder-only 原生多模态 MoE | BF16/FP8/NVFP4 发布变体 | [STEP37]；公开确认；已复核 |
| MiMo-V2.5-Pro | SWA(128) + Full 6:1，128Q/8KV，QK/V=192/128 | Partial RoPE 0.334，theta 1e7 | SwiGLU | RMSNorm；Pre-Norm | 384 / Top-8 / 未配置 shared；sigmoid + noaux_tc | 60 SWA + 10 Full，非 SSM | vocab 152,576；MTP×3 | Decoder-only MoE | FP8 block；模型卡称 KV 近 7× | [MIMO-V25]；公开确认；已复核 |
| Hy3 | GQA 64Q/8KV | RoPE theta 11,158,840 | SwiGLU | RMSNorm + QK-Norm；Pre-Norm | 192 / Top-8 / 1；sigmoid + 均衡偏置 | 纯 Attention，非 SSM | vocab 120,832；MTP×1（3.8B）；FP32 LM head | Decoder-only MoE | BF16 | [HY3]；公开确认；已复核 |
| Mistral-Small-4 | MLA 类 | RoPE | FFN/激活细节待复核 | 归一化/残差细节待复核 | MoE；专家与路由字段待复核 | 纯 Attention 记录，非 SSM | Embedding/输出头未完整披露 | Decoder-only MoE | 量化/并行非固定架构项 | 第九章来源记录；公开确认；待联网复核 |
| Gemma 4 12B | GQA + local/global；Unified K/V | p-RoPE | Gated FFN；激活细节待复核 | Norm/QK-Norm/残差细节待复核 | Dense 12B | Local + Global Attention，非 SSM | 多模态输入/LM head；tie 细节待复核 | Decoder-only 原生多模态 Dense | 推理量化依发布变体 | [GEMMA4]；公开确认；待联网复核 |
| Nemotron-3-Super | 选择性 GQA 层 | 长上下文位置方案待复核 | MoE FFN；激活细节待复核 | 归一化/残差细节待复核 | LatentMoE 120B/12B Active | Mamba-2 + Attention hybrid | MTP；Embedding tie 未披露 | Decoder-only SSM/Attention/MoE 混合 | NVFP4 预训练记录 | [NEMOTRON-SUPER]；公开确认；待联网复核 |
| Zamba2-7B | 13 个 shared MHA 32Q/32KV block | RoPE；4K，可扩展 16K | GELU Dense FFN | 配置级 norm/residual | 无 | 68 Mamba2 + 13 hybrid；`d_state=64` | vocab 32K；标准 LM head | Decoder-only SSM/Attention 混合 | 共享块 + LoRA 差异化 | [ZAMBA2]；公开确认；已复核 |
| Arctic-Instruct | GQA 56Q/8KV | RoPE 4K | SiLU；Dense + MoE residual FFN | 配置级 norm/residual | 128 / Top-2；capacity/drop | 非 SSM | vocab 32K；标准 LM head | Decoder-only Dense-MoE Hybrid | 专家并行、capacity 管理 | [ARCTIC]；公开确认；已复核 |
| GPT/Gemini/Claude 2026 闭源代表 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 官方未完全披露 | 产品输出接口公开，内部 head 未披露 | 闭源多模态产品；内部范式未披露 | 产品上下文/量化/并行不等于架构披露 | 名称/产品页与架构证据分离；架构不得推断；待联网复核 |

### 表 2：Attention 类型能力对比表

| Attention 类型 | 代表模型 | 时间/显存特征 | 典型优势 | 典型局限 | 适用场景 | 推荐位置/归一化搭配 |
|----------------|----------|--------------|----------|----------|----------|------------------------|
| MHA | GPT-3、Llama 1 | 训练 O(n^2)，KV 最大 | 表达力强、实现简单 | 长上下文成本高 | 基线、中小模型 | RoPE/absolute + Pre-Norm；大 head_dim 可加 QK-Norm |
| MQA | PaLM、Falcon | KV 约为 MHA 的 `1/h` | 解码带宽最低 | 表达力可能受限 | 极端解码成本约束 | RoPE 或 ALiBi + Pre-Norm |
| GQA | Llama、Qwen、GLM | KV 约为 MHA 的 `h_kv/h` | 质量与成本平衡 | 注意力仍是 O(n^2) | 主流 Dense/MoE | RoPE/Partial RoPE + RMSNorm，必须满足 head 分组约束 |
| MLA | DeepSeek、Kimi K2、GLM-5 | KV 低秩压缩 | 显存大幅下降 | 实现复杂、需 kernel | 超大 MoE、长上下文 | 解耦/Partial RoPE + RMSNorm，可加 QK-Norm |
| SWA | Mistral 7B、Gemma | 计算 O(nw) | 局部高效 | 长距离受限 | 局部依赖、边缘部署 | RoPE + Pre-Norm，间隔全局层 |
| Block Sparse | MiniMax M3、DeepSeek-V4 | 只计算选中 block | 长上下文成本可控 | 选择器影响召回 | 1M 上下文服务 | Partial/YaRN RoPE + QK-Norm，保留少量全量层 |
| Linear/Hybrid | Qwen3-Next、Kimi K3 | 解码状态不随 `n` 增长 | 解码成本低 | 训练与精度复杂 | 超长上下文、Agent | Linear 层 NoPE/Partial RoPE，Full 层 RoPE + RMSNorm |
| FlashAttention | 主流训练框架 | IO-aware，不物化矩阵 | 训练/推理 kernel 加速 | 不改变模型数学 | 所有 Transformer | 沿用底层 Attention 的位置与 norm，kernel 需支持相应 bias/mask |
| PagedAttention | vLLM | block 管理 KV | 吞吐提升、显存浪费少 | 系统层优化 | 高并发推理服务 | 与模型位置/norm 无关，需匹配 KV 布局 |
| RoPE/YaRN/NTK/ALiBi | Llama、Qwen、Falcon | 位置编码扩展 | 提升外推能力 | 不等同 Attention 升级 | 长上下文训练/推理 | 本行是位置方案，不是 Attention 数学类型 |

### 表 3：FFN/MLP 类型能力对比表

| FFN 类型 | 数学结构 | 参数/计算特征 | 代表模型 | 典型优势 | 典型局限 | 适用场景 | 推荐激活 |
|----------|----------|---------------|----------|----------|----------|----------|----------|
| 两层 ReLU/GELU FFN | `W2·act(W1x)` | 中间维度约 4x | GPT-2/3、BERT | 简单稳定 | 参数效率一般 | 早期 Dense 模型 | GELU（质量）或 ReLU（简洁） |
| SwiGLU | `W2·(SiLU(W1x)⊗W3x)` | 中间维度约 8/3x | PaLM、Llama 系列、2026 多数公开配置 | 质量/参数平衡 | 三组投影矩阵 | 通用 decoder-only 基线 | SiLU/Swish |
| Clamp-SwiGLU | SwiGLU + 上限裁剪 | 多 clamp 超参 | GPT-OSS、MiniMax-M3、DeepSeek-V4 | 抑制极端激活、利于低精度 | 需调 limit/alpha | 超大/低精度训练 | Clipped SiLU |
| GeGLU/ReGLU | GELU/ReLU 门控 | 同 SwiGLU 族 | T5、GLM-130B | 与 SwiGLU 同源 | 近期新模型样本较少 | Encoder-Decoder/存量架构 | GELU/ReLU |
| MoE-FFN | SwiGLU 专家化 + 路由 | 容量与计算解耦 | Mixtral、DeepSeek、Qwen3.8 | 相同激活计算下容量更大 | 路由与均衡复杂 | 大模型/多领域 | SwiGLU 为常见基线，QAT 时可加 clamp |

### 表 4：归一化与残差设计对比表

| 方案 | 核心结构 | 训练稳定性 | 推理开销 | 代表模型 | 典型优势 | 典型局限 | 适用场景 |
|------|----------|------------|----------|----------|----------|----------|----------|
| Post-LN（LayerNorm） | 子层输出后归一 | 深层不稳 | 无额外开销 | 原始 Transformer、GPT-2 前 | 结构直观 | 深层需精心 warmup | 教学/浅层模型 |
| Pre-LN（LayerNorm） | 子层输入前归一 | 稳定 | 无 | GPT-2 起 | 易于加深 | 残差流不归一 | 2020-2022 主流 |
| Pre-LN（RMSNorm） | 去均值平移的 Pre-LN | 稳定 | 更省 | Llama 起的几乎全部 | 稳定且高效 | 无明显短板 | 2023 年后默认 |
| DeepNorm | Post-LN + 残差缩放 | 超深层稳定 | 无 | GLM-130B | 支撑 100+ 层 Post-LN | 引入缩放超参 | 超深 Post-LN |
| QK-Norm（叠加） | Q/K 投影分别归一 | 抑制 logits 漂移 | 微小 | Gemma 3/4、Llama 4、MiniMax-M3、Hy3 | 长训练更稳 | 不改变主结构 | 大 head_dim 模型 |
| Sandwich-LN | 子层前后双归一 | 更稳 | 双倍 norm 开销 | CogView 等 | 双重保险 | 开销与冗余 | 少数研究模型 |
| AttnRes | 学习化跨层残差门控 | 待验证 | 小 | Kimi K3 | 新增跨层信息路径 | 新设计待观察 | 旗舰实验性配置 |
| Parallel Block | Attention 与 FFN 并行 | 中性 | 省串行时延 | GPT-J、PaLM | 训练吞吐高 | 略损质量 | 大集群训练 |

### 表 5：MoE 架构设计对比表

| MoE 类型 | 路由策略 | 共享专家 | 负载均衡 | 代表模型 | 专家/激活 | 典型优势 | 典型局限 | 适用场景 |
|----------|----------|----------|----------|----------|-----------|----------|----------|----------|
| Switch 式 | softmax Top-1 | 无 | 辅助损失 | Switch | 1/1 | 路由极简 | 掉 token 风险 | 早期研究 |
| Mixtral 式 | softmax Top-2 | 无 | 辅助损失 | Mixtral 8x7B | 8/2 | 开源验证充分 | 粒度粗 | 2024 开源 MoE |
| DeepSeekMoE 细粒度 | sigmoid/sqrtsoftplus + noaux_tc | 1-2 个 | 无辅助损失偏置 | DeepSeek V3/V4、GLM-5.2、K3、MiMo、Hy3 | 256-896 / 4-16 | 容量与均衡兼得 | 实现复杂 | 2026 旗舰默认 |
| 分组路由 | 组内 Top-k | 可选 | 组内均衡 | Qwen 系部分模型 | topk_group 控制 | 限制通信域 | 表达受限 | 专家并行 |
| LatentMoE | 潜空间路由 | 1 个 | 同 DeepSeek 系 | Nemotron-3-Super、Kimi K3（潜维度 3584） | 128+1/6 | 路由维度低、专家更省 | 新方案待验证 | 超大 MoE |

### 表 6：SSM/Hybrid 与混合架构对比表

| 架构类型 | 核心结构 | 复杂度 | 代表模型 | 典型优势 | 典型局限 | 适用场景 | 与 Attention 组合方式 |
|----------|----------|--------|----------|----------|----------|----------|------------------------|
| 纯 Mamba-2 | 选择性扫描 SSM | 线性 | Falcon-Mamba | 无 KV cache | 长距离召回弱 | 边缘/流式推理 | 不组合 |
| Mamba + Attention 交替 | SSM 与共享 Attention 块交替 | 混合 | Zamba2 | 效率与召回兼顾 | 块设计复杂 | 中小模型 | 层交替 |
| Linear + 全量 3:1 | GDN/KDA + 周期性全量层 | 线性为主 | Qwen3-Next/3.8、Kimi Linear/K3 | 解码状态固定 | 训练调参复杂 | 超长上下文 | 每 4 层 1 个全量层 |
| SWA + 全局 | 窗口层 + 间隔全局层 | 近线性 | MiMo-V2.5（约 6:1）、Gemma 4 | 实现简单 | 窗口外依赖弱 | 长上下文 Dense/MoE | 按比例间隔 |
| GQA/MLA + 学习式稀疏 | indexer 选择 KV block | 二次但稀疏 | DeepSeek-V4、GLM-5.2、MiniMax M3 | 召回损失小 | indexer 质量关键 | 1M 上下文生产 | 逐层或共享 indexer |

## 九、参考资料

### Qwen

- [QWEN2] Qwen2 Technical Report: https://arxiv.org/abs/2407.10671
- [QWEN25] Qwen2.5 Technical Report: https://arxiv.org/abs/2412.15115
- [QWEN-NEXT] Qwen3-Next 官方模型卡: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- [QWEN38-27B] Qwen3.8-27B 官方模型卡: https://huggingface.co/Qwen/Qwen3.8-27B
- [QWEN38-24T] Qwen3.8-2.4T-A95B 官方模型卡: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B

### DeepSeek

- [DEEPSEEK-V2] DeepSeek-V2: https://arxiv.org/abs/2405.04434
- [DEEPSEEK-V3] DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- [DEEPSEEK-V4] DeepSeek-V4 Technical Report（pp.5, 9–13, 25；访问 2026-08-18）: https://arxiv.org/abs/2606.19348
- [DEEPSEEK-V4-CARD] DeepSeek-V4-Pro-0813 官方模型卡与 inference 代码: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 ; https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/tree/main/inference

### GLM

- [GLM130B] GLM-130B: https://arxiv.org/abs/2210.02414
- [GLM5] GLM-5 官方模型卡: https://huggingface.co/zai-org/GLM-5
- [GLM52] GLM-5.2 官方模型卡: https://huggingface.co/zai-org/GLM-5.2
- [INDEXSHARE] GLM-5.2 官方模型卡（IndexShare 声明；访问 2026-08-18）: https://huggingface.co/zai-org/GLM-5.2
- [INDEXCACHE] IndexCache（pp.7–10；访问 2026-08-18）: https://arxiv.org/abs/2603.12201

### Kimi

- [KIMI-ORIGIN] Kimi 初代技术报告: https://arxiv.org/abs/2310.08588
- [KIMI-LINEAR] Kimi Linear 论文（arXiv:2510.26692，pp.1, 6, 13, 16；访问 2026-08-18）/官方模型卡: https://arxiv.org/abs/2510.26692 ; https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- [KIMI-K3] Kimi K3 官方模型卡: https://huggingface.co/moonshotai/Kimi-K3

### MiniMax

- [MINIMAX-01] MiniMax-01: https://arxiv.org/abs/2501.08313
- [MSA] MiniMax Sparse Attention（pp.1, 8–12；访问 2026-08-18）: https://arxiv.org/abs/2606.13392
- [MSA-REPO] MiniMax MSA 仓库: https://github.com/MiniMax-AI/MSA
- [MINIMAX-M3] MiniMax-M3 官方模型卡: https://huggingface.co/MiniMaxAI/MiniMax-M3

### Llama

- [LLAMA1] Llama 1: https://arxiv.org/abs/2302.13971
- [LLAMA2] Llama 2: https://arxiv.org/abs/2307.09288
- [LLAMA3] Llama 3 Herd: https://arxiv.org/abs/2503.24095
- [LLAMA4] Llama 4 官方模型卡/源码: https://github.com/meta-llama/llama-models/tree/main/models/llama4

### Mistral / Gemma / 其他

- [MISTRAL7B] Mistral 7B: https://arxiv.org/abs/2310.06825
- [MIXTRAL] Mixtral of Experts: https://arxiv.org/abs/2401.04088
- [GEMMA2] Gemma 2: https://arxiv.org/abs/2408.00118
- [GEMMA4] Gemma 4 官方模型卡: https://huggingface.co/google/gemma-4-12B-it
- [FALCON40B] Falcon 模型卡: https://huggingface.co/tiiuae/falcon-40b
- [GROK1] Grok-1 仓库: https://github.com/xai-org/grok-1
- [CLAUDE-2026] Claude Fable 5 / Mythos 5 官方页面: https://www.anthropic.com/news/claude-fable-5-mythos-5

### Phi / DBRX / Nemotron / InternLM / Baichuan

- [PHI4] Phi-4 官方配置: https://huggingface.co/microsoft/Phi-4
- [PHI4-MINI] Phi-4-mini 官方模型卡: https://huggingface.co/microsoft/Phi-4-mini-instruct
- [DBRX] DBRX 官方开源说明: https://www.databricks.com/blog/dbrx-open-source-llm
- [DBRX-CONFIG] DBRX 公开配置镜像: https://huggingface.co/alpindale/dbrx-instruct
- [NEMOTRON-NANO] Nemotron-3-Nano 官方模型卡: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- [NEMOTRON-SUPER] Nemotron-3-Super 官方模型卡: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
- [INTERNLM25] InternLM2.5-7B 官方配置: https://huggingface.co/internlm/internlm2_5-7b
- [INTERNLM3] InternLM3-8B-Instruct 官方配置: https://huggingface.co/internlm/internlm3-8b-instruct
- [BAICHUAN-M2] Baichuan-M2-32B 官方配置: https://huggingface.co/baichuan-inc/Baichuan-M2-32B
- [BAICHUAN-M3] Baichuan-M3-235B 官方配置: https://huggingface.co/baichuan-inc/Baichuan-M3-235B

### Step / MiMo / Zamba / Arctic

- [STEP37] Step-3.7-Flash 官方模型卡: https://huggingface.co/stepfun-ai/Step-3.7-Flash
- [MIMO-V25] Xiaomi MiMo-V2.5-Pro 官方模型卡: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro
- [ZAMBA2] Zamba2-7B-Instruct-v2 官方模型卡: https://huggingface.co/Zyphra/Zamba2-7B-Instruct-v2
- [ARCTIC] Snowflake Arctic-Instruct 官方模型卡: https://huggingface.co/Snowflake/snowflake-arctic-instruct

### Hunyuan

- [HUNYUAN-A13B] Hunyuan-A13B-Instruct 官方模型卡: https://huggingface.co/tencent/Hunyuan-A13B-Instruct
- [HY3] Hy3 官方模型卡: https://huggingface.co/tencent/Hy3
- [HY-MT2] Hy-MT2-30B-A3B 官方模型卡: https://huggingface.co/tencent/Hy-MT2-30B-A3B

### Attention 机制论文

- [TRANSFORMER] Attention Is All You Need: https://arxiv.org/abs/1706.03762
- [MQA] MQA: https://arxiv.org/abs/1911.02150
- [GQA] GQA: https://arxiv.org/abs/2305.13245
- [FA1] FlashAttention: https://arxiv.org/abs/2205.14135
- [FA2] FlashAttention-2: https://arxiv.org/abs/2307.08691
- [FA3] FlashAttention-3: https://arxiv.org/abs/2407.08608
- [PAGEDATTN] PagedAttention: https://arxiv.org/abs/2309.06180
- [RINGATTN] Ring Attention: https://arxiv.org/abs/2310.01889
- [YARN] YaRN: https://arxiv.org/abs/2309.00071
- [GATED-DELTANET] Gated DeltaNet: https://arxiv.org/abs/2412.06464
