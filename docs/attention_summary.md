# 主流大语言模型 Attention 架构全景总结

> 版本口径：主章节覆盖截至 2024 年末的主流开源与闭源模型；2025 年后已由官方技术报告或开源仓库明确公开的重要架构动作单独放在文末附录，避免与 2024 年事实混淆。
> 量化口径：文中的加速倍数、显存压缩、上下文长度均来自论文、官方模型卡或官方技术博客。不同 GPU、批大小、序列长度下数字不能直接横向比较。

---

## 0. 核心术语速查

| 缩写 | 全称 | 核心含义 |
|------|------|----------|
| MHA | Multi-Head Attention | 标准多头注意力，每个 Query 头都有独立的 Key/Value 头 |
| MQA | Multi-Query Attention | 所有 Query 头共享同一组 Key/Value |
| GQA | Grouped-Query Attention | Query 头分组，每组共享一组 Key/Value |
| MLA | Multi-head Latent Attention | 将 Key/Value 联合压缩为低维潜向量，推理时只缓存潜向量 |
| SWA | Sliding Window Attention | 每个 Token 只关注局部窗口，注意力复杂度从 O(n^2) 降为 O(nw) |
| FA | FlashAttention | IO 感知的精确注意力内核，避免显式物化 n x n 注意力矩阵 |
| PA | PagedAttention | 将 KV Cache 分块管理，降低显存碎片和浪费 |
| Ring Attention | Ring Attention | 将长序列分块到多个设备，以环形通信完成精确全量注意力 |
| SSM | State Space Model | 用固定大小状态替代 KV Cache，解码状态与序列长度无关 |
| RoPE | Rotary Position Embedding | 旋转位置编码，把相对位置信息编码进 Q/K |
| ALiBi | Attention with Linear Biases | 按距离给注意力分数加线性偏置，支持长度外推 |

---

## 1. Attention 变体技术基准

### 1.1 MHA

MHA 是 Transformer 最原始的注意力实现。每个注意力头独立计算 Query、Key、Value，头间可以学习不同的关系模式。

| 维度 | 说明 |
|------|------|
| 典型参数 | n_layers 层，每层 h 个 Query 头，d_k 为每个头的维度 |
| 计算复杂度 | 训练/前向为 O(n^2 d)，解码每步仍需访问全部 KV |
| KV Cache | 每 Token 每层约 2 x h x d_k 个元素 |
| 优化目标 | 表达能力优先，适合作为基线架构 |
| 适用场景 | 模型较小、上下文较短、对显存不敏感的训练场景 |
| 代表模型 | GPT-3、LLaMA 1、DeepSeek-V1、Gemma 1 |

### 1.2 MQA

MQA 由 Shazeer 在 2019 年提出，让所有 Query 头共享同一个 Key/Value 头。它把 KV Cache 从 MHA 的 h 组降为 1 组，代价是注意力表达力下降。

| 维度 | 说明 |
|------|------|
| KV Cache | 相对 MHA 约为 1/h，例如 32 个 Query 头时降为 1/32 |
| 优势 | 解码显存占用和带宽显著下降，容易提高推理吞吐 |
| 代价 | 不同 Query 头无法保留各自的 K/V 信息，长上下文或复杂任务质量下降更明显 |
| 代表模型 | PaLM、Falcon 系列、ChatGLM2/ChatGLM3 |

### 1.3 GQA

GQA 是 MHA 与 MQA 的折中方案：将 h 个 Query 头分成 g 组，每组共享一组 Key/Value。g=1 退化为 MQA，g=h 退化为 MHA。

| 维度 | 说明 |
|------|------|
| KV Cache | 相对 MHA 约为 1/g |
| 常见配置 | LLaMA 2 70B 为 64 Query 头/8 KV 头，Mistral 7B 为 32/8，Qwen2 72B 为 64/8 |
| 质量表现 | GQA 论文认为在多数评测上接近 MHA，同时推理速度比 MHA 更接近 MQA |
| 适用场景 | 需要兼顾质量与解码吞吐的开源 Dense 和 MoE 模型 |
| 代表模型 | LLaMA 2/3、Qwen2/2.5、GLM-4、Mistral、Yi |

### 1.4 SWA

SWA 只允许当前 Token 关注其左侧固定窗口 w 内的 Token，因此不再直接依赖完整前缀。

| 维度 | 说明 |
|------|------|
| 计算复杂度 | O(nwd)，w 远小于 n 时低于标准 O(n^2 d) |
| KV Cache | 窗口内缓存可复用滚动缓冲区，解码显存峰值受窗口限制 |
| 局限 | 丢失窗口外的长距离依赖，需要配合全局注意力层、压缩记忆或系统级长上下文方案 |
| 代表模型 | Mistral 7B、Gemma 2 的局部注意力层 |

### 1.5 FlashAttention、PagedAttention、Ring Attention

这三类不是新的注意力数学形式，而是让 MHA/GQA/MLA 更高效的系统级实现。

| 技术 | 核心机制 | 已公开量化效果 |
|------|----------|----------------|
| FlashAttention | 分块计算 + 在线 Softmax，减少 HBM 读写，不物化 n x n 矩阵 | FlashAttention-2 论文报告相对 FlashAttention-1 最高约 2 倍，相对 PyTorch 注意力在 A100 上最高约 9 倍 |
| FlashAttention-3 | 面向 Hopper 的异步与 Warp 特化版本 | 2024 年公开，论文报告相对 FlashAttention-2 在 H100 上继续提升 |
| PagedAttention | 把 KV Cache 切成固定大小 Block，用非连续显存分配和映射表管理 | vLLM 论文报告相对 FasterTransformer、Orca 等基线系统吞吐提升约 2-4 倍，KV 显存接近零浪费 |
| Ring Attention | 把序列切成多个 block 分布到不同设备，环形传递 block 完成精确注意力 | 论文目标是将每设备注意力内存从随序列长度增长降低到近似恒定；Kimi 初代技术报告将其用于 128K 上下文 |

### 1.6 MLA

MLA 由 DeepSeek-V2 引入，核心不是简单共享 KV Head，而是把 Key 和 Value 联合压缩到低维潜空间。推理时只缓存潜向量和一个用于 RoPE 的小维度的解耦 Key。

| 维度 | 说明 |
|------|------|
| 缓存结构 | 每层每 Token 缓存 c_t^KV 潜向量 + 解耦 RoPE Key |
| 显存收益 | DeepSeek-V2 技术报告口径：KV Cache 相对其 MHA 对照减少约 93.3%，同等显存可支撑约 10 倍上下文 |
| 计算收益 | 通过矩阵吸收把 Key/Value 上投影合并进 Query/Output 投影，避免解码时显式重建高维 K/V |
| 质量表现 | DeepSeek-V2 报告称在多数评测上优于或接近同规模 MHA 基线 |
| 适用场景 | 超大 MoE 模型、长上下文、高并发解码 |
| 代表模型 | DeepSeek-V2/V3、MiniCPM3-4B、Kimi K2 |

### 1.7 Linear Attention、SSM 与混合注意力

线性注意力将 softmax(QK^T) 替换为可分解的核函数，使状态可以用矩阵递推更新。SSM 与线性注意力在“固定状态、避免 KV Cache 线性增长”这一目标上同源。

| 技术 | 推理状态 | 训练复杂度 | 代表模型 |
|------|----------|------------|----------|
| 经典 Linear Attention | 固定大小矩阵，每步 O(1) 更新 | O(n d^2)，但表达力弱于 Softmax Attention | 早期线性 Transformer |
| Lightning Attention | 固定状态 + Intra-block Softmax 补充局部精度 | 分块并行 O(n) 量级 | MiniMax-01 |
| Gated DeltaNet | 门控 + Delta 规则控制状态写入 | 分块并行 | Qwen3-Next |
| SSM/Mamba | 固定大小状态矩阵 | O(n) 量级 | Mamba、Falcon-Mamba |
| Hybrid | 部分层用固定状态，部分层保留 Softmax Attention | 由混合比例决定 | MiniMax-01、Qwen3-Next |

### 1.8 长上下文扩展技术

Attention 类型之外，位置编码和上下文扩展方法也会影响长序列表现。

| 技术 | 机制 | 代表使用 |
|------|------|----------|
| NTK-aware / Dynamic NTK | 按序列长度调整 RoPE base 频率 | Yi-34B-200K 等 |
| YaRN | 结合 NTK 与温度缩放，适配 RoPE 长上下文 | Qwen2/2.5 扩展到 128K |
| LongRoPE | 搜索并复用非均匀 RoPE 缩放系数 | Phi-3-mini-128K 等 |
| 2D Position | 处理文档与块级位置信息 | GLM-4 长文档场景 |
| Ring Attention | 分布式精确注意力 | Kimi 初代 128K |

---

## 2. 指定模型系列

### 2.1 Qwen 系列（阿里巴巴）

Qwen 系列是阿里通义实验室的开源 Dense/MoE 大模型。2024 年最重要的架构分界点是 Qwen2 明确全尺寸采用 GQA。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Qwen 初代 | 2023 | 公开配置未像 Qwen2 那样统一披露 KV Head 结构 | 初代以 8K 级为主 | 基于 Transformer decoder，RoPE + SwiGLU + RMSNorm |
| Qwen1.5 | 2024-01 | 多尺寸统一模型家族，为 Qwen2 的 GQA 路线铺垫 | 多尺寸提升到 32K 级 | 覆盖 0.5B 到 72B，强调小模型部署 |
| Qwen2 | 2024-06 | GQA | 原生 32K，YaRN 扩展到 128K | 全尺寸 GQA；RoPE base 设为 1,000,000；代表性配置如 7B 为 28 Query 头/4 KV 头，72B 为 64 Query 头/8 KV 头 |
| Qwen2.5 | 2024-09 | GQA | 32K，YaRN 扩展到 128K | 延续 Qwen2 架构，提升代码、数学与指令能力 |

**Qwen 系列设计逻辑**

Qwen2 的 GQA 是“先压 KV Head，再用位置编码扩展上下文”的典型路线。相比 LLaMA 2 只在 70B 使用 GQA，Qwen 在 0.5B 到 72B 全系列统一 GQA，这降低了小模型的部署显存门槛，也为 MoE 化和长上下文推理提供了稳定基础。

### 2.2 DeepSeek 系列（深度求索）

DeepSeek 是 2024 年对 Attention 架构影响最大的开源系列之一，核心贡献是 MLA。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| DeepSeek LLM 7B/67B | 2024-01 | MHA | 4K 级 | 标准稠密 Transformer，偏训练稳定与数据质量 |
| DeepSeek-V2 | 2024-05 | MLA | 128K | 236B 总参/21B 激活 MoE；KV Cache 相对 MHA 对照减少 93.3% |
| DeepSeek-V2.5 | 2024-09 | MLA | 128K | 合并 Chat/Code 能力，架构沿用 V2 |
| DeepSeek-V3 | 2024-12 | MLA | 128K | 671B 总参/37B 激活 MoE；沿用 MLA，并加入多 Token 预测等训练优化 |

**MLA 的实现细节**

- K/V 由同一个低维潜向量 c_t^KV 通过上投影生成，避免分别缓存完整 K/V。
- RoPE 不再直接施加在压缩潜向量上，而是使用解耦的 q_t^R / k_t^R，保留位置信息且不破坏矩阵吸收。
- 推理时仅缓存 c_t^KV 和解耦 RoPE Key；计算时把 K/V 上投影吸收进 Query/Output 投影，减少显式重建。
- DeepSeek-V3 继续使用 MLA，说明该方案在 671B 超大 MoE 上完成了生产验证。

### 2.3 GLM 系列（智谱 AI）

GLM 系列的公开演进路线是 MHA -> MQA -> GQA。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| GLM-130B | 2022 | MHA | 2K 级 | 130B Dense；DeepNorm + RoPE + GLU |
| ChatGLM-6B | 2023 | MHA | 2K 级 | 面向中文对话的 Prefix LM 变体 |
| ChatGLM2-6B | 2023-06 | MQA | 8K，可扩展 32K | 用 MQA 降低 KV Cache，是 GLM 系列第一个公开强调解码效率的版本 |
| ChatGLM3-6B | 2023-10 | MQA | 8K，可扩展 32K | 延续 ChatGLM2 架构 |
| GLM-4 | 2024-01 | 官方未完整公开 | 128K | 商用 API 首次提供 128K 长上下文 |
| GLM-4-9B | 2024-06 | GQA | 128K | 开源 9B 版本从 MQA 转为 GQA，兼顾质量与长上下文 |
| GLM-4-Long | 2024 | 官方未完整公开 | API 提供 1M 级长上下文 | 系统级长上下文能力，不等于完整公开注意力配置 |

**GLM 系列设计逻辑**

ChatGLM2 选择 MQA，说明在 6B 规模下团队优先把 KV Cache 压到最低；GLM-4 转向 GQA，说明模型升级到 9B 后需要恢复更多注意力表达力，同时仍保留显存收益。

### 2.4 Kimi 系列（月之暗面）

Kimi 的核心公开差异是长上下文工程，而不是单一 Attention 变体。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Kimi 初代 | 2023-10 | 标准 Transformer Attention + Ring Attention | 技术报告 128K；产品曾以约 20 万汉字长文本为卖点 | 用 Ring Attention 做分布式长序列精确注意力，减少单设备显存压力 |
| Kimi k1.5 | 2025-01 | 官方未完整公开 | 256K 级 | 长上下文强化学习模型，属于 2025 年后动作 |

**Kimi 系列设计逻辑**

Kimi 初代解决的不是“单层 KV 怎么压缩”，而是“128K 序列怎么在有限显存内完成训练和推理”。Ring Attention 把序列切成 block 并在多设备间环形传递，使注意力计算可以按序列长度横向扩展，而不需要把完整 n x n 注意力矩阵放到单卡。

### 2.5 MiniMax 系列

截至 2024 年末，MiniMax 主力商用模型的底层架构未公开；其首次完整公开 Attention 细节发生在 2025 年 1 月开源的 MiniMax-01。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| abab 等商用 API 模型 | 2023-2024 | 官方未公开 | 按产品档位提供 | 不能根据 API 规格推断具体 Attention 类型 |
| MiniMax-01 | 2025-01 | Lightning Attention + Softmax Attention 混合 | 1M | 456B 总参/45.9B 激活；32 个专家；线性注意力与全注意力层交错 |

**MiniMax-01 的设计逻辑**

MiniMax-01 没有走“只压缩 KV Head”的路线，而是把大部分层替换为 Lightning Attention。Lightning Attention 通过 Tiling + Intra-block 方式把线性注意力的训练改成可并行计算，同时保留块内 Softmax Attention，避免线性注意力常见的局部精度损失。最终目标是让 1M 上下文在 MoE 上可用。

### 2.6 LLaMA 系列（Meta）

LLaMA 系列是开源模型从 MHA 走向 GQA 的基准路线。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| LLaMA 1 | 2023-02 | MHA | 2K | 7B/13B/33B/65B，RoPE + SwiGLU + RMSNorm |
| LLaMA 2 | 2023-07 | 7B/13B 为 MHA，70B 为 GQA | 4K | 70B 使用 64 Query 头/8 KV 头 |
| LLaMA 3 | 2024-04 | 全尺寸 GQA | 8K | 8B/70B；128K tokenizer，训练上下文先以 8K 为主 |
| LLaMA 3.1 | 2024-07 | 全尺寸 GQA | 128K | 8B/70B/405B；RoPE base 提升到 500K，支持 128K |
| LLaMA 3.2 | 2024-09 | 全尺寸 GQA | 128K | 新增 1B/3B 文本模型和视觉模型 |

**LLaMA 系列设计逻辑**

LLaMA 2 只在最大模型使用 GQA，说明小模型在当时更担心质量损失；LLaMA 3 开始全尺寸统一 GQA，说明 GQA 在更高质量数据和大规模训练下已经足够成熟。LLaMA 3.1 的长上下文主要依赖 RoPE 频率调整，而不是引入稀疏注意力。

---

## 3. 补充主流系列

### 3.1 GPT 系列（OpenAI）

GPT 系列是最早把 Transformer decoder 用于语言生成的系列之一，但 2023 年后闭源，注意力细节不再公开。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| GPT-1 | 2018 | MHA | 512 | 12 层 decoder-only Transformer |
| GPT-2 | 2019 | MHA | 1,024 | 最大 1.5B，分层 decoder |
| GPT-3 | 2020 | MHA | 2,048 | 175B，96 层/96 头，无公开稀疏注意力 |
| GPT-3.5 / ChatGPT | 2022-2023 | 官方未公开 | 4K，后提供 16K 档位 | 从 GPT-3 系列微调而来 |
| GPT-4 / GPT-4 Turbo | 2023 | 官方未公开 | 8K/32K；Turbo 128K | 技术报告未披露 MHA/GQA/MLA 等细节 |
| GPT-4o | 2024-05 | 官方未公开 | 128K | 多模态统一模型，架构细节未公开 |
| o1 | 2024-09 | 官方未公开 | 默认 128K 级 | 强化学习推理模型，重点在推理时计算 |

**GPT 系列说明**

GPT-4 及后续版本只能确认“使用某种 Transformer 变体”，不能确认使用 GQA 或 MLA。文档中不应把未公开模型写成确定使用 GQA。

### 3.2 Gemini 系列（Google）

Gemini 是 Google 的闭源多模态系列。其技术报告公开了 MoE、长上下文基础设施，但没有公开 Attention 数学形式。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Gemini 1.0 | 2023-12 | 官方未公开 | API 初期公开 32K 级 | 多模态 decoder-only Transformer |
| Gemini 1.5 Pro / Flash | 2024-02 起 | 官方未公开 | 1M，后续提供 2M；研究阶段可到 10M | 官方明确提到 MoE 与长上下文基础设施 |
| Gemini 2.0 Flash | 2024-12 | 官方未公开 | 1M | 多模态 Agent 能力强化，架构细节未公开 |

**Gemini 系列说明**

Gemini 是“产品级长上下文领先，但技术透明度低”的典型。Google 开放模型 Gemma 系列反而提供了可验证的注意力细节。

### 3.3 Claude 系列（Anthropic）

Claude 系列全程闭源，Anthropic 没有公开 MHA/GQA/MLA 等实现。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Claude 1 | 2023-03 起 | 官方未公开 | 早期约 9K 级 | 以安全对齐为核心 |
| Claude 2 | 2023-07 | 官方未公开 | 100K | 长上下文产品化 |
| Claude 2.1 | 2023-11 | 官方未公开 | 200K | 长会话与系统级处理优化 |
| Claude 3 / 3.5 | 2024-03 / 2024-06 | 官方未公开 | 200K | 多模态与推理能力升级，官方未公开 Attention |

**Claude 系列说明**

Claude 的“1M 上下文”等后续能力更多依赖系统级上下文压缩、对话摘要和检索机制，不等于公开了某种新的 Attention 架构。

### 3.4 Mistral 系列（Mistral AI）

Mistral 是开源模型中 GQA + SWA 的代表系列。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Mistral 7B | 2023-09 | GQA + SWA | 官方常见 8K；SWA 窗口 4096 | 32 Query 头/8 KV 头；滚动缓冲区 KV Cache |
| Mixtral 8x7B | 2023-12 | GQA + 全注意力 | 32K | 46.7B 总参/12.9B 激活；8 专家 Top-2；公开配置中不依赖 SWA |
| Mistral Large | 2024-02 | 官方未完整公开 | 32K | 123B 级闭源/API 模型，未给出完整 Attention 配置 |
| Codestral | 2024-05 | GQA 系 | 32K | 22B 代码模型 |
| Mistral NeMo | 2024-07 | GQA 系 | 128K | 12B，NVIDIA 合作优化 |
| Mistral Small | 2024-09 | GQA 系 | 32K | 22B，部署导向 |

**Mistral 系列设计逻辑**

Mistral 7B 证明 SWA 可以用局部窗口降低显存和计算，同时保留较长外推能力。Mixtral 8x7B 则显示：模型规模扩大后，团队选择放弃 SWA，恢复完整 32K 注意力，以避免 MoE + 稀疏注意力叠加带来的质量不确定性。

### 3.5 Gemma 系列（Google）

Gemma 是 Google 的开源小模型系列，其技术报告提供了比 Gemini 更明确的 Attention 细节。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Gemma 1 | 2024-02 | MHA | 8K | 2B/7B；RoPE + RMSNorm + SwiGLU |
| Gemma 2 | 2024-06 | GQA + 局部/全局交替 | 8K | 2B/9B/27B；局部层使用 4096 滑动窗口，全局层保持全量注意力 |
| Gemma 3 | 2025-03 | GQA + 窗口注意力 | 128K | 2025 年后版本，上下文扩展到 128K |

**Gemma 2 的设计逻辑**

Gemma 2 把“每层都做全量注意力”改成局部/全局交替：局部层负责高效处理邻近上下文，全局层保留跨序列信息。这种方案比纯 SWA 更稳，比纯 GQA 更省。

### 3.6 Falcon 系列（TII）

Falcon 是 MQA + ALiBi 的代表系列。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Falcon 7B | 2023-03 | MQA + ALiBi | 2K 级 | 单 KV Head，降低推理缓存 |
| Falcon 40B | 2023-05 | MQA + ALiBi | 2K 级 | 用 ALiBi 做长度外推 |
| Falcon-180B | 2023-09 | MQA 系 | 2K 级 | 大规模 MQA 验证 |
| Falcon-Mamba | 2024-10 | Mamba-2 纯状态空间模型 | 32K | 用固定状态替代 KV Cache，对比 Transformer 基线 |

**Falcon 系列说明**

Falcon 证明 MQA 在大模型上可行，但后来开源主流没有大规模跟随 MQA，而是选择 GQA 或 MLA，主要原因是 MQA 的注意力表达力在更复杂任务上不够稳定。

### 3.7 Yi 系列（01.AI）

Yi 系列是基于 LLaMA 代码风格改造的 GQA 模型。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Yi-6B | 2023-11 | GQA | 4K | 32 Query 头/4 KV 头 |
| Yi-34B | 2023-11 | GQA | 4K | 56 Query 头/8 KV 头 |
| Yi-34B-200K | 2023-12 | GQA + 动态 NTK/LongLoRA 类扩展 | 200K | 通过位置编码外推实现超长上下文 |
| Yi-1.5 | 2024-06 | GQA | 16K 级 | 基于 Yi 的指令微调升级 |

**Yi 系列设计逻辑**

Yi 没有发明新的 Attention，而是把 GQA 和 RoPE 外推组合起来，以较小工程成本获得长上下文。

### 3.8 Grok 系列（xAI）

Grok-1 是 2024 年少数公开了部分层稀疏注意力的超大 MoE。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| Grok-1 | 2024-03 | 部分层 Attention + 8 KV Head | 8K | 314B 总参；64 层中约 25% 的层使用 48 个 Attention Head；8 专家 Top-2 |
| Grok-2 | 2024-08 | 官方未公开 | 官方未公开 | 闭源升级 |

**Grok-1 的设计逻辑**

Grok-1 的“只有 25% 的层使用注意力”属于层间稀疏：大部分层由 MoE 前馈网络承载信息，少量注意力层负责全局关系。这与 2025 年后 Linear/SSM 混合注意力共享“减少全量注意力层比例”的思路一致。

### 3.9 PaLM（Google）

PaLM 系列可作为 MQA 的早期大规模验证。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| PaLM 540B | 2022 | MQA | 2,048 | 540B，并行层 + SwiGLU + MQA |
| PaLM 2 | 2023 | 官方未公开 | 官方未公开 | 后续闭源版本未给出完整 Attention 配置 |

### 3.10 MiniCPM（面壁智能）

MiniCPM 是 2024 年少数在 4B 级直接采用 MLA 的开源系列。

| 版本 | 发布时间 | Attention 架构 | 上下文窗口 | 关键点 |
|------|----------|----------------|------------|--------|
| MiniCPM3-4B | 2024-09 | MLA | 32K | 用 MLA 降低小模型 KV Cache，展示 MLA 不限于超大模型 |

---

## 4. 同类 Attention 的横向对比

### 4.1 GQA 实现差异

| 对比维度 | LLaMA 2/3 | Qwen2/2.5 | Mistral 7B | Yi-34B | GLM-4-9B |
|----------|-----------|-----------|------------|--------|----------|
| 是否全尺寸统一 | 2 只有 70B；3 全尺寸 | 全尺寸统一 | 单尺寸 | 6B/34B 统一 | 9B |
| 代表性 KV Head 比例 | 70B 为 8 KV Head | 7B 为 4，72B 为 8 | 8 KV Head | 8 KV Head | 公开配置未完整披露 |
| 长上下文方式 | RoPE base 500K + 训练 128K | RoPE base 1M + YaRN | SWA 窗口 | 动态 NTK | 原生 128K |
| 是否叠加窗口注意力 | 否 | 否 | 是 | 否 | 否 |

**结论**：GQA 本身只决定 KV Head 数量，真正区分模型的是 RoPE 基频、窗口层、QK-Norm 和上下文训练策略。

### 4.2 MQA 实现差异

| 对比维度 | PaLM | Falcon | ChatGLM2/3 |
|----------|------|--------|------------|
| 规模 | 540B | 7B/40B/180B | 6B |
| 位置编码 | 公开实现以 RoPE 类为主 | ALiBi | RoPE |
| 使用原因 | 超大模型解码带宽 | 低资源推理 | 小模型显存压缩 |

**结论**：MQA 在超大和超小模型上都有应用，但 2024 年开源主流更倾向 GQA 或 MLA。

### 4.3 MLA 实现差异

| 对比维度 | DeepSeek-V2/V3 | MiniCPM3-4B | Kimi K2 |
|----------|----------------|-------------|---------|
| 模型规模 | 236B/671B MoE | 4B Dense | 1T MoE |
| 上下文 | 128K | 32K | 256K |
| 组合方式 | MLA + DeepSeekMoE | MLA + Dense | MLA + MoE |
| 公开时间 | 2024-05/2024-12 | 2024-09 | 2025-07 |

**结论**：MLA 可以从 4B 到 1T 规模复用，核心收益都在“KV Cache 数量级压缩 + 解码带宽下降”。

### 4.4 长上下文路线对比

| 路线 | 代表 | 优势 | 代价 |
|------|------|------|------|
| 增大 RoPE base + 长上下文训练 | LLaMA 3.1、Qwen2.5 | 保留全量注意力，质量稳定 | 训练/推理显存成本高 |
| KV 压缩 | DeepSeek-V2/V3、MiniCPM3 | 显存下降明显，架构透明 | 需要低秩假设与吸收技巧 |
| 窗口注意力 | Mistral 7B、Gemma 2 | 计算成本低 | 长距离依赖受限 |
| 分布式精确注意力 | Kimi 初代 | 不损失注意力精度 | 依赖多机通信与调度 |
| 线性/固定状态混合 | MiniMax-01 | 超长上下文下成本可扩展 | 表达力仍弱于全量 Softmax Attention |
| 系统级 KV 分块 | vLLM/PagedAttention | 直接提高吞吐，兼容现有模型 | 不是模型层新注意力 |

### 4.5 2024 年开源模型选型总趋势

- Dense 小模型：GQA 是默认选择，部分小模型仍保留 MHA。
- 超大 MoE：DeepSeek 选择 MLA，Grok-1 选择层间稀疏 Attention。
- 长上下文：128K 成为 2024 年开源主流，1M 主要出现在闭源产品和 MiniMax-01 这类 2025 年初开源模型中。
- 推理系统：FlashAttention 系列和 PagedAttention 与模型架构正交，几乎所有主流推理引擎都在使用。

---

## 5. 汇总表

| 系列 | 代表版本 | 发布时间 | 基础架构 | Attention 核心类型 | 上下文窗口 | 关键优化点 |
|------|----------|----------|----------|--------------------|------------|------------|
| Qwen | Qwen2.5 | 2024-09 | Transformer Dense 0.5B-72B | GQA | 32K，YaRN 128K | 全尺寸 GQA、RoPE base 1M、SwiGLU/RMSNorm |
| DeepSeek | DeepSeek-V2 | 2024-05 | MoE 236B/21B Active | MLA | 128K | KV Cache 减少约 93.3%，解耦 RoPE，矩阵吸收 |
| DeepSeek | DeepSeek-V3 | 2024-12 | MoE 671B/37B Active | MLA | 128K | MLA + DeepSeekMoE + 多 Token 预测 |
| GLM | ChatGLM2/3 | 2023 | Transformer Dense 6B | MQA | 8K，扩展 32K | MQA 压 KV Cache |
| GLM | GLM-4-9B | 2024-06 | Transformer Dense 9B | GQA | 128K | MQA 转 GQA，长上下文 |
| Kimi | Kimi 初代 | 2023-10 | Transformer Dense | MHA 类 + Ring Attention | 技术报告 128K | 分布式长序列精确注意力 |
| MiniMax | MiniMax-01 | 2025-01 | MoE 456B/45.9B Active | Lightning Attention + Softmax Attention | 1M | 线性/全量混合，Tiling + Intra-block |
| LLaMA | LLaMA 1 | 2023-02 | Transformer Dense 7B-65B | MHA | 2K | 标准 decoder-only |
| LLaMA | LLaMA 2 70B | 2023-07 | Transformer Dense 70B | GQA | 4K | 64 Query 头/8 KV 头 |
| LLaMA | LLaMA 3.1 | 2024-07 | Transformer Dense/MoE 8B-405B | GQA | 128K | RoPE base 500K |
| GPT | GPT-3 | 2020 | Transformer Dense 175B | MHA | 2,048 | 标准 MHA |
| GPT | GPT-4o | 2024-05 | 官方未公开 | 官方未公开 | 128K | 多模态，闭源 |
| Gemini | Gemini 1.5 Pro | 2024-02 | 官方未公开，含 MoE | 官方未公开 | 1M，后 2M | 长上下文基础设施 |
| Claude | Claude 3.5 Sonnet | 2024-06 | 官方未公开 | 官方未公开 | 200K | 系统级长会话处理 |
| Mistral | Mistral 7B | 2023-09 | Transformer Dense 7B | GQA + SWA | 8K 级，窗口 4096 | 滚动 KV Cache |
| Mistral | Mixtral 8x7B | 2023-12 | MoE 46.7B/12.9B Active | GQA | 32K | 8 专家 Top-2，全注意力 |
| Gemma | Gemma 1 | 2024-02 | Transformer Dense 2B/7B | MHA | 8K | 标准小模型 |
| Gemma | Gemma 2 | 2024-06 | Transformer Dense 2B/9B/27B | GQA + 局部/全局交替 | 8K | 4096 滑动窗口局部层 |
| Falcon | Falcon 40B | 2023-05 | Transformer Dense 40B | MQA + ALiBi | 2K 级 | 单 KV Head + ALiBi |
| Yi | Yi-34B | 2023-11 | Transformer Dense 34B | GQA | 4K；200K 变体 | 56 Query 头/8 KV 头 |
| Grok | Grok-1 | 2024-03 | MoE 314B | 25% 层 Attention + 8 KV Head | 8K | 层间稀疏 + 8 专家 Top-2 |
| PaLM | PaLM 540B | 2022 | Transformer Dense 540B | MQA | 2,048 | 大模型 MQA |
| MiniCPM | MiniCPM3-4B | 2024-09 | Transformer Dense 4B | MLA | 32K | 小模型 MLA |

---

## 6. 2024 年主流选型与未来趋势

### 6.1 已形成共识的选型

1. **GQA 是 2024 年开源 Dense/MoE 的默认层内压缩方案**。它比 MHA 省显存，比 MQA 稳，且实现成本低。
2. **MLA 是 2024 年最具冲击力的新方案**。DeepSeek 把 KV Cache 压缩从“共享 Head”升级为“低秩潜空间”，并解决了 RoPE 兼容问题。
3. **FlashAttention 和 PagedAttention 成为推理基础设施标配**。它们不改变模型数学，但直接改善吞吐和显存利用率。
4. **长上下文主流从 8K/32K 快速移动到 128K**，1M 以上更多依赖分布式、系统级和混合方案。
5. **闭源模型不再公开 Attention 细节**。GPT、Gemini、Claude 都不能仅凭上下文窗口推断架构。

### 6.2 未来演进方向

- **层内压缩继续深化**：GQA 之后，MLA 以及类似低秩潜空间方案会继续与 MoE 组合。
- **层间混合常态化**：全量 Softmax Attention 层会与 Linear Attention、SSM、稀疏 Attention 按比例交错，而不是所有层使用同一种 Attention。
- **序列维度稀疏化**：在长度方向压缩或选择重要 Token，例如把多个 Token 压缩成单个 KV Entry，或用 Top-k 选择关键历史 Token。
- **位置编码与固定状态解耦**：当线性注意力或 SSM 承担主要状态记忆时，RoPE 不再是唯一方案，NoPE、门控衰减和跨层残差会成为研究方向。
- **硬件与算法协同设计**：FlashAttention 证明了内核优化带来的数量级收益，未来新 Attention 会更早考虑 Hopper/Blackwell 等硬件的异步、张量核心和低精度特性。
- **推理吞吐优先**：大模型发布时会更明确公布“激活参数、KV Cache、上下文、首 Token/解码吞吐”，Attention 选型将越来越像工程约束下的系统设计。

---

## 7. 附录：2025 年后已公开的 Attention 动向

以下内容超出“截至 2024”截止线，仅收录有官方技术报告、开源仓库或官方博客支撑的信号。

| 模型/事件 | 时间 | 已公开 Attention 信号 |
|-----------|------|----------------------|
| MiniMax-01 开源 | 2025-01 | Lightning Attention + Softmax Attention 混合，1M 上下文 |
| DeepSeek-R1 | 2025-01 | 基于 DeepSeek-V3-Base，架构沿用 MLA |
| Qwen3 | 2025-04 | 延续 GQA 路线，公开配置加入 QK-Norm 等训练稳定性优化 |
| Qwen3-Next | 2025-09 | Gated DeltaNet + Gated Attention 混合，3:1 线性/全量比例 |
| GLM-4.5 等后续版本 | 2025 | 延续 GQA 并强化长上下文与训练稳定，具体以官方报告为准 |
| Kimi K2 | 2025-07 | 1T MoE 采用 MLA，256K 上下文 |
| DeepSeek-V3.2 | 2025-09 | 在 MLA 上引入 DeepSeek Sparse Attention，包含序列压缩与稀疏选择 |

---

## 8. 主要参考资料

- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Fast Transformer Decoding: One Write-Head Is All You Need (MQA): https://arxiv.org/abs/1911.02150
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints: https://arxiv.org/abs/2305.13245
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness: https://arxiv.org/abs/2205.14135
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning: https://arxiv.org/abs/2307.08691
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision: https://arxiv.org/abs/2407.08608
- Efficient Memory Management for Large Language Model Serving with PagedAttention: https://arxiv.org/abs/2309.06180
- Ring Attention with Blockwise Transformers for Near-Infinite Context: https://arxiv.org/abs/2310.01889
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model: https://arxiv.org/abs/2405.04434
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Qwen2 Technical Report: https://arxiv.org/abs/2407.10671
- Qwen2.5 Technical Report: https://arxiv.org/abs/2412.15115
- GLM-130B: An Open Bilingual Pre-trained Model: https://arxiv.org/abs/2210.02414
- ChatGLM2-6B 官方仓库: https://github.com/THUDM/ChatGLM2-6B
- Kimi: A Series of Large-scale Chinese Language Models: https://arxiv.org/abs/2310.08588
- MiniMax-01: Scaling Foundation Models with Lightning Attention: https://arxiv.org/abs/2501.08313
- LLaMA: Open and Efficient Foundation Language Models: https://arxiv.org/abs/2302.13971
- Llama 2: Open Foundation and Fine-Tuned Chat Models: https://arxiv.org/abs/2307.09288
- The Llama 3 Herd of Models: https://arxiv.org/abs/2503.24095
- Mistral 7B: https://arxiv.org/abs/2310.06825
- Mixtral of Experts: https://arxiv.org/abs/2401.04088
- Gemma: A Family of Lightweight State-of-the-Art Open Models: https://arxiv.org/abs/2403.08295
- Gemma 2: Improving Open Language Models at a Practical Size: https://arxiv.org/abs/2408.00118
- Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context: https://arxiv.org/abs/2403.05530
- Falcon 模型卡: https://huggingface.co/tiiuae/falcon-40b
- Yi: Open Foundation Models by 01.AI: https://arxiv.org/abs/2403.04652
- Grok-1 开源仓库: https://github.com/xai-org/grok-1
- PaLM: Scaling Language Modeling with Pathways: https://arxiv.org/abs/2204.02311
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces: https://arxiv.org/abs/2312.00752
- Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention: https://arxiv.org/abs/2006.16236
