# Attention Summary
     
下面按模型家族归纳各系列所采用的 Attention 架构，并补充更多主流/新兴模型。为便于理解，先简要说明文中出现的几种核心 Attention 变体：

| 缩写 | 全称 | 核心特征 |
|------|------|----------|
| **MHA** | Multi-Head Attention | 标准多头注意力，每头独立 Q/K/V，表达能力最强但 KV Cache 最大 |
| **MQA** | Multi-Query Attention | 所有头共享同一组 K/V，Cache 最小但质量下降明显 |
| **GQA** | Grouped-Query Attention | 将 Query 头分组，每组共享 K/V，平衡效率与质量 |
| **MLA** | Multi-head Latent Attention | 低秩联合压缩 K/V 为潜向量，Cache 极小且性能优于 MHA |
| **SWA** | Sliding Window Attention | 限制每 token 只关注局部窗口，降低长序列计算量 |
| **KDA** | Kimi Delta Attention | 基于 DeltaNet 的线性注意力，用门控机制更新循环状态 |
| **SSM** | State Space Model | 如 Mamba，用固定大小状态替代 KV Cache，O(1) 解码 |



如果把近几年的主流开源大模型放在一起，会发现一个非常清晰的趋势：

> **MHA → MQA → GQA → MLA → MLA + Sparse Attention → Linear/Full Hybrid Attention**

而且 2025–2026 年已经出现了两条明显分叉：

1. **DeepSeek / Kimi / GLM / Mistral：继续沿着 KV-cache 压缩路线发展，核心是 MLA**
2. **Qwen / Kimi Linear / MiniMax：开始探索 Linear Attention / Gated DeltaNet / Lightning Attention 与 Full Attention 混合**

下面我按这个思路系统归纳。

---

# 1. 先给一个总览

| 模型系列         | 代表版本            | Attention 架构                                | KV Cache 思路              | Long Context 思路  |
| ------------ | --------------- | ------------------------------------------- | ------------------------ | ---------------- |
| **Qwen**     | Qwen2.5         | GQA                                         | KV Head 共享               | RoPE + GQA       |
|              | Qwen3           | GQA                                         | KV Head 共享               | RoPE + GQA       |
|              | Qwen3-Next      | Gated DeltaNet + Gated Attention            | Linear + Full Hybrid     | 线性注意力            |
|              | **Qwen3.5**     | **Gated DeltaNet + Gated Attention**        | 大部分层不需要传统 KV Cache       | 3:1 Linear/Full  |
| **DeepSeek** | V2              | **MLA**                                     | Latent KV Compression    | MLA              |
|              | V3 / R1         | **MLA**                                     | Latent KV Compression    | MLA              |
|              | V3.2            | **MLA + DSA**                               | Latent KV + Sparse Token | Sparse Attention |
| **GLM**      | GLM-4           | GQA 类 Transformer                           | GQA                      | RoPE             |
|              | GLM-4.5         | **GQA + Partial RoPE + QK-Norm**            | GQA                      | GQA              |
|              | **GLM-5**       | **MLA**                                     | Latent KV Compression    | MLA              |
| **Kimi**     | Kimi K2         | **MLA**                                     | Latent KV Compression    | MLA              |
|              | **Kimi Linear** | **KDA + MLA**                               | Linear State + MLA       | Hybrid           |
| **MiniMax**  | MiniMax-01      | **Lightning Attention + Softmax Attention** | Linear State + KV        | Hybrid           |
|              | MiniMax M2      | **Full Attention / GQA**                    | KV Cache                 | Full Attention   |
| **Llama**    | Llama 2 70B     | GQA                                         | KV Head 共享               | GQA              |
|              | Llama 3/3.1/3.2 | GQA                                         | KV Head 共享               | GQA              |
|              | Llama 4         | GQA 系                                       | KV Head 共享               | MoE + GQA        |
| **Mistral**  | Mistral 7B      | GQA + Sliding Window                        | 局部 KV                    | Sliding Window   |
|              | Mixtral         | GQA + Sliding Window                        | 局部 KV                    | SWA              |
|              | Mistral Large 3 | **MLA-style**                               | Latent KV                | MLA              |
| **Gemma**    | Gemma 2/3       | GQA                                         | KV Head 共享               | RoPE + GQA       |
| **OLMo**     | OLMo 2/3        | GQA / MHA                                   | KV sharing               | Sliding Window   |
| **Phi**      | Phi-3/4         | GQA                                         | KV Head 共享               | RoPE + GQA       |
| **Falcon**   | Falcon 40B      | MQA                                         | 单 KV Head                | MQA              |
| **PaLM**     | PaLM            | MQA                                         | 单 KV Head                | MQA              |

这里最值得关注的是 **Qwen3.5、DeepSeek V3.2、Kimi Linear、GLM-5、MiniMax M2**，因为它们代表了 2025–2026 年 Attention 架构的几个不同方向。([Hugging Face][1])



---

## 一、Qwen 系列（阿里巴巴）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **Qwen2 / Qwen2.5** | **GQA** | 全系列采用 GQA + RoPE + SwiGLU + RMSNorm，保留 QKV-bias  |
| **Qwen3 (Dense & MoE)** | **GQA + QK-Norm** | 延续 Qwen2.5 架构，移除 QKV-bias，引入 QK-Norm 稳定训练；MoE 版有 128 专家，每 token 激活 8 个  |
| **Qwen3.5** | **Gated DeltaNet (GDN) + 全局注意力 混合** | 采用 3:1 比例（3 层线性注意力 + 1 层全局 softmax 注意力），40 层中仅 10 层使用标准 GQA  |

**演进逻辑**：从标准 GQA 向"线性注意力 + 稀疏全局注意力"的混合架构演进，用线性层处理长序列，全局层保证精确召回。

---

## 二、DeepSeek 系列

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **DeepSeek-V1 (67B Dense)** | **MHA** | 传统稠密 Transformer |
| **DeepSeek-V2 / V3 / V3.1** | **MLA** | **低秩 KV 联合压缩**：将 K/V 压缩为低维潜向量 $c_t^{KV}$，推理时仅缓存潜向量 + 解耦 RoPE 的 Key，Cache 约为 MHA 的 7%（93.3% 压缩率）；通过矩阵吸收避免显式重建高维 K/V   |
| **DeepSeek-V3.2 / V4** | **MLA + CSA/HCA** | 在 MLA 基础上引入 **Compressed Sparse Attention**：序列长度压缩（每 m 个 token 压缩为 1 个 KV Entry）+ 稀疏 Top-k 选择，进一步削减长上下文成本  |

**核心创新**：MLA 的 **Decoupled RoPE** 解决了低秩压缩与位置编码的兼容性难题——将位置信息剥离到独立的 $q^R/k^R$ 中，避免 RoPE 与上投影矩阵耦合 。

---

## 三、GLM 系列（智谱 AI）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **GLM-130B** | **MHA** | 传统多头注意力 + DeepNorm + RoPE + GLU |
| **ChatGLM2 / ChatGLM3** | **GQA** | 上下文从 2K 扩展到 32K |
| **GLM-4** | **GQA** | 上下文 128K~1M；除 QKV 外移除所有 bias；RMSNorm + SwiGLU；RoPE 扩展为 2D 形式  |
| **GLM-4.7 / GLM-5.2** | **MLA** | 跟进 DeepSeek 的 MLA 设计，GLM-5.2 还结合了 DSA (DeepSeek Sparse Attention) Lightning Indexer  |
| **GLM-Edge** | **GQA** | 端侧轻量模型，沿用 GQA  |

---

## 四、Kimi 系列（Moonshot AI）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **Kimi (初代) / k1.5** | **标准 Transformer + Ring Attention** | 支持 128K~2M 超长上下文，通过 Ring Attention 实现分布式长序列计算  |
| **Kimi K2 (1T MoE)** | **MLA** | 遵循 DeepSeek 配方，使用 MLA + MoE |
| **Kimi K3 (2.8T MoE)** | **KDA + Gated MLA + Attention Residuals** | **3:1 混合**：69 层 KDA（线性注意力）+ 24 层 Gated MLA；**NoPE**（完全移除 RoPE，依赖线性注意力的位置感知）；**Attention Residuals** 替代标准残差连接，跨深度选择性检索表征   |
| **Kimi Linear** | **KDA + MLA 混合** | 48B 参数，KDA 与 MLA 3:1 交错，长上下文生成速度比 DeepSeek-V3 快 2.3×  |

**KDA 技术本质**：基于 Gated DeltaNet，用广义 Householder 变换的 delta 规则更新线性循环状态，并加入对角门控实现细粒度衰减控制，可改写为 chunk-wise 并行格式 。

---

## 五、MiniMax 系列

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **MiniMax-M1** | **Lightning Attention + Full Attention 混合** | 456B 总参 / 46B 激活，尝试线性注意力与全注意力的混合 |
| **MiniMax-M2** | **Full Attention** | 旗舰模型回归全注意力，因线性注意力在多轮推理任务上精度不足  |
| **MiniMax-M2.7 / M3** | **GQA** | 48Q/8KV 或 64Q/4KV，带 partial RoPE 64 和 per-layer/per-head QK-Norm  |

---

## 六、LLaMA 系列（Meta）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **LLaMA 1** | **MHA** | 全系列标准多头注意力 |
| **LLaMA 2** | **MHA (7B/13B) / GQA (70B)** | 仅最大模型使用 GQA，小模型仍用 MHA  |
| **LLaMA 3 / 3.1 / 3.2** | **全系列 GQA** | 从 3B 到 70B 统一使用 GQA + RoPE (base 500K) + SwiGLU + RMSNorm，支持 128K 上下文  |

---

## 七、Mistral / Mixtral 系列（Mistral AI）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **Mistral 7B** | **GQA + SWA** | 32 Query 头 / 8 KV 头；**Sliding Window Attention** (4096 token 窗口) + Rolling Buffer KV Cache，理论上可外推至 128K  |
| **Mixtral 8x7B** | **GQA + 全密集注意力** | 46.7B 总参 / 12.9B 激活；**注意**：虽然继承 Mistral 架构，但官方配置中 `sliding_window` 为 null，实际使用全密集 32K 上下文，非 SWA  |
| **Mistral Large / Codestral** | **GQA** | 延续 GQA 设计，去除 SWA |

---

## 八、Yi 系列（01.AI）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **Yi-6B / Yi-34B** | **GQA** | 基于 LLaMA 代码修改，但**全系列统一使用 GQA**（不像 LLaMA 2 仅 70B 使用）；Yi-6B 为 32Q/4KV，Yi-34B 为 56Q/8KV  |

---

## 九、Gemma 系列（Google）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **Gemma 2** | **GQA + 混合局部/全局注意力** | 1:1 交替局部滑动窗口层（4096 窗口）与全局全注意力层  |
| **Gemma 3** | **GQA + 5:1 局部/全局混合 + QK-Norm** | 5 层局部滑动窗口（1024 窗口）+ 1 层全局注意力；全局层 RoPE base 1M，局部层保持 10K；支持 128K 上下文  |

---

## 十、Nemotron 系列（NVIDIA）

| 代际 | Attention 架构 | 关键细节 |
|------|---------------|----------|
| **Nemotron 3 Nano (30B-A3B)** | **Mamba-2 + GQA 混合** | 23 层 Mamba-2 (SSM) + 6 层 GQA 注意力交错，仅 12% 层使用 Attention，KV Cache 压缩率 94%   |
| **Nemotron 3 Super (120B-A12B)** | **Mamba-2 + Latent-MoE + GQA 混合** | 88 层 = 40 Mamba-2 + 40 Latent-MoE + 8 GQA；使用 NoPE，无 QK-Norm  |
| **Nemotron 4B (边缘版)** | **GQA** | 轻量版使用标准 GQA  |

**设计哲学**：用 Mamba-2 的固定大小状态替代大部分层的 KV Cache，仅在少数层保留 GQA 以保证精确内容检索。

---

## 十一、其他重要模型补充

### 11.1 闭源/商业模型

| 模型 | Attention 架构 | 说明 |
|------|---------------|------|
| **GPT-4 / GPT-4 Turbo / GPT-4o** | 未公开（推测 GQA 或变体） | OpenAI 未披露架构细节，128K 上下文 |
| **Claude 3.5 / 4 (Opus/Sonnet)** | 未公开（标准 Transformer 变体） | Anthropic 支持 1M 上下文，使用自动 context compaction 处理长会话  |
| **Gemini 1.5 Pro / 2.0** | 未公开 | Google 支持 1M~10M 上下文，技术报告未披露注意力细节 |
| **GPT-OSS-120B (OpenAI 开源)** | **Sliding Window GQA** | 交替滑动窗口 GQA 层  |

### 11.2 轻量/端侧模型

| 模型 | Attention 架构 | 说明 |
|------|---------------|------|
| **MiniCPM3-4B（面壁智能）** | **MLA** | 小模型采用 MLA，受 DeepSeek 启发  |
| **Phi-3 / Phi-4（Microsoft）** | **GQA + Block Expansion** | 基于 LLaMA 架构，128K 上下文，使用 Block Expansion 技术扩展深度 |
| **Gemma 2/3（Google）** | **GQA + 混合局部/全局** | 见上文 |
| **Fox-1.6B / Index-1.9B** | **GQA / MQA** | 边缘模型实验不同注意力配置  |

### 11.3 非 Transformer / 状态空间模型

| 模型 | 架构 | 说明 |
|------|------|------|
| **Mamba / Mamba-2** | **纯 SSM** | 完全摒弃标准注意力，用固定大小状态矩阵实现 O(1) 解码复杂度，无 KV Cache |
| **Falcon-Mamba-7B** | **Mamba 混合** | 纯 SSM 架构的 LLM，用于与 Transformer 对比研究  |
| **Zamba（AI21）** | **Mamba + Attention 混合** | 层间交替 Mamba 与注意力 |

---

## 十二、架构演进趋势总结

### 12.1 效率优化三维度

现代 LLM 的 Attention 优化已从单一技术演变为**三个正交维度**的组合 ：

1. **层内压缩**（单层的 KV 怎么存）：MHA → GQA → MQA → MLA
2. **层间稀疏**（哪些层用 Attention）：全 Attention → Hybrid（部分层用 Linear/SSM）
3. **序列压缩**（KV 在长度维度怎么压缩）：CSA/HCA（每 m 个 token 压缩为 1 个 KV Entry）

### 12.2 当前主流格局（2025-2026）

```
高性能开源模型：
├── DeepSeek 系：MLA（层内极致压缩）
├── Kimi 系：KDA + Gated MLA（线性注意力混合）
├── Qwen 系：GQA → Gated DeltaNet 混合（Qwen3.5）
├── GLM 系：GQA → MLA（跟进 DeepSeek）
├── LLaMA 系：全系列 GQA（最保守但稳定）
├── Mistral 系：GQA + SWA（局部窗口优化）
└── Nemotron 系：Mamba-2 + GQA（SSM 混合，层间极致稀疏）

闭源商业模型：
├── GPT-4/Claude/Gemini：架构未公开，推测使用 GQA 或等效优化
└── 长上下文依赖：Ring Attention、Context Compaction 等系统级优化
```

### 12.3 关键分水岭

- **2023 年**：GQA 成为开源模型标配（LLaMA 2 70B、Mistral 7B、Yi 系列）
- **2024 年**：MLA 由 DeepSeek-V2 引入，实现 KV Cache 数量级压缩，随后被 GLM、MiniCPM、Kimi K2 等跟进
- **2025 年**：**线性注意力复兴** — Qwen3.5、Kimi K3、Nemotron 3 等转向"线性/SSM + 稀疏全局注意力"的混合架构，用 3:1 或更高的比例替换标准注意力层，追求 O(1) 解码复杂度
- **2026 年**：**NoPE（无位置编码）** 在 Kimi K3 等模型中验证可行；**Attention Residuals** 等跨层残差变体出现，挑战传统 Pre-Norm/Post-Norm 设计 