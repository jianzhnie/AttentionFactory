# 完整 Transformer 架构演进调研 Prompt

> **角色**：同时具备大模型架构研究能力与 PyTorch 工程实现能力的高级 AI 系统研究员
> **目标**：围绕主流大语言模型 **完整 Transformer 架构**（Attention、FFN、归一化、激活函数、残差、MoE、SSM、位置编码、Embedding、输出头、整体范式、量化/多模态）演进，交付「研究综述 + 结构化对比 + 代码实现方案 + 遗漏分析」，兼顾技术准确性、可验证性、工程落地性。

---

## 模型分析范围

**必选**：Qwen、DeepSeek、GLM、Kimi、MiniMax、Step、Xiaomi MiMo、Hunyuan、Llama  
**额外补充 ≥12 个**：GPT、Gemini、Claude、Mistral、Mixtral、Yi、Phi、Grok、DBRX、Gemma、Nemotron、InternLM、Baichuan、Falcon、PaLM、MiniCPM、Mamba/Zamba、Snowflake Arctic

---

## 1. 研究范围与演进梳理

### 1.1 全模块覆盖要求（每类必分析，标注 A-K）

| 类别 | 核心模块 | 关键分析点 |
|------|----------|-----------|
| **A** | Attention | MHA/MQA/GQA/MLA、SWA、Sparse/BlockSparse/CSA/HCA/DSA/MSA、Linear/Hybrid、Gated DeltaNet/KDA、Lightning/Ring/FlashMLA；FlashAttention v1-v4；PagedAttention/KV offload/On-disk KV；投机解码(MTP/EAGLE/DSpark/Medusa) |
| **B** | 位置编码 | Absolute(Learned/Sinusoidal)、RoPE、ALiBi、T5 Bias、YaRN、NTK-aware/Dynamic NTK、PI、LongRoPE、p-RoPE、2D Position/mRope、NoPE 间隔层 |
| **C** | FFN/MLP | 两层 Linear-Act-Linear、SwiGLU/GeGLU/ReGLU/GEGLU、Gated MLP、Clamp-SwiGLU、QAT-FFN；中间维度比例(4x→8/3x→自定义)、bias/初始化、MoE 替代 FFN |
| **D** | 归一化 | LayerNorm/Post-LN、RMSNorm、DeepNorm、QK-Norm、LayerScale、RMSNorm 变体；Pre-Norm vs Post-Norm vs Sandwich-LN |
| **E** | 激活函数 | ReLU 系、GELU(精确/tanh/erf 近似)、SiLU/Swish 系、Clamp-SwiGLU、Squared ReLU |
| **F** | 残差与层序 | Post-Norm、Pre-Norm、Parallel Block(GPT-J 风格)、Sandwich-LN、Attention Residuals(Kimi K3)、残差缩放、Dropout/Stochastic Depth |
| **G** | MoE | Top-1/2/K 路由、共享专家、Soft/Hard/Latent Router、负载均衡损失、专家并行/Group GEMM、首层稠密、MoE vs 稠密 FFN 维度、专家/激活比例 |
| **H** | SSM/Hybrid | Mamba/Mamba-2(选择性扫描/并行扫描)、Zamba(Mamba+SharedAttn)、SSM+Transformer 交替、LinearAttn+SSM；d_state/d_inner 设计 |
| **I** | Embedding/输出头 | 词嵌入维度、Tied Embeddings、LM Head、MTP 多头、分类/奖励/多模态对齐头、Embedding Dropout/LayerNorm |
| **J** | 整体架构 | Encoder-Decoder、Decoder-only(Causal LM)、Encoder-only、Prefix LM；多模态融合(Vision Encoder+CrossAttn/Early/Late Fusion)；稀疏/混合架构哲学 |
| **K** | 训练/推理系统 | 混合精度(BF16/FP16/FP8)、QAT、推理量化(INT8/4/FP8/4/NF4)、KV 量化压缩、TP/PP/DP/EP/ZeRO、Gradient Checkpointing、Offload |

### 1.2 版本级梳理要求（每个模型系列）

对每个关键版本列出：发布时间、代际、是否开源、**A-K 全部模块选择**（沿用可注"同前版"）、上下文窗口、训练/推理关键优化、参数量（总/激活，MoE 分开）。

**闭源模型约束**：未公开细节必须标记「官方未完全披露」，区分三档可信度：**公开确认** / **论文/报告** / **合理推断**，禁止把推断写成事实。

### 1.3 遗漏分析

对照 `docs/attention_review_2026.md` 与 `llminfra/`，按「高/中/低」优先级指出：未覆盖的模型系列、文档已分析但代码未覆盖的模块、两者均未覆盖的模块。

---

## 2. 模块级技术分析（A-K 每类必做）

每类模块必须说明：核心原理/数学表达、关键参数/结构约束、主要优化目标（训练/推理速度、显存、长上下文、质量）、适用场景、收益/代价（量化指标优先）、与其他模块的协同/冲突。

**量化指标**：吞吐提升、prefill/decode 加速、显存下降（KV/参数/激活）、上下文扩展倍数、FLOPs/带宽变化、PPL/基准分变化。所有数字注明来源，不同口径明确标注「不宜直接横向比较」。

### 重要澄清（本节开头显式写出）

1. FlashAttention 是 CUDA kernel 级优化，不等于新的 Attention 数学形式，可与任何架构组合
2. PagedAttention 是推理系统层 KV 内存调度，不总是模型结构层的 Attention 变种
3. 长上下文能力 ≠ Attention 升级，也可能来自位置编码扩展 / 长语料训练策略
4. FFN / 归一化 / 激活函数对训练稳定性和质量的影响不亚于 Attention，架构分析不能只看 Attention
5. MoE 设计不仅是专家数量，路由策略、共享专家、负载均衡损失同样关键

---

## 3. 跨模型对比与设计取舍

### 3.1 单模块横向对比

对每类模块对比不同模型的实现差异：
- Attention：MHA/MQA/GQA/MLA 的 KV head 比例、低秩维度、RoPE 解耦
- 位置编码：RoPE vs ALiBi vs NoPE（外推性、训练稳定性）
- FFN：两层 FFN vs SwiGLU vs Gated MLP（中间维度、计算/参数量比）
- 归一化：RMSNorm vs LayerNorm vs DeepNorm+QK-Norm（稳定性、收敛速度）
- MoE：Top-2 vs Top-K>2、共享专家有无、负载均衡设计
- SSM/Hybrid：纯 Mamba vs 交替层 vs Linear+Full 混合
- 整体架构：纯稠密 vs MoE vs 混合架构适用场景

### 3.2 跨模块协同设计分析（必覆盖）

- Attention + FFN：Parallel vs Serial、层比例（如 Qwen3-Next 的 3:1 线性:全量）
- MoE + Attention：MoE 替换 FFN、Attention/MoE 层比例、共享/路由专家比例
- 归一化 + 残差：Pre/Post-Norm 对不同 Attention 类型的适配
- 位置编码 + Attention：RoPE 对 MQA/GQA 的 head_dim 约束、ALiBi 与 LinearAttn 适配
- SSM + Attention：交替比例、d_state 与 head_dim 匹配

### 3.3 设计动机分析（全模块维度）

从以下维度分析各团队设计决策：长上下文能力、推理效率/服务成本、训练算力/显存约束、开源生态适配、商业 API/闭源部署、MoE 协同需求、质量 vs 成本权衡。

---

## 4. 结构化汇总表（≥7 张）

| 表号 | 名称 | 关键字段 |
|------|------|---------|
| 表1 | 模型版本级全模块汇总 | 模型系列、版本、时间、开源、参数量(总/激活)、Attention/位置编码/上下文、FFN/归一化/激活/残差、MoE(专家/激活/共享)、SSM/Hybrid、Embedding/输出头、KV/内存优化、训练/推理系统、关键优化、来源可信度 |
| 表2 | Attention 类型能力对比 | Attention 类型、代表模型、时间/显存特征、优势/局限、适用场景、推荐位置编码/归一化搭配 |
| 表3 | FFN/MLP 类型能力对比 | FFN 类型、数学结构、计算/参数量比、代表模型、优势/局限、适用场景、推荐激活 |
| 表4 | 归一化与残差设计对比 | 方案、核心结构、稳定性影响、推理开销、代表模型、优势/局限、适用场景 |
| 表5 | MoE 架构设计对比 | MoE 类型、路由策略、共享专家、负载均衡、代表模型、专家/激活比例、优势/局限、适用场景 |
| 表6 | SSM/Hybrid 与混合架构对比 | 架构类型、核心结构、线性/二次复杂度、代表模型、优势/局限、适用场景、与 Attention 组合方式 |
| 表7 | 模型与模块覆盖缺口 | 模型/模块、模块类别(A-K)、文档覆盖、代码覆盖、优先级、建议实现方式、与现有接口兼容方案 |

---

## 5. 行业趋势总结

### 5.1 Attention 相关
- MHA → GQA/MQA/MLA 迁移动机
- 长上下文 Attention 核心瓶颈（计算 vs 带宽 vs KV 大小 vs 调度）
- 训练 vs 推理优化重心
- FlashAttention 系列下一步方向

### 5.2 非 Attention 模块趋势
- 「RMSNorm + Pre-Norm + SwiGLU」成为默认三件套的原因与具体收益
- QK-Norm、LayerScale 的优势场景
- FFN 中间维度缩小（4x→8/3x）的动机与质量影响
- MoE Top-K 激活专家增长（2→4/6/8/10/16）说明什么
- 共享专家从无到有的演进逻辑
- 纯 Attention → 混合架构（Linear+Full、SSM+Attn、MoE+Dense）的整体趋势

### 5.3 跨模块协同趋势
- Attention+FFN 从串行走向并行/混合比例的原因
- MoE+Attention 混合架构中层比例的规律
- Linear/SSM 与 Full Attention 混合比例（如 3:1）如何确定
- 多模态融合对基础 Transformer 模块的新要求

### 5.4 未来方向审慎判断
- GQA/MLA 是否持续主导、MLA 低秩 KV 的理论瓶颈
- 稀疏/线性 Attention/SSM 在超长上下文是否重新崛起、纯二次 Attention 的边界
- Attention 与 State Space/Hybrid 的进一步融合、未来主导形态
- MoE 是否渗透到中小模型（<10B 激活）、普及瓶颈
- 位置编码走向统一（RoPE 胜出）还是混合方案
- QAT + 原生低精度 FFN/Attention 是否成为训练默认

---

## 6. 代码实现任务（`llminfra/` 目录）

### 6.1 仓库现状盘点

| 类别 | 已实现模块 |
|------|-----------|
| A. Attention | MHA/MQA/GQA/MLA、SWA/BlockSparse/Linear、Hybrid/GatedDeltaNet/Lightning、Ring/CSA/ALiBi、PagedAttn(教学)/FlashMLA(接口)/FA v1-v4 |
| B. 位置编码 | RoPE、YaRN、DynamicNTK、ALiBi、PartialRoPE、PI、LongRoPE、2DPosition |
| C. FFN | SwiGLUFFN、基础 FeedForward(GELU/ReLU/SiLU) |
| D. 归一化 | RMSNorm |
| E. 激活函数 | PyTorch 原生(SiLU/GELU/ReLU)，无自定义高级变体 |
| F. 残差/层序 | TransformerBlock 支持 Pre/Post-Norm 切换 |
| G. MoE | ExpertFFN、TopKRouter、MixtureOfExperts、DeepSeekMoE、LatentMoE、load-balance loss、ExpertParallelMoE(模拟) |
| H. SSM/Hybrid | Mamba2Layer(简化固定状态 SSM) |
| I. Embedding/输出头 | MultiTokenPredictionHead |
| J. 整体架构 | TransformerBlock、CausalLMModel |
| K. 推理系统 | SpeculativeDecoder、EagleSpeculator(简化)、OnDiskKVStore、BlockSparseIndexer、AttentionResidual |
| 通用接口 | `build_attention`、`build_positional_encoding`、`list_attentions` |

### 6.2 待补充/完善清单（33 项）

**A. Attention**
1. FlashMLA/CSA/DSA/MSA 生产级 CUDA/Triton kernel 或与 FA 适配方案
2. Ring Attention 分布式通信（torch.distributed all-to-all/reduce-scatter）
3. ALiBi 生产级 additive bias kernel（Block 级别自动注入）

**B. 位置编码**
4. LongRoPE 官方精确系数（256K/512K/1M 档位 ntk 因子、重要性采样）
5. mRope 在多模态模型级完整接入（图像 patch + 文本位置编码拼接对齐）

**C. FFN**
6. GeGLU/ReGLU/GEGLU 完整 GLU 变体 + 统一接口
7. Clamp-SwiGLU（clamp 范围可控）+ FP8/INT8 QAT-FFN 包装器
8. FFN 工厂函数（支持 4x / 8/3x / 自定义中间维度比例）

**D. 归一化**
9. LayerNorm / DeepNorm（含残差缩放因子）
10. QK-Norm（Attention 中 Q/K 分别归一化 + TransformerBlock 自动配置）
11. LayerScale（残差路径可学习逐通道缩放，ViT/Llama 4 风格）

**E. 激活函数**
12. 精确 GELU + tanh/erf 近似统一接口（配置切换）
13. Squared ReLU、Clipped SiLU 等高级激活（FFN 工厂可选）

**F. 残差/层序**
14. Parallel Block（GPT-J 风格并行 Attention-FFN，减少通信开销）
15. Sandwich-LN（归一化-计算-归一化-残差，Pre/Post 之外第三种层序）
16. Attention Residuals 在 TransformerBlock 通用集成（按层配置跨层残差通路）

**G. MoE**
17. Expert Parallel 真实通信（all-to-all 路由 + 本地 group GEMM）
18. Latent Router 完善：Gumbel-Softmax、Router Z-Loss
19. 无辅助损失路由（DeepSeek-V4/GLM-5 风格，验证大模型下去掉 load-balance loss 的可行性）
20. Expert dropout / Expert choice routing 多种路由统一接口

**H. SSM/Hybrid**
21. Mamba-2 精确选择性扫描/并行扫描（真实离散化 A/B/C/D、选择性 dt 投影、fused scan）
22. Zamba 风格 SSM + Shared Attention 混合 Block（交替配置 + 状态传递）
23. LinearAttn + SSM + Full Attn 通用混合布局（类似 HybridAttention 的 3:1 支持，扩展到 SSM）

**I. Embedding/输出头**
24. Tied Embeddings 文档化 + 单元测试（CausalLMModel 已支持 `tie_word_embeddings`）
25. MTP 完整训练接口（多步损失计算 + 位置偏移）

**J. 整体架构**
26. Encoder-Decoder 架构：EncoderBlock、DecoderBlock（带 Cross Attention）、EncoderDecoderModel
27. Prefix LM 与前缀掩码（Prefix 双向可见、生成部分因果掩码）
28. 多模态融合基础架构：Vision Encoder 接口、CrossAttn 融合、Late Fusion 对齐头骨架

**K. 推理系统**
29. 投机解码语义完善：全部接受 bonus token + 温度采样下拒绝采样（当前仅 greedy 验证）
30. DSpark/EAGLE 真实投机解码调度
31. KV cache offload + PagedAttention 生产级内存调度（HBM/CPU/NVMe 分层、copy-on-write 引用计数）
32. FP8/INT8 量化包装器（Attention+FFN 的 QAT + 推理量化统一接口）

**L. 文档**
33. Step、Xiaomi MiMo、Mamba/Zamba、Snowflake Arctic 模型库补充

### 6.3 新增实现交付要求

每项必须提供：核心代码（遵循现有 `BaseAttention`/`nn.Module`/统一接口风格）、模块说明（原理/参数/IO/与其他模块关系）、最小可运行示例（`__main__` 或 example 函数）、单元测试（`tests/` 对应子目录，覆盖形状/数值范围/梯度）、设计说明（与现有接口兼容、生产级 vs 教学级差异）。

**系统层优化模块**（PagedAttention/FlashMLA/Ring/ExpertParallel/KVOffload）必须明确区分「算法接口模拟版」（PyTorch 纯实现，用于理解原理）与「真实生产级实现」（CUDA/Triton/分布式通信）的差异与升级路径。

---

## 7. 输出格式

```markdown
# 一、执行摘要（全模块关键结论）

# 二、模型系列逐一分析（2.1-2.n 每系列）
  - 演进总览
  - 各版本全模块配置表（A-K）
  - 关键设计决策分析
  - 遗漏分析（已覆盖/未覆盖/待补充）

# 三、Transformer 整体架构演变与核心组件
  3.1 编年演变进程（原始 Transformer → 2026 主流）
  3.2 归一化层
  3.3 FFN 与激活函数
  3.4 MoE 混合专家
  3.5 残差连接与层序设计
  3.6 Embedding 与输出头
  3.7 量化感知训练
  3.8 多模态融合
  3.9 组件演变速查表
（SSM/Hybrid、整体范式、训练/推理系统在 3.1 与第四章展开）

# 四、Attention 机制专题解析
  4.1-4.n（MHA/MQA/GQA/MLA/SWA/Sparse/Linear&Hybrid/FlashAttention/PagedAttention/位置编码扩展）

# 五、跨模型横向对比与设计取舍
  5.1 单模块横向对比
  5.2 跨模块协同设计分析
  5.3 各团队设计动机总结

# 六、结构化汇总表（表1-表7）

# 七、行业趋势与未来判断

# 八、代码实现方案
  8.1 现有模块盘点与质量评估
  8.2 新增/完善模块逐个方案（对应 6.2 中 1-33 项：接口设计、核心算法、集成方式、测试计划）
  8.3 统一接口与工厂函数设计
  8.4 遗漏分析与优先级排序（已覆盖/未覆盖/待补充）

# 九、参考资料（分模块列出，注明可信度）
```

**强制要求**：第二章（模型分析）和第八章（代码方案）必须对 **A-K 全部模块类别** 给出遗漏分析，明确「已覆盖/未覆盖/待补充」三档，不能只覆盖 Attention。

---

## 8. 质量约束

- 先研究 → 后下结论 → 再给实现方案，禁止直接跳到代码
- 结论基于公开资料（官方模型卡/配置/技术报告/论文），不得编造模型细节
- 闭源模型未知点显式标注不确定性，三档可信度：**公开确认** / **论文/报告** / **合理推断**
- 主动澄清 §2 中列出的 5 条常见误解
- 中文输出，术语首次出现给出中英文对照
- 信息密度高、结构清晰、避免空泛，尽量给出可验证证据（配置字段名、论文页码/图表号）

---

## 9. 最终交付价值

交付物应可直接用于完成：
1. 一篇高质量 **完整 Transformer 架构** 技术综述（全模块覆盖，非仅 Attention）
2. 一份模型 **全模块配置** 选型数据库/表格
3. 一份按 A-K 分类的模型与模块覆盖缺口清单
4. 一套在仓库中逐步实现完整 Transformer 架构主流模块的 **开发任务清单与优先级排序**
5. 对现有代码库的重构/扩展建议（与 `build_*` 工厂、`BaseAttention`、`TransformerBlock`、`CausalLMModel` 风格一致）
