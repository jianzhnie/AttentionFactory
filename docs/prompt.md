你是一名同时具备大模型架构研究能力与 PyTorch 工程实现能力的高级 AI 系统研究员。请围绕主流大语言模型的 Attention 架构演进，完成一份“研究综述 + 结构化对比 + 代码实现方案 + 遗漏分析”的高质量交付。输出必须兼顾技术准确性、可验证性、工程可落地性与文档可读性。

请重点分析以下模型系列：
- 必选：Qwen、DeepSeek、GLM、Kimi、MiniMax、Step、xiaomi/Mimo、Hunyuan, Llama
- 额外补充至少 12 个主流模型系列：GPT、Gemini、Claude、Mistral、Mixtral、Yi、Phi、Grok、DBRX、Gemma、Nemotron、InternLM、Baichuan、Falcon、PaLM、MiniCPM、Mamba/Zamba、Snowflake Arctic 等

请严格按以下要求执行：

## 1. 研究范围与演进梳理

- 针对每一个模型系列，梳理从初代到截至 2026 年 8 月最新版本的 Attention 架构演进路线。
- 明确列出每个关键版本的发布时间、模型代际、是否开源、上下文窗口、KV cache 策略、训练/推理侧关键优化点。
- 对每个版本标注具体 Attention 类型或相关机制，包括但不限于：
  - MHA
  - MQA
  - GQA
  - MLA
  - SWA
  - Sparse Attention / Block Sparse Attention
  - CSA / HCA / DSA
  - Linear Attention / Hybrid Linear Attention
  - Gated DeltaNet / KDA
  - Lightning Attention
  - Ring Attention
  - FlashAttention v1/v2/v3/v4
  - PagedAttention / FlashMLA
  - RoPE、YaRN、NTK-aware、ALiBi、位置插值、p-RoPE、LongRoPE、2D Position
  - MTP、投机解码、KV offload、on-disk KV cache 等与 Attention 强相关的工程机制
- 若某版本为闭源模型且官方未公开完整实现细节，必须明确标记“官方未完全披露”，并区分“公开确认信息”“论文/技术报告信息”“基于公开线索的合理推断”，禁止把推断写成事实。
- 必须做“遗漏分析”：对照 `docs/attention_review_2026.md` 与 `llminfra/`，指出当前综述未覆盖的模型和当前代码未覆盖的核心模块。

## 2. Attention 机制级别的技术分析

- 对每类 Attention 架构说明：
  - 核心原理
  - 关键参数或结构约束
  - 主要优化目标
  - 适用场景
  - 主要收益与代价
- 尽可能给出量化指标：
  - 推理吞吐提升比例
  - 预填充/解码阶段速度提升
  - KV cache 显存占用下降比例
  - 上下文窗口扩展倍数
  - 训练 FLOPs 或带宽压力变化
- 所有量化结论必须注明来源类型；若不同论文/团队测试口径不一致，必须明确写出“不宜直接横向比较”。

## 3. 跨模型对比与设计取舍分析

- 对比不同模型系列在相同或相近 Attention 方案上的实现差异。
- 必须覆盖：
  - MHA vs MQA vs GQA
  - MLA 与传统 KV cache 设计
  - FlashAttention 作为训练优化、推理优化或工程加速组件
  - 长上下文路线中“稀疏化”“窗口化”“位置编码缩放”“缓存压缩”的取舍
- 从以下维度分析各团队的设计动机：
  - 长上下文能力
  - 推理效率与服务成本
  - 训练算力/显存约束
  - 开源生态适配
  - 商业 API 场景与闭源部署需求
  - MoE 架构协同需求

## 4. 结构化汇总表

至少输出 3 张高质量表格：

1. 模型系列/版本级汇总表，字段包括：模型系列、具体版本、发布时间、是否开源、基础架构特征、Attention 核心类型、位置编码/长上下文方案、上下文窗口长度、KV cache/内存优化策略、关键优化点、信息来源与可信度标记。
2. Attention 类型能力对比表，字段包括：Attention 类型、代表模型、时间复杂度/显存特征、典型优势、典型局限、适用场景。
3. 模型与模块覆盖缺口表，字段包括：模型/模块名称、当前文档是否覆盖、当前代码是否覆盖、优先级、建议实现方式。

## 5. 行业趋势总结

- 总结截至 2026 年 8 月 Attention 架构的主流演进趋势。
- 重点回答：
  - 为什么越来越多模型从 MHA 迁移到 GQA/MQA/MLA 类方案
  - 长上下文时代 Attention 的核心瓶颈是什么
  - 训练优化与推理优化的重心分别是什么
  - 哪些技术更可能成为未来 1 到 2 年的主流方向
- 对未来方向给出审慎判断：
  - GQA/MLA 是否会继续主导主流大模型
  - 稀疏/线性 Attention 是否会在超长上下文中重新崛起
  - Attention 与 State Space / Hybrid 架构是否会进一步融合

## 6. 代码实现任务

在 `/Users/jianzhengnie/LLMInfra/llminfra` 中实现或完善主流 Attention 架构、位置编码、MoE 和其他相关模块，并补充测试。

当前仓库已包含：
- MHA、MQA、GQA、MLA
- SWA、Block Sparse Attention、Linear Attention
- Hybrid Attention、Gated DeltaNet、Lightning Attention
- Ring Attention、Compressed Sparse Attention、ALiBi Attention
- PagedAttention 教学接口、FlashMLA 接口、FlashAttention v1-v4
- RoPE、YaRN、Dynamic NTK、ALiBi、Partial RoPE、Position Interpolation、LongRoPE、2D Position
- ExpertFFN、TopKRouter、MixtureOfExperts、DeepSeekMoE、LatentMoE、load-balance loss
- RMSNorm、SwiGLU FFN、FeedForward、TransformerBlock、CausalLMModel
- BlockSparseIndexer、AttentionResidual、MultiTokenPredictionHead
- SpeculativeDecoder、OnDiskKVStore、Mamba2Layer
- `build_attention`、`build_positional_encoding`、`list_attentions`

仍需重点补充或完善：
1. 真实生产级 FlashMLA / CSA / DSA CUDA kernel，当前为 PyTorch 教学接口
2. 分布式 Ring Attention 的多设备通信实现，当前为单机分块在线 Softmax
3. ALiBi 在 TransformerBlock / CausalLMModel 中的自动 additive bias 集成，当前为独立 AlibiAttention
4. LongRoPE 官方精确系数与 2D Position 在模型级组合中的完整接入
5. Mamba-2 精确选择性扫描 / 并行扫描，当前为简化固定状态 SSM
6. DSpark / EAGLE 真实投机解码调度，当前为简化 draft-target 验证
7. KV cache offload 与 PagedAttention 的生产级内存调度、copy-on-write
8. MoE expert parallelism / group GEMM，当前为按专家循环的教学实现
9. 文档与模型库补充：Step、xiaomi/Mimo、Mamba/Zamba、Snowflake Arctic

对每个新增实现需提供：
- 核心代码
- 清晰的模块说明
- 最小可运行示例
- 单元测试
- 与现有 `BaseAttention`、`nn.Module` 或统一接口风格一致的设计说明

若某 Attention 架构更偏向系统层调度优化，例如 PagedAttention、FlashMLA、Ring Attention，必须明确区分“算法接口模拟版”和“真实生产级实现”的差异。

## 7. 输出格式要求

请严格按以下结构输出：

# 一、执行摘要

# 二、模型系列逐一分析

# 三、Attention 机制专题解析

# 四、跨模型横向对比

# 五、结构化汇总表

# 六、行业趋势与未来判断

# 七、代码实现方案

# 八、参考资料

在“二”和“七”中必须给出遗漏分析，明确“已覆盖”“未覆盖”“待补充”三档结论。

## 8. 质量约束

- 输出必须“先研究、后下结论、再给实现方案”，不能直接跳到代码。
- 结论必须尽量基于公开资料，不得编造不存在的模型细节。
- 对闭源模型的未知点必须显式标注不确定性。
- 必须主动澄清：
  - 训练时使用 FlashAttention 不等于模型架构本身采用了新的 Attention 类型
  - PagedAttention 更多是推理系统优化，并不总是模型结构层的 Attention 变种
  - 长上下文能力不完全等同于 Attention 结构升级，也可能来自位置编码扩展与训练策略
- 语言要求：中文输出，术语首次出现时给出中英文对照。
- 风格要求：信息密度高、结构清晰、避免空泛表述、尽量给出可验证证据。

## 9. 最终交付要求

最终交付应当让我可以直接据此完成：
- 一篇高质量技术综述文档
- 一份模型 Attention 选型数据库或表格
- 一份模型与模块覆盖缺口清单
- 一套在 `LLMInfra` 仓库中逐步实现主流 Attention 架构及相关模块的开发任务清单
