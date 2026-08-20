# LLMInfra 代码审计报告

- 审计日期:2026-08-19 ~ 2026-08-20
- 审计范围:`llminfra/` 全部 8 个子包(约 12.3k 行)
- 审计基线:318 个测试通过,ruff 干净,mypy 20 个错误
- 审计方法:逐文件人工审查 + 对可疑点编写探针脚本实际复现(所有 HIGH/MEDIUM 发现均有运行证据)
- 修复后状态:**392 个测试通过(新增 74 个回归测试),ruff check/format 干净,mypy 0 错误;全部 49 个发现已闭环**

严重级别定义:

- **HIGH**:正常使用场景下产生错误结果或崩溃
- **MEDIUM**:边界情形失败、静默错误语义或数值风险
- **LOW**:性能、风格、文档保真度问题

状态定义:✅ 已修复 | 📝 文档声明(行为保留,补充说明)

---

## 1. `llminfra/attention`

### 1.1 [HIGH] LinearAttention / LightningAttention 对 4D 稠密 mask 静默出错,3D 稠密 mask 直接崩溃 — ✅ 已修复

- 位置:`linear_attention.py:132-133`、`lightning_attention.py:169-170`
- 问题:对 4D mask 只取 query 第 0 行(`attention_mask[..., 0, :]`)作为 key padding mask。标准 `(b,1,s,s)` 因果 mask 第 0 行只有 key 0 可见,导致除第一个 token 外所有 key/value 被清零,**不报错但结果错误**(实测 maxdiff=2.34);3D 稠密 mask `(b,s,s)` 则直接广播失败 RuntimeError。
- 修复:与 GatedDeltaNet / KimiDeltaAttention 对齐 —— 4D 用 `any(dim=-2)` 归约(任意 query 可见即可见的 key),3D 且 `size(1)!=1` 时 `any(dim=1, keepdim=True)`。
- 回归测试:`tests/attention/test_linear.py` 4 个(causal=True 下传等价 tril mask 输出不变;3D mask 不崩溃)。

### 1.2 [HIGH] CompressedSparseAttention 压缩 key mask 犯同样错误 — ✅ 已修复

- 位置:`compressed_sparse_attention.py:121-123`
- 问题:同样只取 query 第 0 行,`causal=False` 传稠密因果 mask 时仅 block 0 保持可见(实测 maxdiff=1.7)。纯 padding mask `(b,1,1,s)` 恰好无此问题,故原测试未覆盖。
- 修复:`any(dim=-2)` 归约。
- 回归测试:`tests/attention/test_sparse.py::test_csa_dense_causal_mask_matches_implicit_causal`。

### 1.3 [MEDIUM] LightningAttention 块内 softmax 缩放因子用错维度 — ✅ 已修复

- 位置:`lightning_attention.py:105,115`
- 问题:einsum `"bhid,bhjd->bhij"` 的收缩维是 `feature_dim`,却用 `1/sqrt(head_dim)` 缩放。默认 `feature_dim == head_dim` 时不触发,一旦两者不同缩放即错。
- 修复:改用 `self.feature_dim`。
- 回归测试:`test_lightning_attention_intra_block_scale_uses_feature_dim`(feature_dim=8 ≠ head_dim=16,与手算参考对比)。

### 1.4 [MEDIUM] CompressedSparseAttention 因果模式下前 `compress_ratio - 1` 个 token 永远输出零 — 📝 文档声明

- 位置:`compressed_sparse_attention.py:194-204`
- 问题:压缩 entry 仅在「最后一个源 token 已过去」才可见,且 fallback 对 qb=0 只选 block 0,导致 query 位置 `0..ratio-2` 的所有 key 被 mask,经 `nan_to_num` 输出恰好为零(梯度也为零)。这是防泄漏设计与 fallback 策略叠加的隐性后果。
- 处理:选择逻辑保守不动,类 docstring 显式声明该行为。

### 1.5 [LOW] BaseAttention 不校验 mask 形状,float 加性 mask 被静默忽略 — ✅ 已修复

- 位置:`base_attention.py:153-156, 218-228`
- 问题:`attention_mask == 0` 的语义意味着 HF 风格 0/-inf 加性 float mask 会被静默忽略;mask 的 seq 维不匹配时错误在后续 `masked_fill` 处以隐晦广播错误爆出。
- 修复:float dtype mask 显式抛 `ValueError` 提示 1/0 约定;补 mask 末维 = seq_len 校验。
- 回归测试:`tests/attention/test_base.py` 2 个。

### 1.6 [LOW] GatedDeltaNet 的 "delta-rule" 名不副实 — ✅ 已修复

- 位置:`gated_delta_net.py:112-115`
- 问题:docstring 称 delta-rule,但状态更新是 `S = (1-g)S + g·k⊗v`(门控线性注意力),无误差修正项 `v - Sᵀk`;真正的 delta rule 在同目录 KimiDeltaAttention。
- 修复:模块与类 docstring 术语修正为 gated linear attention,并注明与 KimiDeltaAttention 的 delta rule 的区别(仅文档,行为不变)。

### 1.7 [LOW] LinearAttention 因果路径显存 O(s·f·d) — ✅ 已修复

- 位置:`linear_attention.py:87`
- 问题:`einsum(...).cumsum(dim=2)` 显式物化 `(b,h,s,f,d)` 的逐步状态,seq 稍大即爆内存。
- 修复:新增 `chunk_size: int = 64` 构造参数(向后兼容),因果路径改分块扫描 —— 跨块维护运行状态 `S(b,h,f,d)`,块内 tril 掩码局部贡献 + `phi(q) @ S` 历史贡献。显存峰值从 O(s·f·d)(s=4096 时状态张量约 1GB)降到 O(c·f·d + c²)(约 0.5MB);masked key/value 进入扫描前已清零,mask 语义不变。
- 回归测试:`test_linear_attention_chunked_matches_cumsum_reference`(seq_len ∈ {1,5,16,17,40} × 3 种核,带/不带 padding mask,float64 对拍;seq=4096 最大相对误差 3.3e-9)。

---

## 2. `llminfra/flash_attention`

### 2.1 [MEDIUM] 混合 dtype 输入:前向静默接受,反向崩溃 — ✅ 已修复

- 位置:`common/tiles.py:22-33`(`prepare_inputs`)
- 问题:只校验形状不校验 dtype;q/k fp16 + v fp32 时前向正常返回(输出静默转 q.dtype),backward 在 `common/ops.py:235` 抛 `RuntimeError: expected scalar type Half but found Float`。
- 修复:`prepare_inputs` 增加 `q.dtype == k.dtype == v.dtype` 校验,前向即抛带语义 ValueError。
- 回归测试:`tests/flash_attention/test_versions.py::test_mixed_input_dtypes_raise`(4 版本参数化)。

### 2.2 [MEDIUM] `kv_len=0` 时 tiled 实现与参考实现行为不一致 — ✅ 已修复

- 位置:`common/reference.py:48` vs fa1–fa4
- 问题:reference 对空序列抛 IndexError,而 fa1–fa4 因 block_slices 返回空列表静默返回全零输出和 LSE=0。
- 修复:`prepare_inputs` 统一显式拒绝 `q_len==0` / `kv_len==0` / `head_dim==0`,抛带语义 ValueError(同时消除 `q_len==0` 的 `torch.cat([])` 无上下文报错和 `head_dim==0` 的 ZeroDivisionError)。
- 回归测试:`test_degenerate_shapes_raise`(12 例)。

### 2.3 [LOW] debug trace 无条件构建,`keep_debug_state=False` 也付开销 — ✅ 已修复

- 位置:`flash_attention_v2.py`、`flash_attention_v3.py:269-286`、`flash_attention_v4.py` 的 `query_owners`/`pipeline_trace`/`scheduler_trace` append
- 修复:全部包进 `if config.keep_debug_state:`。

### 2.4 [LOW] `keep_debug_state` 默认 True,CUDA 上每 tile 一次主机同步 — ✅ 已处理

- 位置:`common/config.py:23`、`flash_attention_v4.py:267`(`bool(rescaled.any().item())`)、`flash_attention_v3.py:83`(fp8 路径 `.item()`)
- 决策:**保持默认 True** —— `test_versions.py:186` 与 `test_cuda.py:97` 隐式依赖默认值(用 `FlashAttentionConfig(fp8=True)` 却断言 debug 元数据),翻转会破坏测试契约。
- 处理:config docstring 显著注明性能代价(教学/调试定位,性能敏感场景应设 `False`);fa2/fa3/fa4 的 trace 与 `.item()` 已全部纳入 `keep_debug_state` 门控,`False` 时零 trace 开销(验证 `saved_state == {}`)。

### 2.5 [LOW] FA4 wave 结构物化整行 score tile — ✅ 已修复

- 位置:`flash_attention_v4.py:215-231`(forward)及 backward 对应段
- 问题:`main_outputs`/`softmax_outputs` 列表同时持有一个 wave 内全部 k tile 的 score 块,内存峰值是整行 `(b,h,block_q,kv_len)`。
- 修复:两个列表物化循环改为生成器流式 merge —— score 块惰性逐 tile 传递、correction merge 后立即释放,峰值降为单块 `(b,h,block_q,block_kv)`;wave/角色划分、`_correction_merge` 三分支、scheduler_trace 逐 tile 记录全部不变。
- 等价性:6 种形状 × 3 种 tile 配置 forward+backward 与 reference 对拍最大误差 1.9e-6,scheduler_trace 条目数与键集合不变。

---

## 3. `llminfra/inference`

### 3.1 [HIGH] paged_attention 的 einsum 跨 head 求和,注意力分数计算错误(测试参考公式同错)— ✅ 已修复

- 位置:`paged_attention.py:221,230`;测试 `tests/inference/test_inference.py:41`
- 问题:`einsum("qhd,khd->qk")` 中 `h` 出现在两个输入但不在输出,被**求和消去**:多 head 点积压成一组分数,所有 head 共享同一组权重。`num_heads=2` 时与 per-head 参考最大误差 0.888;`num_heads=1` 误差为 0,且测试的"参考实现"复制了同一错误公式,因此测试与实现错得一致、无法暴露。
- 修复:分数改 `"qhd,khd->hqk"`、输出改 `"hqk,khd->qhd"`,causal mask 加 head 维广播;**同步修正测试参考公式**。
- 回归测试:`test_paged_attention_computes_scores_per_head`(num_heads=2 逐 head 验证)。

### 3.2 [HIGH] `__init__.py` 的 `__all__` 导出 4 个未导入的名字 — ✅ 已修复

- 位置:`inference/__init__.py:13-20`
- 问题:`MedusaHead`、`MultiTokenPredictionHead`、`medusa_loss`、`mtp_loss` 列入 `__all__` 但从未导入(实际定义在 `speculative_decoding`),`from llminfra.inference import MedusaHead` 直接 ImportError。
- 修复:从 `__all__` 删除(顶层 `llminfra/__init__.py` 已从正确位置导出)。
- 回归测试:`test_inference_all_exports_resolve`。

### 3.3 [MEDIUM] 空 KV 缓存时 `PagedAttentionCache.get` 与 `paged_attention` 崩溃 — ✅ 已修复

- 位置:`paged_attention.py:153,210`
- 问题:`append` 对 seq_len=0 输入创建空 block table 条目后,`get`/函数对空 `rows` 调 `torch.cat` 抛 ValueError。
- 修复:`num_tokens==0` 时返回形状正确的空/零张量。
- 回归测试:`test_paged_cache_get_empty_sequence_returns_empty_tensors`。

### 3.4 [MEDIUM] TieredKVCache.put 不拷贝输入,同设备时与调用方共享存储 — ✅ 已修复

- 位置:`kv_cache_offload.py:88-92,139`
- 问题:`key.to(self.hbm_device)` 同设备是 no-op;hbm_device="cpu" 时 `put` 后调用方 `fill_(999)` 即污染缓存;HBM→CPU 驱逐路径同样。
- 修复:入库与驱逐两处 `.detach().clone().to(...)`。
- 回归测试:`test_tiered_kv_cache_copies_input_tensors`。

### 3.5 [LOW] BlockSparseIndexer 对 `seq_len=0` 崩溃 — ✅ 已修复

- 位置:`sparse_attention_indexer.py:81`
- 修复:block_count==0 时提前返回 `new_zeros((batch, num_heads, 0, top_k), dtype=long)`。
- 回归测试:`test_sparse_indexer_handles_empty_sequence`。

### 3.6 [LOW] BlockSparseIndexer 逐 block Python 循环做 topk — ✅ 已修复

- 位置:`sparse_attention_indexer.py:70-81`
- 问题:对每个 query block 单独切片 + `torch.topk` + pad,O(num_blocks) 次 kernel 调用。
- 修复:消除 Python 循环 —— causal block mask 广播 + `masked_fill(-inf)` 后一次批量 `torch.topk`,padding 语义(不足 top_k 用最后合法 index 填充)逐位复刻。
- 回归测试:`test_sparse_indexer_vectorized_matches_loop_reference`(12 组配置与内联慢速参考 `torch.equal` 对拍,含不能整除 block、`top_k > block_count`、`top_k=1`)。

---

## 4. `llminfra/layers`

### 4.1 [HIGH] HyperConnection 的可学习混合矩阵对输出完全无效 — ✅ 已修复

- 位置:`hyper_connection.py:72-77`
- 问题:所有 `hc_mult` 条 stream 是同一份 `hidden` 的 expand,branch 也以相同方式加到每条 stream —— 任意时刻所有 stream 恒等;行和为 1 的双随机矩阵对相同向量加权不变,再取 mean 仍不变。前向恒等于 `hidden + branch_scale*branch`,与 logits/hc_mult/sinkhorn_iters 全部无关(实测把 logits 换成 randn*5 输出逐位不变),模块静默退化为 per-channel gate。
- 修复:引入逐流可学习 `stream_scale`(ones 初始化)/`stream_bias`(std=0.02 随机初始化打破对称),归约从 mean 改为可学习 softmax 权重 `stream_weight`(近均匀初始化);混合矩阵从初始化起即对输出有实际作用。
- 回归测试:`tests/layers/test_hyperconnection.py` 新增"改变 logits 必须改变输出"等断言。

### 4.2 [MEDIUM] Sinkhorn 未做数值稳定化,logits 较大时输出 NaN — ✅ 已修复

- 位置:`hyper_connection.py:54`
- 问题:`matrix = self.logits.exp()`,fp32 下 logits≥~89 即 inf,inf/inf→nan 污染整个残差流(实测 logits=100 全 NaN)。
- 修复:`(self.logits - self.logits.amax()).exp()`;行/列归一加分母 `clamp_min(eps)`。

### 4.3 [MEDIUM] Hybrid 默认布局中 full attention 非因果,静默泄漏未来 — ✅ 已修复

- 位置:`hybrid_layers.py:70,162,173`
- 问题:同 stack 的 linear(causal=True)与 ssm(天然因果)都是因果的,而 full/attn mixer 的 MultiHeadAttention 无任何因果约束;实测 layer_map="full" 时扰动最后一个 token 改变位置 0 的输出。
- 修复:`attention_mask is None` 时为 full/attn mixer 默认构造 tril 因果 mask(用户传 mask 则尊重用户);配合 1.1 的修复,tril mask 转发给 LinearAttention 也安全。

### 4.4 [MEDIUM] 同一 `attention_mask` 转发给 mask 语义不兼容的两种 mixer — ✅ 已修复(随 1.1)

- 位置:`hybrid_layers.py:229` → `linear_attention.py:133`
- 问题:用户传入 tril 因果 mask 时,linear 层只保留 key 0(实测与无 mask 相差 1.96)。根因是 LinearAttention 的 4D mask 归约,已在 1.1 修复。

### 4.5 [LOW] deepnorm / post+parallel 风格下 `norm1`/`norm2` 是死参数 — ✅ 已修复

- 位置:`transformer_block.py:119-120`
- 问题:无条件构造但 deepnorm 与 post+parallel 分支从不调用,参数进 state_dict、永不获梯度。
- 修复:参照 norm3/norm4 的惰性构造,仅当 style 实际调用时创建。

### 4.6 [LOW] Mamba2 `_discretize` 显存 O(B·S·d_inner·d_state) — ✅ 已修复

- 位置:`state_space.py:169-170`
- 问题:一次性物化整条序列的 `a_bar`/`b_bar`(B=32,S=4096,d_inner=2048,d_state=16 时约 17GiB fp32),但两个 scan 都只逐步消费切片。
- 修复:`_discretize` 改单步张量签名,`_recurrent_scan`/`_chunked_scan` 循环内按步离散化;峰值张量从 2×(B,S,d_inner,d_state) 降到 (B,d_inner,d_state)(审计规模下 ~17GiB → 4MiB 级,改善倍数即序列长度 S)。逐步计算与"整量算再切片"是相同逐元素算子,结果位级一致。
- 回归测试:`test_mamba2_per_step_discretize_matches_full_materialization`(atol=1e-6)。

---

## 5. `llminfra/moe`

### 5.1 [MEDIUM] gumbel 路由模式下 `router_bias` 污染路由权重 — ✅ 已修复

- 位置:`mixture_of_experts.py:152-171`
- 问题:违反 docstring 明确约定("selection 用 `logits + router_bias`,权重仍从无偏 logits 计算",DeepSeek-V3 式 aux-free 均衡)。实测 `top_k=num_experts` 时选择集与 bias 无关,加 bias 前后权重最大差 0.872(应为 0)。
- 修复:gumbel 分支中选择概率用带偏置 logits、权重概率单独从无偏 logits 再算一次。
- 回归测试:`test_gumbel_weights_ignore_router_bias`。

### 5.2 [MEDIUM] gumbel hard 模式前向权重退化为均匀 1/top_k,train/eval 目标不一致 — ✅ 已修复

- 位置:`mixture_of_experts.py:160-164`
- 问题:直通估计后前向值全为 1(实测 top_k=2 时 weights=[0.5,0.5]),专家打分信息被丢弃,`scoring_func` 配置被静默忽略;且 gumbel 仅 training 生效,训练优化"均匀权重"目标、推理用"打分加权"目标。
- 修复:hard 模式前向权重改用无偏 logits 的 `_score` 打分(与 topk 路径和 eval 一致),gumbel 概率仅经直通项保留梯度(梯度语义不变)。
- 回归测试:`test_gumbel_hard_weights_match_unbiased_scores`。

### 5.3 [LOW] `load_balance_loss` 空输入返回 NaN — ✅ 已修复

- 位置:`mixture_of_experts.py:621`
- 修复:`router_logits.size(0) == 0` 时返回 `new_zeros(())`。
- 回归测试:`test_load_balance_loss_empty_tokens`。

### 5.4 [LOW] `ExpertChoiceRouter` 对 `top_tokens > num_tokens` 无显式校验 — ✅ 已修复

- 位置:`mixture_of_experts.py:251`
- 修复:forward 开头显式 ValueError。回归测试:`test_expert_choice_router_rejects_too_few_tokens`。

### 5.5 [LOW] `TopKRouter` docstring 缺三个公开参数 — ✅ 已修复

- 位置:`mixture_of_experts.py:58-79`
- 修复:补齐 `routing_strategy`、`gumbel_temperature`、`gumbel_hard` 说明(含 gumbel 仅 training 生效、hard 权重语义)。

### 5.6 [LOW] `DeepSeekMoE.forward` 多余 stack 物化副本 — ✅ 已修复

- 位置:`mixture_of_experts.py:377`
- 修复:`torch.stack([...]).sum(dim=0)` 改累加循环。

---

## 6. `llminfra/models`

### 6.1 [MEDIUM] CausalLMModel 默认 GQA 配置在部分合法 `num_heads` 下构造即崩溃 — ✅ 已修复

- 位置:`language.py:123-124`
- 问题:`num_kv_groups = max(1, num_heads // 2)` 不保证整除(如 num_heads=5 → 2 → ValueError)。
- 修复:取不超过 `num_heads//2` 的最大约数。
- 回归测试:`tests/models/test_language.py`(num_heads=5 构造并前向)。

### 6.2 [MEDIUM] `CausalLMModel._build_mask` 不把 `attention_mask` 移到输入设备 — ✅ 已修复

- 位置:`language.py:368-378`
- 问题:CPU mask × MPS/CPU 模型直接 RuntimeError(MPS 实测复现);兄弟类 EncoderOnlyModel 能容忍,行为不一致。
- 修复:`_build_mask` 中统一 `.to(device=device)`。
- 回归测试:加速器可用时验证(本机 MPS 通过)。

### 6.3 [LOW] MTP head 在带 labels 训练时被前向两次 — ✅ 已修复

- 位置:`language.py:296-314` + `speculative_decoding/mtp.py:135`
- 修复:`mtp_loss` 新增可选 `logits_list` 参数(向后兼容),训练分支复用已算的 `mtp_logits`。
- 回归测试:monkeypatch 计数 head 只前向一次。

### 6.4 [LOW] `MultimodalCausalLM` 的 `alignment_logits` 对 padding 不敏感 — ✅ 已修复

- 位置:`multimodal.py:310-311`
- 修复:朴素 mean 改 `_normalize_mask` + `pool_hidden_state` 的 masked mean。
- 回归测试:padding 扰动后 alignment_logits 不变(early/cross 各参数化)。

### 6.5 [LOW] early-fusion 路径重复构建 position ids — ✅ 已修复

- 位置:`multimodal.py:281-285`
- 修复:删除 forward 中仅为校验的冗余调用(`_early_fusion` 内已有同参数调用与校验;cross 模式保留必要校验)。

### 6.6 mypy override 错误(3 处)— ✅ 已修复

- 位置:`encoder_decoder.py:60`、`multimodal.py:99`(cross-attention 子类有意扩展签名)、`language.py:397`(PrefixLM 将 prefix 收紧为必需 keyword-only)
- 修复:均为有意的接口设计,加带注释说明的 `# type: ignore[override]`。`mypy llminfra/models` 已干净。

---

## 7. `llminfra/positional`

### 7.1 [HIGH] YaRN 插值/外推方向完全反了 — ✅ 已修复

- 位置:`rope_scaling.py:166-170`
- 问题:两重错误 —— ① ramp=0(高频维)应插值、ramp=1(低频维)应保持外推,实现恰好相反;② `linear_factor = max(1.0, 1.0/scale)` 在需要缩放时恒被钳到 1.0,高频端完全不缩放。与 HF 参考最大偏差 0.75,长上下文下位置编码语义错误。
- 修复:`extrapolation = inv_freq`、`interpolation = inv_freq / factor`(与 HF `_compute_yarn_parameters` 一致),删除失效的 linear_factor 逻辑。
- 回归测试:`test_yarn_interpolates_high_and_keeps_low_frequencies`。

### 7.2 [HIGH] `TwoDimensionalPositionEmbedding` 传入逐样本 id 时输出形状静默错误 — ✅ 已修复

- 位置:`two_dimensional.py:47-50`
- 问题:无条件 `.unsqueeze(0)`,传 `(batch, seq)` 的 block_ids/positions 时输出多出前导维 `(1,batch,seq,dim)`,无报错无校验。
- 修复:仅当嵌入张量比 x 少一维时才 unsqueeze。
- 回归测试:`test_two_dimensional_position_embedding_preserves_shape`。

### 7.3 [MEDIUM] `TwoDimensionalPositionEmbedding` 空输入崩溃、负 id 未校验 — ✅ 已修复

- 位置:`two_dimensional.py:43`
- 修复:`numel()` 防护 + 负 id 显式 ValueError。
- 回归测试:2 个。

### 7.4 [MEDIUM] `DynamicNTKRotaryEmbedding` dim=2 时除零 — ✅ 已修复

- 位置:`rope_scaling.py:212`
- 问题:`scale ** (dim / (dim - 2))`,dim=2 且 seq 超长按需缩放时 ZeroDivisionError。
- 修复:构造时 `dim <= 2` 显式 ValueError。
- 回归测试:`test_dynamic_ntk_requires_dim_greater_than_two`。

### 7.5 [MEDIUM] LongRoPE factor 语义与官方配置相反(文档陷阱)— 📝 文档声明

- 位置:`rope_scaling.py:368`
- 问题:实现是 `inv_freq * factors`(内置 preset 存倒数,自洽),但 docstring 称 tuple 可直接从官方 config 拷贝 —— 官方 `long_factor` 是 >1 的除法语义系数,照抄会反向作用(缩短而非扩展上下文)。
- 处理:数学不动,`LongRoPEPreset` / `register_longrope_preset` docstring 明确"tuple 存 inv_freq 的乘性系数,即官方 long_factor 的倒数"。

### 7.6 [LOW] RoPE 频率在 `x.dtype` 下计算,fp16 长序列精度丢失 — ✅ 已修复

- 位置:`rotary.py:67-68` 及 rope_scaling 中 YaRN/DynamicNTK/PositionInterpolation/LongRoPE
- 修复:positions/freqs/cos/sin 一律 fp32 计算,最后 cast 回 `x.dtype`(对齐 HF 做法)。

### 7.7 [LOW] factory `2d` 分支静默吞掉多余 kwargs — ✅ 已修复

- 位置:`position_factory.py:115-122`
- 修复:与 longrope 分支对齐,多余参数显式 ValueError。
- 回归测试:`test_positional_factory_2d_rejects_extra_kwargs`。

### 7.8 mypy 错误(positional 部分,约 16 处)— ✅ 已修复

- 位置:`rope_scaling.py`、`multimodal_rope.py:104`、`alibi.py`、`classic_position.py` 的 `register_buffer` 联合类型
- 修复:局部变量 + `assert isinstance(..., torch.Tensor)` 收窄。`mypy llminfra/positional` 已干净。

---

## 8. `llminfra/speculative_decoding`

### 8.1 [MEDIUM] `EagleSpeculator` 不校验 `hidden_states` 长度,过短时抛裸 IndexError — ✅ 已修复

- 位置:`eagle.py:50-67`
- 修复:显式 ValueError(hidden_states 形状、batch 一致性、序列长度 ≥ num_speculative_tokens);`MTPDecoder` 同步补 batch/维度校验。
- 回归测试:2 个。

### 8.2 [LOW] `NGramSpeculator` draft 是同一 token 重复 K 次 — ✅ 已修复

- 位置:`ngram.py:47`
- 问题:只取匹配点的单个后继 token 复制 K 份,多数 draft 第 2 步即被拒,削弱教学演示效果。
- 修复:拷贝匹配位置之后最多 K 个真实连续 token,不足时末位补齐。
- 回归测试:`test_ngram_speculator_drafts_observed_continuation`。

### 8.3 [LOW] `DSparkDecoder.forward` 有隐式状态副作用 — 📝 文档声明

- 位置:`dsflash.py:91-106`
- 说明:每次调用(含 eval/no_grad)都推进 scheduler,同一输入重复调用产生不同 draft 长度;docstring 已声明。

### 8.4 [LOW] Eagle/MTP 批量验证为 batch 级提前终止 — 📝 文档声明

- 位置:`eagle.py:67`、`mtp.py:43`
- 说明:任一行失配即整个 batch 停止(结果仍正确,只是比 base.py 逐行验证拿到的 token 少);docstring 已注明与 SpeculativeDecoder 的差异。

### 8.5 [LOW] `SpeculativeDecoder.forward` 逐行×逐步 Python 循环 — ✅ 已修复

- 位置:`base.py:75-100`
- 问题:B×K 次 `draft_model` 调用 + B 次 `target_model` 调用,而行间完全独立。
- 修复:draft 阶段整 batch 自回归(B×K → K 次 draft 调用),target 一次批量调用(B → 1 次),验证阶段保持逐行语义(贪婪/拒绝采样、bonus、padding 不变)。
- 回归测试:`test_batched_draft_matches_row_by_row` 等 2 个(temperature=0 下与逐行参考逐位相等,覆盖全接受+bonus / 部分接受+纠错 / 首步全拒;并断言 draft 调用次数 K 而非 B×K)。

---

## 汇总

| 模块 | HIGH | MEDIUM | LOW | 已修复 | 文档声明 |
|---|---|---|---|---|---|
| attention | 2 | 2 | 3 | 6 | 1 |
| flash_attention | 0 | 2 | 3 | 5 | 0 |
| inference | 2 | 2 | 2 | 6 | 0 |
| layers | 1 | 3 | 2 | 6 | 0 |
| moe | 0 | 2 | 4 | 6 | 0 |
| models | 0 | 2 | 4 | 6 | 0 |
| positional | 2 | 3 | 3 | 7 | 1 |
| speculative_decoding | 0 | 1 | 4 | 3 | 2 |
| **合计** | **7** | **17** | **25** | **45** | **4** |

验证基线对比:

| 指标 | 修复前 | 修复后 |
|---|---|---|
| pytest | 318 passed | **392 passed**(+74 回归测试),9 skipped(无 GPU 的 CUDA 测试) |
| ruff check | 干净 | 干净 |
| ruff format | 干净 | 干净 |
| mypy | 20 errors / 7 files | **0 errors** |

性能改善摘要(第三轮优化):

| 项 | 修复前 | 修复后 |
|---|---|---|
| LinearAttention 因果路径显存 | O(s·f·d)(s=4096 约 1GB) | O(c·f·d + c²)(约 0.5MB) |
| Mamba2 `_discretize` 峰值张量 | 2×(B,S,d_inner,d_state)(约 17GiB) | (B,d_inner,d_state)(约 4MiB) |
| FA4 wave 内存峰值 | 整行 (b,h,block_q,kv_len) | 单块 (b,h,block_q,block_kv) |
| BlockSparseIndexer | O(num_blocks) 次 kernel 调用 | 1 次批量 topk |
| SpeculativeDecoder draft | B×K 次 draft 调用 | K 次(batch 并行) |
