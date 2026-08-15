# Attention 架构总结（2026-08 版）

> 本文是快速入口。完整研究综述、模型逐一分析、机制专题、结构化表、趋势与参考资料见 [attention_review_2026.md](./attention_review_2026.md)。

## 一、执行摘要

1. 截至 2026-08，主流 Attention 演进主线是 MHA -> MQA/GQA -> MLA -> MLA + 稀疏/线性混合。
2. 1M 上下文已成为前沿开源模型的标准战场：DeepSeek-V4、GLM-5.2、Kimi K3、MiniMax M3 均支持 1M。
3. GQA 仍是通用默认方案；MLA 用于 KV 显存压缩；Block Sparse 和 Linear Attention 用于超长上下文降本。
4. FlashAttention、PagedAttention、FlashMLA、FP4 indexer cache 等是系统级优化，不等同于新的 Attention 数学类型。
5. 闭源模型如 GPT-5.6 Sol、Claude Fable 5、Gemini 3.x 均未披露架构细节，不能把推断写成事实。

## 二、截至 2026-08 的关键版本速览

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

## 三、Attention 演进主线

- 2023 年：GQA 成为开源模型默认选择。
- 2024 年：DeepSeek-V2 提出 MLA，KV Cache 相对 MHA 减少约 93.3%。
- 2025 年：Qwen3-Next 和 Kimi Linear 验证 Gated DeltaNet/KDA 线性注意力；Llama 4 和 Gemma 3 验证 MoE + 局部注意力。
- 2026 年：DeepSeek-V4、GLM-5.2、MiniMax M3 把稀疏/压缩注意力推到 1M 上下文生产场景；Kimi K3 达到 2.8T 参数。

## 四、代码交付

仓库已新增：

- `attentionfactory/sliding_window_attention.py`：SWA 教学实现。
- `attentionfactory/block_sparse_attention.py`：Block Sparse Attention 教学实现。
- `attentionfactory/linear_attention.py`：Linear Attention 教学实现。
- `attentionfactory/paged_attention.py`：PagedAttention 的 block table 接口模拟。
- `tests/test_extended_attention.py`：对应单元测试。

运行：

```bash
python -m pytest -p no:capture -q
python -m ruff check attentionfactory tests
```

注意：当前机器默认 pytest 在 capture 初始化阶段会触发 macOS readline 相关段错误，使用 `-p no:capture` 可正常运行。
