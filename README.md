# AttentionFactory

基于 PyTorch 实现的多种注意力机制模块集合，包含 MHA、MQA、GQA 和 MLA 四种经典注意力变体的简洁实现，以及 FlashAttention v1–v4 的教学用分块（tiled）实现。

## 特性

- **统一接口**：所有注意力模块继承自统一的基类 `BaseAttention`，接口一致，易于切换
- **完整实现**：
  - **MHA (Multi-Head Attention)** - 多头注意力 (Vaswani et al., 2017)
  - **MQA (Multi-Query Attention)** - 多查询注意力 (Shazeer, 2019)
  - **GQA (Group Query Attention)** - 分组查询注意力 (Chen et al., 2023)
  - **MLA (Multi-Head Latent Attention)** - 多头潜空间注意力
- **FlashAttention 教学实现**：`flashattention` 子包包含 FA1–FA4 四个版本的纯 PyTorch 在线 softmax 分块实现（含 forward/backward），用于理解各版本算法结构的演进
- **支持 Attention Mask**：支持广播的注意力掩码（padding mask 或完整 mask）
- **权重返回**：可选返回注意力权重矩阵，便于可视化分析
- **Xavier 初始化**：默认使用 Xavier 均匀初始化
- **完整测试**：`tests/` 包含数值正确性、掩码语义、梯度流等 90+ 个测试用例

## 安装

### 依赖

- Python >= 3.10
- PyTorch

```bash
pip install torch
```

### 克隆仓库

```bash
git clone https://github.com/jianzhnie/AttentionFactory.git
cd AttentionFactory
```

## 快速开始

```python
import torch
from attentionfactory import MultiHeadAttention

hidden_size = 512
num_heads = 8
batch_size = 2
seq_len = 128

attention = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
hidden_state = torch.randn(batch_size, seq_len, hidden_size)

output = attention(hidden_state)
print(output.shape)  # torch.Size([2, 128, 512])

output, attn_weights = attention(hidden_state, return_attention_weights=True)
print(attn_weights.shape)  # torch.Size([2, 8, 128, 128])
```

## 模块详解

### 1. Multi-Head Attention (MHA)

多头注意力机制，出自论文 *Attention is All You Need*。每个 Query、Key、Value 各有 `num_heads` 组独立的投影矩阵。

```python
from attentionfactory import MultiHeadAttention

attention = MultiHeadAttention(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    bias=True
)
```

参数说明：
- `hidden_size`：输入和输出特征维度
- `num_heads`：注意力头数量，必须能整除 `hidden_size`
- `dropout`：注意力权重的 dropout 概率，默认 0.1
- `bias`：线性投影是否使用偏置，默认 True

### 2. Multi-Query Attention (MQA)

多查询注意力机制，出自论文 *Fast Transformer Decoding: One Write-Head is All You Need*。所有 Query 头共享同一组 Key 和 Value 投影，显著减少显存占用和推理计算量。

```python
from attentionfactory import MultiQueryAttention

attention = MultiQueryAttention(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    bias=True
)
```

### 3. Group Query Attention (GQA)

分组查询注意力机制，出自论文 *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。MHA 与 MQA 的折中方案，将 Query 头分组，每组共享一组 Key/Value 头。

```python
from attentionfactory import GroupQueryAttention

attention = GroupQueryAttention(
    hidden_size=512,
    num_heads=8,
    num_kv_groups=2,  # G=2 时每组 4 头共享一组 KV
    dropout=0.1,
    bias=True
)
```

特殊参数：
- `num_kv_groups`：KV 分组数量，必须能整除 `num_heads`
  - 当 `num_kv_groups == num_heads`：等价于 MHA
  - 当 `num_kv_groups == 1`：等价于 MQA
  - 当 `1 < num_kv_groups < num_heads`：标准 GQA

### 4. Multi-Head Latent Attention (MLA)

多头潜空间注意力，在注意力计算前先将特征投影到潜空间，再映射回原空间进行注意力计算。

```python
from attentionfactory import MultiHeadLatentAttention

attention = MultiHeadLatentAttention(
    hidden_size=512,
    num_heads=8,
    q_latent_size=256,    # Query 潜空间维度
    kv_latent_size=256,   # Key/Value 潜空间维度
    dropout=0.0,
    bias=True
)
```

特殊参数：
- `q_latent_size`：Query 分支的潜空间维度
- `kv_latent_size`：Key/Value 分支的潜空间维度

### 5. FlashAttention（教学实现）

`attentionfactory.flashattention` 子包以纯 PyTorch 实现了 FlashAttention v1–v4 的核心算法结构（在线 softmax、分块循环、LSE 重算梯度），用于教学目的。提供与 PyTorch 一致的调用方式——函数式接口 `flash_attention`（类似 `F.scaled_dot_product_attention`）和 `nn.Module` 包装，均支持 autograd：

```python
import torch
from attentionfactory import FlashAttention, flash_attention

q = torch.randn(2, 8, 128, 64, requires_grad=True)
k = torch.randn(2, 8, 128, 64, requires_grad=True)
v = torch.randn(2, 8, 128, 64, requires_grad=True)

# 函数式调用，version 选择 FA 版本（fa1/fa2/fa3/fa4）
out = flash_attention(q, k, v, version="fa2", causal=True)
out.sum().backward()  # 梯度走 fa2 的分块 backward

# 或者像普通 nn.Module 一样使用
attn = FlashAttention(version="fa3", causal=True)
out = attn(q, k, v)
```

四个版本（`fa1`/`fa2`/`fa3`/`fa4`）也各自暴露底层的 `forward` / `backward` 以及可微分的 `flash_attention_v1`–`flash_attention_v4` 函数，差异体现在循环结构、工作划分与调度方式上，分别对应各代论文的算法改进点。

## 使用 Attention Mask

掩码遵循 `1（True）= 保留，0（False）= 屏蔽` 约定，并对注意力分数广播，因此 padding mask 和完整 mask 都支持：

```python
batch_size = 2
seq_len = 128

# padding mask：屏蔽每个样本末尾的若干 key 位置
attention_mask = torch.ones(batch_size, 1, 1, seq_len)
attention_mask[0, 0, 0, -10:] = 0  # 屏蔽第一个样本的最后 10 个 key

output = attention(hidden_state, attention_mask=attention_mask)
```

注意：若某一行所有 key 均被屏蔽，该行的注意力权重和输出定义为 0（而非 NaN）。

## 运行测试

```bash
python -m pytest tests/
```

## 项目结构

```
AttentionFactory/
├── attentionfactory/
│   ├── __init__.py          # 模块导出
│   ├── base.py              # 注意力基类与共享工具函数
│   ├── mha.py               # 多头注意力
│   ├── mqa.py               # 多查询注意力
│   ├── gqa.py               # 分组查询注意力
│   ├── mla.py               # 多头潜空间注意力
│   └── flashattention/      # FlashAttention v1-v4 教学实现
│       ├── fa1.py ... fa4.py
│       └── common/          # 共享的在线 softmax / 掩码 / 分块原语
├── tests/                   # pytest 测试套件
├── pyproject.toml
├── LICENSE
└── README.md
```

## 参考文献

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
2. Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." 2019.
3. Chen, W., et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." 2023.
4. Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." 2022.
5. Dao, T. "FlashAttention-2: Better Attention with Better Parallelism and Work Partitioning." 2023.
6. Shah, J., et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024.

## 许可证

[Apache License 2.0](LICENSE)
