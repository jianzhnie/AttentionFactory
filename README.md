# Attention

基于 PyTorch 实现的多种注意力机制模块集合，包含 MHA、MQA、GQA 和 MLA 四种经典注意力变体的简洁实现。

## 特性

- **统一接口**：所有注意力模块继承自统一的基类，接口一致，易于切换
- **完整实现**：
  - **MHA (Multi-Head Attention)** - 多头注意力 (Vaswani et al., 2017)
  - **MQA (Multi-Query Attention)** - 多查询注意力 (Shazeer, 2019)
  - **GQA (Group Query Attention)** - 分组查询注意力 (Chen et al., 2023)
  - **MLA (Multi-Head Latent Attention)** - 多头潜空间注意力
- **支持 Attention Mask**：支持自定义注意力掩码
- **权重返回**：可选返回注意力权重矩阵，便于可视化分析
- **Xavier 初始化**：默认使用 Xavier 均匀初始化

## 安装

### 依赖

- Python >= 3.9
- PyTorch >= 1.13

```bash
pip install torch
```

### 克隆仓库

```bash
git clone <repository-url>
cd Attention
```

## 快速开始

```python
import torch
from attention import MultiHeadAttention

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

多头注意力机制，出自论文 *Attention is All You Need。每个 Query、Key、Value 各有 `num_heads` 组独立的投影矩阵。

```python
from attention import MultiHeadAttention

attention = MultiHeadAttention(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    bias=True
)
```

| 参数说明：
- `hidden_size`：输入和输出特征维度
- `num_heads`：注意力头数量，必须能整除 `hidden_size`
- `dropout`：注意力权重的 dropout 概率，默认 0.1
- `bias`：线性投影是否使用偏置，默认 True

### 2. Multi-Query Attention (MQA)

多查询注意力机制，出自论文 *Fast Transformer Decoding: One Write-Head is All You Need。所有 Query 头共享同一组 Key 和 Value 投影，显著减少显存占用和推理计算量。

```python
from attention import MultiQueryAttention

attention = MultiQueryAttention(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    bias=True
)
```

### 3. Group Query Attention (GQA)

分组查询注意力机制，出自论文 *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints。MHA 与 MQA 的折中方案，将 Query 头分组，每组共享一组 Key/Value 头。

```python
from attention import GroupQueryAttention

attention = GroupQueryAttention(
    hidden_size=512,
    num_heads=8,
    num_kv_groups=2,  # G=2 时每组4头共享一组 KV
    dropout=0.1,
    bias=True
)
```

| 特殊参数：
- `num_kv_groups`：KV 分组数量，必须能整除 `num_heads`
  - 当 `num_kv_groups == num_heads`：等价于 MHA
  - 当 `num_kv_groups == 1`：等价于 MQA
  - 当 `1 < num_kv_groups < num_heads`：标准 GQA

### 4. Multi-Head Latent Attention (MLA)

多头潜空间注意力，在注意力计算前先将特征投影到潜空间，再映射回原空间进行注意力计算。

```python
from attention import MultiHeadLatentAttention

attention = MultiHeadLatentAttention(
    hidden_size=512,
    num_heads=8,
    q_latent_size=256,    # Query 潜空间维度
    kv_latent_size=256,     # Key/Value 潜空间维度
    dropout=0.0,
    bias=True
)
```

| 特殊参数：
- `q_latent_size`：Query 分支的潜空间维度
- `kv_latent_size`：Key/Value 分支的潜空间维度

## 使用 Attention Mask

```python
batch_size = 2
seq_len = 128
num_heads = 8

attention_mask = torch.ones(batch_size, num_heads, seq_len, seq_len)
attention_mask[:, :, -10:, :] = 0  # 屏蔽最后 10 个位置的查询

output = attention(hidden_state, attention_mask=attention_mask)
```

## 项目结构

```
Attention/
├── attention/
│   ├── __init__.py      # 模块导出
│   ├── base.py          # 注意力基类与工具函数
│   ├── mha.py           # 多头注意力
│   ├── mqa.py           # 多查询注意力
│   ├── gqa.py           # 分组查询注意力
│   └── mla.py           # 多头潜空间注意力
├── LICENSE
└── README.md
```

## 参考文献

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
2. Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." 2019.
3. Chen, W., et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." 2023.

## 许可证

[Apache License 2.0](LICENSE)
