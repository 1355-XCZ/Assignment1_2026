# H1: QANet Encoder Block 子组件因果角色图谱

## 1. 研究问题

> QANet Model Encoder 中的三类子组件——Depthwise Separable Convolution、Multi-Head Self-Attention、Feed-Forward Network——在不同层深度对答案 span 预测的因果贡献是否存在显著差异？这种差异如何随层深度和 encoder pass 变化？

## 2. 研究假设

**H1**: QANet 的 Conv、Self-Attention 和 FFN 在 Model Encoder 中扮演不同的因果角色，且这种角色分工随层深度变化：

- **预测 1**: Conv 的因果贡献集中在浅层（捕获局部 n-gram 模式用于答案边界识别）
- **预测 2**: Self-Attention 的因果贡献在深层增强（建模长距离依赖用于语义匹配）
- **预测 3**: FFN 的因果贡献在中间层达到峰值（类似 ROME 论文发现 MLP 在 GPT 中间层存储事实知识）

**与现有工作的关系**:
- QANet 论文 (Yu et al., 2018) Table 5 仅做了全局消融：去掉所有 Conv → -2.7 F1，去掉所有 Attn → -1.3 F1。这是粗粒度的，无法回答逐层差异的问题。
- ROME 论文 (Meng et al., 2022) 在 GPT 上做了 MLP vs Attention 的因果追踪，发现 MLP 在中间层、最后 subject token 处具有最强因果效应。本实验将其扩展到三组件架构。

## 3. 方法论

### 3.1 因果追踪流程

基于 Meng et al. (2022) Section 2.1 的方法适配：

#### Step 1: Clean Run

```python
model.eval()
with torch.no_grad():
    p1_clean, p2_clean = model(Cwid, Ccid, Qwid, Qcid)

# 记录正确答案的 span 概率
prob_clean = (
    F.softmax(p1_clean, dim=1)[range(B), y1] *
    F.softmax(p2_clean, dim=1)[range(B), y2]
)
```

同时通过 forward hooks 收集所有中间激活：
- 每个 encoder block 中每个子组件（conv_0, conv_1, self_attn, ffn）的输出

#### Step 2: Corrupted Run

在 `proj_conv` 输出后对 Context 表征注入高斯噪声：

```python
# C: [B, d_model, Lc] — proj_conv 的输出
noise_std = 3.0 * C.std().item()
C_corrupt = C + torch.randn_like(C) * noise_std
```

然后正常运行后续计算，得到 `prob_corrupt`。

噪声强度选择依据：ROME 论文 Section 2.1 使用 3 倍 embedding 标准差。

#### Step 3: Corrupted-with-Restoration Runs

对每个目标位置 (pass, block, component)：
1. 使用腐蚀后的 context 作为输入
2. 在目标组件处，用 hook 将其输出替换为 Step 1 收集的 clean 激活
3. 后续计算正常执行（不再干预）
4. 测量 `prob_restored`

#### Step 4: 计算 Indirect Effect

```python
IE(pass, block, component) = prob_restored - prob_corrupt
TE = prob_clean - prob_corrupt
NIE(pass, block, component) = IE / TE  # Normalized, ∈ [0, 1]
```

对多个样本取平均得到 AIE。

### 3.2 Hook 系统设计

QANet 的 `EncoderBlock.forward()` 结构如下（参考 `encoder.py` 第 119-151 行）：

```
输入 x
 │
 ├─ PosEncoder(x)
 │
 ├─ [Conv Loop] ─────────────── conv_0, conv_1 (各有 norm + dropout + conv + act + layer_dropout)
 │    每个 conv 的 hook 点: conv(out) 之后、act() 之后、layer_dropout 之前
 │
 ├─ [Self-Attention] ────────── norm + dropout + self_att + layer_dropout
 │    hook 点: self_att(out) 返回后、layer_dropout 之前
 │
 └─ [FFN] ───────────────────── norm + dropout + fc1 + act + fc2 + layer_dropout
      hook 点: fc2(out) 返回后、layer_dropout 之前
```

**关键决策：hook 点选在残差连接之前**

每个子组件都有残差连接 `out = f(x) + x`。我们恢复的是 `f(x)` 部分（即子组件自身的贡献），而非 `f(x) + x`（包含 skip connection 的完整输出）。这样才能隔离出该组件自身的因果效应。

**实现方式**: 需要修改 `EncoderBlock.forward()` 使其在每个子层计算后暴露中间结果，而非依赖 `nn.Module` 级别的 hook（因为残差连接在模块外面）。

建议方案：为 `EncoderBlock` 添加一个 `forward_with_hooks()` 方法，或在 `CausalTracer` 中重新实现 forward 逻辑以精确控制恢复点。

```python
def encoder_block_forward_traced(block, x, mask, l, total_layers,
                                  clean_acts=None, restore_target=None):
    """
    与 EncoderBlock.forward 相同的逻辑，但可以在指定子组件处
    替换为 clean_acts 中保存的激活。
    
    restore_target: None | "conv_0" | "conv_1" | "self_attn" | "ffn"
    """
    drop_scale = block.dropout / total_layers if total_layers > 0 else 0.0
    out = block.pos(x)

    for i, conv in enumerate(block.convs):
        res = out
        out = block.normb(out) if i == 0 else block.norms[i - 1](out)
        if i % 2 == 0:
            out = F.dropout(out, p=block.dropout, training=block.training)
        out = conv(out)
        out = block.act(out)
        
        # >>> RESTORATION POINT for conv_i <<<
        if restore_target == f"conv_{i}" and clean_acts is not None:
            out = clean_acts[f"conv_{i}"]
        
        out = block._layer_dropout(out, res, drop_scale * l)
        l += 1

    res = out
    out = block.norms[block.conv_num - 1](out)
    out = F.dropout(out, p=block.dropout, training=block.training)
    out = block.self_att(out, mask)
    
    # >>> RESTORATION POINT for self_attn <<<
    if restore_target == "self_attn" and clean_acts is not None:
        out = clean_acts["self_attn"]
    
    out = block._layer_dropout(out, res, drop_scale * l)
    l += 1

    res = out
    out = block.norme(out)
    out = F.dropout(out, p=block.dropout, training=block.training)
    out = out.transpose(1, 2)
    out = block.fc1(out)
    out = block.act(out)
    out = block.fc2(out)
    out = out.transpose(1, 2)
    
    # >>> RESTORATION POINT for ffn <<<
    if restore_target == "ffn" and clean_acts is not None:
        out = clean_acts["ffn"]
    
    out = block._layer_dropout(out, res, drop_scale * l)

    return out
```

### 3.3 遍历所有恢复点

QANet Model Encoder 的结构：
- 7 个 encoder blocks，共享权重
- 每个 block: 2 conv + 1 self_attn + 1 ffn = 4 个子组件
- 重复 3 次 pass → M1, M2, M3

**总恢复点数**: 3 passes × 7 blocks × 4 components = **84 个**

对每个恢复点，需要跑一次完整的 corrupted-with-restoration forward pass。

### 3.4 子分析：Start vs End 分解

对每个恢复点，分别计算：

```python
IE_p1(pass, block, comp) = softmax(p1_restored)[y1] - softmax(p1_corrupt)[y1]
IE_p2(pass, block, comp) = softmax(p2_restored)[y2] - softmax(p2_corrupt)[y2]
```

这不需要额外的 forward pass，只是从同一次恢复运行中分别提取 p1 和 p2 的指标。

### 3.5 子分析：按 Answer 长度分层

将评估样本按 answer span 长度分为三组：
- 短答案: len ∈ [1, 2] tokens
- 中等答案: len ∈ [3, 5] tokens
- 长答案: len ≥ 6 tokens

对每组分别计算 AIE，观察组件重要性是否随答案长度变化。

预测：长答案场景下 Self-Attention 的 AIE 增加（需要更长距离的上下文理解来确定 end position）。

## 4. 控制变量

| 变量 | 设定 | 说明 |
|------|------|------|
| 模型 checkpoint | Stage 1&2 修复后的 best checkpoint | 固定，不重新训练 |
| 模型模式 | `model.eval()` | 禁用 dropout 和 stochastic depth |
| 推理模式 | `torch.no_grad()` | 不计算梯度 |
| 噪声类型 | 高斯 $\epsilon \sim \mathcal{N}(0, \nu^2)$ | 与 ROME 论文一致 |
| 噪声强度 $\nu$ | $3 \times \text{std}(C)$, 其中 $C$ 是 clean 的 proj_conv 输出 | 每个样本独立计算 std |
| 随机种子 | 固定 seed=42 | 噪声可复现 |
| 恢复粒度 | 单一子组件 | 每次只恢复一个 (pass, block, component) |

## 5. 评估样本

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据来源 | SQuAD v1.1 dev set | 不使用训练集，避免过拟合效应 |
| 样本数量 | 300 | 平衡统计可靠性与计算成本 |
| 选取方式 | 随机抽取，但确保三组 answer 长度均有 ≥50 个样本 | 支持分层分析 |
| 噪声重复次数 | 每样本 3 次（不同随机种子），取均值 | 减少噪声采样方差 |
| 前置过滤 | 仅保留 clean run 中 $P_{\text{span}} > 0.01$ 的样本 | 排除模型本就无法回答的样本 |

## 6. 计算量估算

- 每个样本需要: 1 (clean) + 1 (corrupt) + 84 (restorations) × 3 (噪声重复) = **254 次 forward pass**
- 300 个样本: 300 × 254 = **76,200 次 forward pass**
- QANet 单次 forward pass（batch=1）约 5-10ms (GPU) → 总计约 6-13 分钟
- 使用 batch=1 逐样本处理（因为每个样本的恢复激活不同）
- 如果计算资源紧张，可减少样本数到 100（约 2-4 分钟）

## 7. 可视化方案

### 7.1 主图：因果热力图

```
         Pass 1              Pass 2              Pass 3
     B0 B1 B2 B3 B4 B5 B6  B0 B1 B2 B3 B4 B5 B6  B0 B1 B2 B3 B4 B5 B6
Conv0 [                    |                    |                    ]
Conv1 [                    |                    |                    ]
Attn  [                    |                    |                    ]
FFN   [                    |                    |                    ]
```

- 色图: `RdYlBu_r` (红=高 AIE, 蓝=低 AIE)
- 尺寸: 21 列 × 4 行
- 标注数值在高 AIE 的格子中

### 7.2 组件级汇总条形图

```python
# 对每个组件类型，将所有 (pass, block) 的 AIE 求和
aie_conv = sum of AIE for all conv_0 + conv_1 across all passes/blocks
aie_attn = sum of AIE for all self_attn across all passes/blocks
aie_ffn  = sum of AIE for all ffn across all passes/blocks
```

3 个 bar + 95% CI error bars。

### 7.3 逐层趋势折线图

- X 轴: Block 0-6
- Y 轴: AIE
- 3 条线: Conv (avg of conv_0, conv_1), Attn, FFN
- 分 3 个子图 (Pass 1, Pass 2, Pass 3) 或重叠在一张图上

### 7.4 Start vs End 差异图

- 与主图相同布局
- 颜色: `AIE_p1 - AIE_p2` (红=偏向 start, 蓝=偏向 end)
- 使用 diverging colormap `RdBu`

### 7.5 Answer 长度分层折线图

- 3 个子图 (短/中/长答案)
- 每个子图 3 条线 (Conv/Attn/FFN)
- X 轴: Block 0-6 (可在 Pass 级别聚合)

## 8. 预期结果与分析框架

### 如果假设成立

- 热力图呈现类似 ROME 论文 Figure 2 的结构：某些 (block, component) 组合显著亮于其他
- Conv 在浅层 block 有更高 AIE，Attn 在深层 block 有更高 AIE
- FFN 在中间层有峰值
- Start vs End 差异图显示 Conv 偏向 start (答案边界的局部 pattern)

### 如果假设部分不成立

- 可能所有组件在所有层的 AIE 近似均匀 → 说明 QANet 的信息处理是分布式的，没有明显的功能定位
- 可能某个 Pass 的 AIE 远高于其他两个 → 与 H3 的研究产生关联
- 可能 FFN 的因果效应非常低 → 与 ROME 的发现矛盾，说明 QA 任务与事实回忆任务的机制不同

### 与 QANet 论文消融实验的对比

- 论文: 去掉全部 Conv → -2.7 F1, 去掉全部 Attn → -1.3 F1
- 本实验: 提供逐层细粒度分解，如果 Conv 的总 AIE > Attn 的总 AIE，则与论文一致
- 如果不一致，需要讨论消融 vs 因果追踪两种方法的差异（消融会移除组件导致 cascading failure，因果追踪只恢复单组件）

## 9. 潜在风险与应对

| 风险 | 可能性 | 应对 |
|------|--------|------|
| AIE 值普遍接近 0（噪声太小） | 中 | 调高噪声倍数到 5×std，或改用 zero-out corruption |
| AIE 值普遍接近 TE（噪声太大） | 低 | 降低噪声倍数到 1×std |
| 所有组件 AIE 几乎相同（无差异） | 中 | 这本身是一个有意义的发现（分布式处理），报告并讨论 |
| Forward pass 中 eval 模式下 layer_dropout 行为 | 低 | `_layer_dropout` 在 `not self.training` 时直接返回 `inputs + residual`，不会 drop。已确认。 |
| 共享权重导致三次 pass 的 hook 冲突 | 中 | 需要用 pass index 区分 hook，不能简单对 module 注册 hook |

## 10. 参考文献

- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.
- Yu, A. W., Dohan, D., Luong, M. T., et al. (2018). QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension. ICLR 2018.
- Pearl, J. (2001). Direct and indirect effects.
- Vig, J., et al. (2020). Investigating gender bias in language models using causal mediation analysis. NeurIPS 2020.
