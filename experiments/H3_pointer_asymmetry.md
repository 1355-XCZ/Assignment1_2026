# H3: Pointer 层不对称连接的必要性与 Pass 角色分化

## 1. 研究问题

> QANet 的 Pointer 输出层采用不对称连接——p1 (start) = W1·[M1; M2]，p2 (end) = W2·[M1; M3]。三次共享权重的 Model Encoder pass 是否因为这种不对称的梯度信号而发展出功能特化？这种不对称设计是否优于对称替代方案？

## 2. 研究假设

**H3**: 尽管三次 Model Encoder pass 共享权重，Pointer 层的不对称连接会导致 M1、M2、M3 的表征产生有意义的分化——M2 特化于 answer start 位置的表征，M3 特化于 answer end 位置的表征，M1 提供通用的上下文编码。

- **预测 1**: M2 和 M3 的表征在 answer span 位置存在显著差异，但在远离 answer 的位置差异较小。
- **预测 2**: 互换 M2 和 M3（让 p1=[M1;M3], p2=[M1;M2]）会导致性能下降，但幅度可能不大（因为共享权重限制了特化程度）。
- **预测 3**: 去掉多次 pass（仅用 M1）会导致显著性能下降，说明多次 pass 不是冗余的。

## 3. 背景

### 3.1 QANet 的 Model Encoder 与 Pointer 结构

```python
# qanet.py 第 89-101 行
M1 = self.cq_resizer(X)
for i, enc in enumerate(self.model_enc_blks):       # Pass 1
    M1 = enc(M1, cmask, l=..., total_layers=...)

M2 = M1
for i, enc in enumerate(self.model_enc_blks):       # Pass 2
    M2 = enc(M2, cmask, l=..., total_layers=...)

M3 = M2
for i, enc in enumerate(self.model_enc_blks):       # Pass 3
    M3 = enc(M3, cmask, l=..., total_layers=...)

p1, p2 = self.out(M1, M2, M3, cmask)
```

```python
# heads.py 第 22-29 行
X1 = torch.cat([M1, M2], dim=1)   # [B, 2*d_model, L]
X2 = torch.cat([M1, M3], dim=1)   # [B, 2*d_model, L]
Y1 = torch.matmul(self.w1, X1)    # [B, L] → p1 (start logits)
Y2 = torch.matmul(self.w2, X2)    # [B, L] → p2 (end logits)
```

### 3.2 设计来源

这个不对称设计来自 BiDAF (Seo et al., 2016)。直觉是：
- M1 是基础表征（过了 7 个 encoder block 一次）
- M2 是二次精炼（基于 M1 再过一次 7 个 block）
- M3 是三次精炼（基于 M2 再过一次）
- Start 位置需要"基础+中间精炼"，End 位置需要"基础+深度精炼"

但这个直觉从未被实验验证过。

### 3.3 为什么共享权重是关键

三次 pass 使用**完全相同的 encoder 权重**。这意味着：
- 任何 M2 和 M3 的差异**不是**来自不同的参数，而是来自不同的输入（M1 vs M2）
- M2 = f(f(f(...f(X)...)))，M3 = f(f(f(...f(M2)...)))，其中 f 是同一个函数
- 问题变成：**同一个函数的不同迭代深度**是否产生了质的差异？

## 4. 实验设计

### Phase A: 替换实验（Eval-Time Pointer Reconfiguration）

**方法**: 使用训练好的模型（不重新训练），仅在推理时修改 Pointer 层的输入连接方式。

**实现**: 修改 `QANet.forward()` 的最后几行，或直接修改 `Pointer.forward()` 的调用方式。

```python
# 在 eval 时修改 forward pass
def forward_with_pointer_config(model, Cwid, Ccid, Qwid, Qcid, config="original"):
    # ... (正常计算 M1, M2, M3) ...
    
    if config == "original":
        p1, p2 = model.out(M1, M2, M3, cmask)        # p1=[M1;M2], p2=[M1;M3]
    elif config == "swap":
        p1, p2 = model.out(M1, M3, M2, cmask)        # p1=[M1;M3], p2=[M1;M2]
    elif config == "sym_M2":
        p1 = torch.matmul(model.out.w1, torch.cat([M1, M2], dim=1))
        p2 = torch.matmul(model.out.w2, torch.cat([M1, M2], dim=1))  # p2 也用 M2
        p1 = mask_logits(p1, cmask)
        p2 = mask_logits(p2, cmask)
    elif config == "sym_M3":
        p1 = torch.matmul(model.out.w1, torch.cat([M1, M3], dim=1))  # p1 也用 M3
        p2 = torch.matmul(model.out.w2, torch.cat([M1, M3], dim=1))
        p1 = mask_logits(p1, cmask)
        p2 = mask_logits(p2, cmask)
    elif config == "only_M1":
        p1 = torch.matmul(model.out.w1, torch.cat([M1, M1], dim=1))
        p2 = torch.matmul(model.out.w2, torch.cat([M1, M1], dim=1))
        p1 = mask_logits(p1, cmask)
        p2 = mask_logits(p2, cmask)
    elif config == "M2_M3":
        p1 = torch.matmul(model.out.w1, torch.cat([M2, M3], dim=1))  # 去掉 M1
        p2 = torch.matmul(model.out.w2, torch.cat([M2, M3], dim=1))
        p1 = mask_logits(p1, cmask)
        p2 = mask_logits(p2, cmask)
    
    return p1, p2
```

#### 配置清单与预期

| ID | 配置 | p1 | p2 | 预期 F1 变化 | 检验什么 |
|----|------|----|----|------------|---------|
| **A0** | 原版 | [M1;M2] | [M1;M3] | baseline | — |
| **A1** | 互换 | [M1;M3] | [M1;M2] | 小幅下降 | M2/M3 是否可互换 |
| **A2** | 对称-M2 | [M1;M2] | [M1;M2] | 下降 | M3 的独特贡献 |
| **A3** | 对称-M3 | [M1;M3] | [M1;M3] | 下降 | M2 的独特贡献 |
| **A4** | 仅 M1 | [M1;M1] | [M1;M1] | 显著下降 | 多次 pass 的必要性 |
| **A5** | 去 M1 | [M2;M3] | [M2;M3] | 下降 | M1 的基础作用 |

**重要说明**: 这些替换实验不需要重新训练。W1 和 W2 是训练时学到的，它们"期望"看到特定格式的输入。替换后性能下降既可能因为 M2/M3 确实不同，也可能因为 W1/W2 没有被训练来处理新的输入组合。因此替换实验的结果需要结合 Phase B 的表征分析来解释。

### Phase B: 表征相似性分析

**目的**: 直接测量 M1、M2、M3 的表征差异，独立于 Pointer 层的权重。

#### B1: 全局相似度

对 dev set 中的每个样本，计算 M1、M2、M3 之间的 token-level cosine similarity：

```python
def compute_pairwise_similarity(M_a, M_b, mask):
    """
    M_a, M_b: [B, d_model, L]
    mask: [B, L], True=PAD
    Returns: per-sample average cosine similarity over non-PAD tokens
    """
    M_a = M_a.transpose(1, 2)  # [B, L, d_model]
    M_b = M_b.transpose(1, 2)
    
    cos = F.cosine_similarity(M_a, M_b, dim=2)  # [B, L]
    cos = cos.masked_fill(mask, 0.0)
    
    valid_counts = (~mask).float().sum(dim=1)  # [B]
    avg_cos = cos.sum(dim=1) / valid_counts.clamp(min=1)  # [B]
    return avg_cos
```

产出: 三个相似度分布 `sim(M1,M2)`, `sim(M1,M3)`, `sim(M2,M3)`

#### B2: 按位置分区分析

将 tokens 分为四类：
- **Answer start** (y1 位置 ± 1 token)
- **Answer end** (y2 位置 ± 1 token)
- **Answer interior** (y1 < pos < y2)
- **Non-answer** (其他所有位置)

对每个分区分别计算 `sim(M2, M3)`：

```python
# 对每个样本
start_region = set(range(max(0, y1-1), min(L, y1+2)))
end_region   = set(range(max(0, y2-1), min(L, y2+2)))
interior     = set(range(y1+1, y2)) - start_region - end_region
non_answer   = set(range(L)) - start_region - end_region - interior

# 在每个区域内计算 M2 vs M3 的 cosine similarity
```

**预测**: 如果 M2 特化于 start、M3 特化于 end，那么：
- `sim(M2, M3)` 在 answer start 和 answer end 区域应该**低于** non-answer 区域
- M2 在 answer start 区域的表征与 M3 在 answer end 区域的表征应该在某种意义上"对偶"

#### B3: CKA (Centered Kernel Alignment) 分析

CKA 是一个更鲁棒的表征相似性指标，不受旋转和缩放的影响：

```python
def linear_CKA(X, Y):
    """
    X, Y: [n_samples, n_features]
    """
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    
    hsic_xy = (XtY ** 2).sum()
    hsic_xx = (XtX ** 2).sum()
    hsic_yy = (YtY ** 2).sum()
    
    return hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())
```

对整个 dev set 的所有 non-PAD tokens，计算：
- `CKA(M1, M2)`, `CKA(M1, M3)`, `CKA(M2, M3)`

产出：3×3 CKA 矩阵（对角线为 1.0）。

#### B4: 主成分分析可视化

对 M1、M2、M3 的表征做 PCA，投影到 2D：
- 选取 answer span 附近的 tokens
- 用不同颜色标记 M1/M2/M3 的 token 表征
- 观察三者是否形成可分离的聚类

### Phase C: 按 p1/p2 分解的精细分析

#### C1: M2 对 p1 vs p2 的贡献差异

由于原始设计中 M2 只参与 p1，M3 只参与 p2，我们无法直接测量"M2 对 p2 有多有用"。但可以通过以下方式间接评估：

在 **A1 (互换)** 配置下：
- 新的 p1 = W1·[M1; M3]，其 F1 相比原版 p1 下降了多少 → M3 替代 M2 做 start 的能力
- 新的 p2 = W2·[M1; M2]，其 F1 相比原版 p2 下降了多少 → M2 替代 M3 做 end 的能力

如果两者下降幅度不同，说明 M2 和 M3 的特化方向确实不同。

#### C2: Start-Only 和 End-Only 评估

对每个替换配置，分别报告：
- **Start F1**: 仅基于 p1 的预测质量（固定 p2 为原版预测）
- **End F1**: 仅基于 p2 的预测质量（固定 p1 为原版预测）

这需要修改评估流程，分别用原版和替换版的 p1/p2 组合来预测 span。

## 5. 控制变量

| 变量 | 设定 |
|------|------|
| 模型 | 与 H1, H2 相同的 checkpoint |
| 模型模式 | `eval()` |
| 评估数据 | SQuAD v1.1 dev set（**完整 dev set**，非子集） |
| 评估指标 | F1, EM（与标准 SQuAD 评估一致） |
| 替换实验 | 不重新训练，仅修改推理时的连接 |
| 表征分析 | 在 `torch.no_grad()` 下收集 M1, M2, M3 |

## 6. 评估样本

### Phase A (替换实验)

- 使用**完整 dev set**（~10,000 样本）
- 报告标准 F1 和 EM
- 因为不涉及随机噪声，不需要重复运行

### Phase B (表征分析)

- 使用完整 dev set 或随机 1000 个样本（CKA 计算量较大）
- Cosine similarity: 完整 dev set
- CKA: 随机 1000 样本
- PCA: 随机 100 样本用于可视化

## 7. 计算量估算

### Phase A
- 6 种配置 × 1 次 full dev set 评估
- 每种配置只需改最后一步（M1/M2/M3 已经计算好了）
- 可以一次 forward 收集 M1/M2/M3，然后对 6 种 Pointer 配置分别计算 p1/p2
- 总计: **1 次 full dev forward pass** + 6 次轻量 Pointer 计算
- 约 1-2 分钟

### Phase B
- 复用 Phase A 已收集的 M1/M2/M3
- Cosine similarity: O(n × L × d) → 秒级
- CKA: O(n² × d) → 对 1000 样本约 1 分钟
- PCA: 秒级

**总计约 5 分钟**。这是三个假设中计算成本最低的。

## 8. 可视化方案

### 8.1 替换实验结果表格

| 配置 | p1 | p2 | F1 | EM | ΔF1 | ΔEM |
|------|----|----|----|----|-----|-----|
| A0 原版 | [M1;M2] | [M1;M3] | xx.x | xx.x | — | — |
| A1 互换 | [M1;M3] | [M1;M2] | xx.x | xx.x | -x.x | -x.x |
| A2 对称-M2 | [M1;M2] | [M1;M2] | xx.x | xx.x | -x.x | -x.x |
| A3 对称-M3 | [M1;M3] | [M1;M3] | xx.x | xx.x | -x.x | -x.x |
| A4 仅 M1 | [M1;M1] | [M1;M1] | xx.x | xx.x | -x.x | -x.x |
| A5 去 M1 | [M2;M3] | [M2;M3] | xx.x | xx.x | -x.x | -x.x |

### 8.2 表征相似度分布图

三个 histogram/violin plot 并排：
- `sim(M1, M2)` 分布
- `sim(M1, M3)` 分布
- `sim(M2, M3)` 分布

标注均值和标准差。

### 8.3 按位置分区的相似度箱线图

```
         Answer Start   Answer End   Interior   Non-Answer
sim(M2,M3)  [boxplot]    [boxplot]   [boxplot]   [boxplot]
```

检验 M2/M3 在 answer 边界处是否分化。

### 8.4 CKA 热力图

```
     M1    M2    M3
M1  1.00  0.xx  0.xx
M2  0.xx  1.00  0.xx
M3  0.xx  0.xx  1.00
```

3×3 对称矩阵，热力图形式。

### 8.5 PCA 散点图

2D 散点图，answer span 附近的 tokens：
- 红色: M1 中的 token 表征
- 蓝色: M2 中的 token 表征
- 绿色: M3 中的 token 表征

## 9. 预期结果与分析框架

### 场景 1: 强特化（最有趣的结果）

- A1 (互换) 下降 > 2 F1
- `sim(M2, M3)` 在 answer 边界处显著低于 non-answer 位置
- CKA(M2, M3) < CKA(M1, M2) 或 CKA(M1, M3)

**解释**: 共享权重的 encoder 确实通过不同迭代深度学到了不同的表征，且 Pointer 的不对称连接提供了特化的梯度信号。这验证了 BiDAF 设计的合理性。

### 场景 2: 弱特化（最可能的结果）

- A1 (互换) 下降 < 1 F1
- `sim(M2, M3)` 整体很高（> 0.95），但在 answer 边界处稍低
- A4 (仅 M1) 显著下降（> 5 F1）

**解释**: M2 和 M3 几乎相同，不对称设计的收益不来自 pass 特化，而来自 (1) 更深的迭代处理和 (2) W1/W2 学到的不同读出方式。多次 pass 本身是必要的（A4 大幅下降），但不对称连接不是关键。

### 场景 3: M1 主导

- A4 (仅 M1) 下降不大（< 2 F1）
- A5 (去 M1) 大幅下降

**解释**: M1 已经包含了足够的信息，M2 和 M3 的额外处理收益有限。模型主要依赖第一次 pass 的输出。

### 综合分析要点

无论结果如何，都需要讨论：
1. **共享权重的限制**: 三次 pass 使用相同权重，本质上是同一函数的不同迭代深度。特化程度受限于函数的动力学特性。
2. **训练信号的传播**: p1 的 loss 梯度会通过 M2 反传到 Pass 2 的权重，p2 的 loss 梯度通过 M3 反传到 Pass 3。但权重共享意味着这两个梯度被**累加**后更新同一组权重，可能互相"稀释"了特化信号。
3. **替代设计**: 如果不共享权重（三组独立 encoder），特化程度可能更强，但参数量增加 3 倍。这是模型设计的 capacity vs efficiency tradeoff。

## 10. 与 H1, H2 的关系

| | H1 | H2 | H3 |
|--|----|----|-----|
| 研究对象 | Model Encoder 内部 | Embedding Encoder + CQ Attention | Pointer 层 + 三次 Pass |
| 方法论 | 因果追踪 | 因果追踪 | 替换实验 + 表征分析 |
| 覆盖阶段 | 后段 | 前段 + 中段 | 后段（不同视角） |
| 独特贡献 | 组件级因果图谱 | 双流与融合分析 | 架构设计选择的验证 |

三者互补：
- H1 发现各组件的因果角色
- H2 发现信息在融合前后的变化
- H3 验证一个具体的架构设计决策

## 11. 潜在风险与应对

| 风险 | 可能性 | 应对 |
|------|--------|------|
| W1/W2 对输入格式敏感，替换后性能下降只因权重不匹配 | 高 | 这就是为什么需要 Phase B 表征分析作为独立证据。如果表征分析显示 M2≈M3 但替换实验下降大 → 确认是权重不匹配而非真正的特化 |
| M1, M2, M3 几乎完全相同 | 中 | 报告 CKA ≈ 1.0 本身就是一个发现。讨论为什么三次迭代没有产生显著差异（可能是 7 个 block 已经让表征收敛到不动点） |
| 完整 dev set 评估中 F1 差异在噪声范围内 | 低 | 使用 bootstrap 重采样计算置信区间。10K 样本的 dev set 通常能检测到 > 0.5 F1 的差异 |
| PCA 可视化看不出明显模式 | 中 | PCA 只保留主要方差方向，可能遗漏细微差异。补充 t-SNE 或用 CKA 数值分析替代 |

## 12. 参考文献

- Yu, A. W., et al. (2018). QANet: Combining Local Convolution with Global Self-Attention. ICLR 2018.
- Seo, M. J., et al. (2016). Bidirectional Attention Flow for Machine Comprehension. (不对称 Pointer 设计的来源)
- Kornblith, S., et al. (2019). Similarity of Neural Network Representations Revisited. ICML 2019. (CKA 方法)
- Raghu, M., et al. (2017). SVCCA: Singular Vector Canonical Correlation Analysis. NeurIPS 2017. (表征相似性分析)
