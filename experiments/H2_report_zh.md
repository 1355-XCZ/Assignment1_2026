# H2：Context 与 Question 双流信息流分析

## 假设

> Question 流的腐蚀对模型的破坏性大于 Context 流。CQ Attention 是两流合并的关键信息瓶颈。Context 的信息集中在答案区间，而 Question 的信息分布在所有 token 上，因此 Question 具有更高的信息密度。

## 背景

QANet 将 Context (C) 和 Question (Q) 分别通过独立的 Embedding Encoder 处理，然后在 CQ Attention（BiDAF 风格的上下文-问题注意力）处合并：

```
Context → EmbEnc_C ─┐
                     ├→ CQ Attention → CQ Resizer → Model Encoder (×3 passes)
Question → EmbEnc_Q ─┘
```

**核心问题**：C 和 Q 是否等重要？CQ Attention 是否真的是信息瓶颈？

---

## 实验 1：ROME 风格因果追踪（3 种腐蚀条件）

**方法**：对每个样本，分别腐蚀 C、Q 或同时腐蚀（高斯噪声，σ = 3× 输入标准差），然后恢复各组件的干净激活并测量恢复程度。

- **TE（总效应）**：P(span|clean) − P(span|corrupted)
- **IE（间接效应）**：P(span|corrupted+恢复组件) − P(span|corrupted)
- **NIE（归一化 IE）**：IE / TE

**样本数**：200，噪声重复：3

### 分析框架

1. **TE 比较**：TE(Q) > TE(C) → 问题流的因果权重更大
2. **可加性检验**：TE(both) vs TE(C) + TE(Q) → 亚可加 = 信息重叠；超可加 = 协同效应
3. **CQ Attention NIE**：所有条件下 NIE > 70% → 确认为信息瓶颈
4. **各组件 NIE**：哪些 EmbEnc 子层对各流最重要？

---

## 实验 2：CQ Attention 消融（独立瓶颈验证）

**方法**：eval 时将 CQ Attention 输出置零，评估全 dev 集 F1/EM。

该实验独立于因果追踪（因果追踪使用概率指标和子样本），提供 F1/EM 级别的直接验证。

### 分析框架

1. **F1 下降幅度**：> 50% → 灾难性，确认 CQ-Att 是瓶颈
2. **与实验 1 交叉验证**：因果追踪测量信息*流动*（恢复）；消融测量信息*必要性*（移除）
3. **残差贡献**：CQ Attention 之后有残差连接，剩余 F1 表示通过残差的信息泄漏

### 局限性

- 零化消融同时移除信息内容和激活幅度
- 确认必要性但非充分性

---

## 实验 3：选择性腐蚀 — 信息定位分析

**方法**：选择性地只腐蚀 Context 中的特定位置，确定 Context 信息是集中的（聚焦在答案区间）还是分散的（分布在所有 token 上）。

**条件**：
- `ans_only`：只腐蚀答案区间位置（y1 到 y2）
- `non_ans_only`：只腐蚀非答案、非填充位置
- `full_context`：腐蚀所有 Context 位置

**样本数**：200，噪声重复：3

### 分析框架

1. **每 token TE 比较**：答案位置 TE/token >> 非答案位置 → Context 信息高度定位
2. **可加性**：TE(ans) + TE(non_ans) vs TE(full) → 信息交互模式
3. **CQ Attention 恢复模式**：瓶颈对定位型和分布型信息的中介是否不同？
4. **解释 TE 不对称**：如果 Context 信息集中在约 N 个答案 token，而 Question 信息分布在约 M 个 token（M ≈ 全部 Q token），则 Question 的全局信息密度更高 → 解释了为什么 Q 腐蚀破坏性更大

### 局限性

- 腐蚀答案区间 token 直接移除了目标信号（高 TE 是预期的、较 trivial 的）
- 真正有信息量的发现是非答案腐蚀是否具有显著影响
- 基于位置的噪声无法完美分离位置信息和语义信息

---

## 综合分析

### 证据链

1. **实验 1（因果追踪）**：建立 C/Q 流的 TE 不对称性；CQ-Att 具有高 NIE（信息流动证据）
2. **实验 2（CQ 消融）**：独立确认 CQ-Att 是性能所必需的（信息必要性证据）
3. **实验 3（选择性腐蚀）**：通过信息定位分析解释*为什么*存在不对称性

### 多方法一致性

| 结论 | 实验 1 证据 | 实验 2 证据 | 实验 3 证据 |
|---|---|---|---|
| CQ-Att 是瓶颈 | NIE > 70% | F1 下降 > 50% | CQ 恢复因位置而异 |
| Q 比 C 更重要 | TE(Q) > TE(C) | N/A | 信息密度分析 |
| Context 信息定位化 | N/A | N/A | TE/token 比率 |

### 方法论说明

- 三个实验使用不同指标（概率 TE/IE、F1/EM、每 token TE 密度），避免单一方法偏差
- 因果追踪和消融互补：一个恢复信息，另一个移除信息
- 选择性腐蚀提供了全局指标无法捕获的机制级理解
