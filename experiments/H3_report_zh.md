# H3：Pointer 层非对称接线与 Pass 角色分化

## 假设

> QANet 的非对称 Pointer 设计（M1+M2 → start, M1+M3 → end）具有因果合理性：M2 和 M3 编码不同的信息，M2 专注于 start 边界特征，M3 专注于 end 边界特征。非对称接线利用了这种专业化。

## 背景

QANet 的 Model Encoder 产生三个中间表示：
- **M1**：Pass 1 输出（7 个 block）
- **M2**：Pass 2 输出（7 个 block）
- **M3**：Pass 3 输出（7 个 block）

Pointer 层计算：
```
P(start) = softmax(w1 · [M1; M2])
P(end)   = softmax(w2 · [M1; M3])
```

**核心问题**：为什么 M2 用于 start，M3 用于 end？这是任意设计还是训练诱导了真正的 start/end 专业化？

---

## Phase A：Pointer 接线替换（消融实验）

**方法**：eval 时替换 Pointer 层的 M1/M2/M3 输入，在全 dev 集上测量 F1/EM。这是**因果干预** — 改变接线并观察效果。

**测试配置**：

| 配置 | Start 输入 | End 输入 | 设计意图 |
|---|---|---|---|
| original | M1 + M2 | M1 + M3 | 论文设计 |
| swap | M1 + M3 | M1 + M2 | 测试 M2/M3 是否可互换 |
| sym\_M2 | M1 + M2 | M1 + M2 | 双侧使用 M2 |
| sym\_M3 | M1 + M3 | M1 + M3 | 双侧使用 M3 |
| only\_M1 | M1 + M1 | M1 + M1 | 移除 M2/M3 贡献 |
| no\_M1 | M2 + M3 | M2 + M3 | 移除 M1 贡献 |

### 分析框架

1. **swap vs original**：swap 下降 → M2 和 M3 不可互换 → 编码不同信息
2. **sym\_M2 vs sym\_M3**：哪个 pass 更通用？
3. **only\_M1**：量化 M2/M3 的联合贡献
4. **no\_M1**：量化 M1 的贡献
5. **非对称优势**：original 是否严格优于所有对称替代？

### 局限性

- eval 时接线替换存在分布不匹配：Pointer 权重是为 original 接线优化的
- 差的 swap 性能证明 M2≠M3，但不直接证明 M2=start 和 M3=end

---

## Phase B：表示相似性分析（CKA + 余弦相似度）

**方法**：收集 M1/M2/M3 token 表示，计算成对相似性指标。

**指标**：
- **全局余弦相似度**：每 token 余弦相似度，跨 token 和样本平均
- **位置分层余弦（M2 vs M3）**：答案起始/内部/结束/非答案位置的相似度
- **线性 CKA**：数据集级表示相似性（对线性变换鲁棒）

### 分析框架

1. **渐进变换**：M2\_M3 CKA > M1\_M3 CKA → 每个 pass 渐进变换表示
2. **M2/M3 分歧**：CKA(M2, M3) < 1 → pass 未收敛到相同表示
3. **位置分层余弦**：答案边界处 M2-M3 相似度更低 → 在关键位置分歧更大

### 局限性

- CKA 和余弦相似度是**相关性**指标，非因果性
- 高相似 ≠ 相同信息；低相似 ≠ 互补信息
- 无法直接证明专业化方向

---

## 实验 C：线性探针 — Start/End Token 专业化

**方法**：在 M1/M2/M3 的每 token 表示上训练线性分类器来预测：
- `is_start_token`：token 位置 = 答案起始位置则为 1
- `is_end_token`：token 位置 = 答案结束位置则为 1

这提供了专业化声明的**直接证据**：如果 M2 更擅长预测 start token，M3 更擅长预测 end token，则非对称接线具有因果合理性。

**样本数**：500（80/20 训练/测试划分）
**分类器**：LogisticRegression（平衡类权重，C=1.0）
**指标**：ROC-AUC（处理极端类别不平衡）

### 分析框架

1. **如果 M2 start 优势 > 0 且 M3 end 优势 > 0**：专业化得到确认
2. **如果两个优势接近零**：线性可检测的专业化不存在，Pointer 的双线性投影可能利用了非线性差异
3. **渐进编码**：比较 M1→M2→M3 对 start 和 end 的 AUC 轨迹

### 专业化指标

- **M2 start 优势** = AUC(M2→start) − AUC(M3→start)
- **M3 end 优势** = AUC(M3→end) − AUC(M2→end)

### 局限性

- 线性探针只检测线性可解码的信息
- 极端类别不平衡（约 0.25% 正样本）
- Pointer 使用双线性投影 w1·[M1;M2]，而非线性读出 — 探针测试的访问机制比模型实际使用的更简单
- Start/end 标签是二元的（单一位置），但模型可能将答案边界编码为跨相邻位置的软梯度

---

## 综合分析

### 证据链

1. **Phase A（接线消融）**：swap 下降 → M2 ≠ M3（专业化的必要条件）
2. **Phase B（CKA/余弦）**：M2 和 M3 相似但不相同；答案边界处分歧更大 → 关键位置存在结构性差异
3. **实验 C（线性探针）**：直接测试 M2→start / M3→end 的方向性专业化

### 多方法一致性

| 结论 | Phase A | Phase B | 实验 C |
|---|---|---|---|
| M2 ≠ M3 | swap ΔF1 < 0 | CKA < 1 | 不同的 AUC 特征 |
| M2 → start | swap 损害 start | start 处 M2-M3 相似度更低 | AUC(M2→start) > AUC(M3→start) |
| M3 → end | swap 损害 end | end 处 M2-M3 相似度更低 | AUC(M3→end) > AUC(M2→end) |
| 非对称合理 | original > 所有对称 | — | 专业化分数 > 0 |

### 方法论说明

- 三个方法在不同层级验证假设：行为级（F1/EM）、表示级（CKA/余弦）、信息级（线性探针）
- Phase A 是因果的（干预），Phase B 是相关的，实验 C 是混合的（探测因果可读性）
- eval 时接线替换有分布不匹配的注意事项 — Pointer 权重是为 original 接线训练的
