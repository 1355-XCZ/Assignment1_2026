# Stage 3 实验策划：基于因果追踪的 QANet 组件分析

## 总体背景

本实验将 Meng et al. (NeurIPS 2022) 的 **Causal Tracing** 方法从 GPT（自回归 Transformer）适配到 **QANet**（卷积 + 自注意力阅读理解模型），旨在通过因果中介分析精确定位 QANet 中各子组件（Depthwise Separable Conv, Self-Attention, FFN）在答案预测中的因果作用。

### 与原论文的核心差异

| 维度 | 原论文 (GPT) | 本实验 (QANet) |
|------|-------------|---------------|
| 任务 | 事实补全 (next-token prediction) | 抽取式 QA (span prediction) |
| 架构 | [Attention + MLP] × L | [Conv × N + Attention + FFN] × L |
| 输入 | 单一 token 序列 | 双流 (Context + Question)，经 CQ Attention 融合 |
| 输出 | 下一个 token 的概率 | Answer span 的起止位置概率 p1, p2 |
| 组件分解 | 2 类 (MLP, Attention) | 3 类 (Conv, Self-Attention, FFN) |
| 权重共享 | 无 | Model Encoder 7 个块共享权重，重复 3 次 → M1, M2, M3 |

---

## 实验指标定义

### 核心概率指标

给定一个 QA 样本 `(context, question, y1, y2)`，其中 `y1`, `y2` 是真实答案的起止位置：

$$P_{\text{span}} = P(y_1) \cdot P(y_2) = \text{softmax}(p_1)[y_1] \cdot \text{softmax}(p_2)[y_2]$$

### 因果效应计算

- **Total Effect (TE):**
  $$\text{TE} = P_{\text{clean}}[\text{span}] - P_{\text{corrupt}}[\text{span}]$$

- **Indirect Effect (IE) of component $c$ at layer $l$:**
  $$\text{IE}(c, l) = P_{\text{corrupt, restore } c^{(l)}}[\text{span}] - P_{\text{corrupt}}[\text{span}]$$

- **Normalized IE:**
  $$\text{NIE}(c, l) = \frac{\text{IE}(c, l)}{\text{TE}}$$
  值域 [0, 1]，表示该组件恢复了多少比例的 total effect。

---

## 实验一：QANet Encoder Block 子组件因果追踪

### 1.1 研究假设

**H1**: 在 QANet 的 Encoder Block 中，Depthwise Separable Convolution、Self-Attention 和 FFN 对答案预测的因果贡献存在显著差异，且这种差异随层深度变化。

**具体预测**（基于 QANet 论文的设计动机）：
- 卷积在浅层贡献更大（捕获局部 n-gram 模式，对定位答案边界至关重要）
- 自注意力在深层贡献更大（建模长距离依赖，对理解问题-上下文关系至关重要）
- FFN 在中间层贡献最大（类似 ROME 论文发现 MLP 在 GPT 中间层存储事实）

### 1.2 实验设计

#### 控制变量
- 使用**同一个训练好的 QANet checkpoint**
- 固定随机种子
- 固定噪声强度 ν（设为 context embedding 标准差的 3 倍，与 ROME 论文一致）
- 在相同的评估样本集上计算所有指标

#### 腐蚀策略 (Corruption)

对 context 的 word embedding 注入高斯噪声：
```python
# 在 model.forward() 中，proj_conv 之后的 C 上加噪声
# C: [B, d_model, Lc]
noise_std = 3.0 * C.std().item()
C_corrupt = C + torch.randn_like(C) * noise_std
```

#### 恢复策略 (Restoration)

QANet Model Encoder 有 7 个 block，每个 block 内有 3 类子组件：
- `Conv[0]`, `Conv[1]` (2 个 depthwise separable conv)
- `Self-Attention` (multi-head attention)
- `FFN` (2-layer feed-forward)

对每个 block `b ∈ {0, 1, ..., 6}` 的每个子组件 `c ∈ {conv_0, conv_1, self_attn, ffn}`：
1. 运行 clean forward pass，记录该子组件的输出 `out_clean[b][c]`
2. 运行 corrupt forward pass，在第 `b` 个 block 的第 `c` 个子组件输出处，用 `out_clean[b][c]` 替换
3. 继续完成后续计算，得到 `p1, p2`
4. 计算 `IE(b, c)`

#### 具体实现方式：Hook-Based Intervention

```python
class CausalTracer:
    def __init__(self, model):
        self.model = model
        self.clean_activations = {}
        self.hooks = []

    def collect_clean(self, sample):
        """第一步：clean run，收集所有中间激活"""
        handles = []
        # 对每个 encoder block 的每个子组件注册 forward hook
        for blk_idx, blk in enumerate(self.model.model_enc_blks):
            for conv_idx, conv in enumerate(blk.convs):
                handle = conv.register_forward_hook(
                    self._save_hook(f"blk{blk_idx}_conv{conv_idx}")
                )
                handles.append(handle)
            handles.append(blk.self_att.register_forward_hook(
                self._save_hook(f"blk{blk_idx}_attn")
            ))
            handles.append(blk.fc2.register_forward_hook(
                self._save_hook(f"blk{blk_idx}_ffn")
            ))
        
        with torch.no_grad():
            p1, p2 = self.model(*sample)
        
        for h in handles:
            h.remove()
        return p1, p2

    def corrupt_and_restore(self, sample, restore_key):
        """第二步：corrupt + 恢复指定组件"""
        # 注册 corruption hook（在 proj_conv 后加噪声）
        # 注册 restoration hook（在指定组件处替换为 clean 激活）
        ...

    def _save_hook(self, name):
        def hook(module, input, output):
            self.clean_activations[name] = output.detach().clone()
        return hook
```

#### 注意事项：三次重复 (M1, M2, M3)

QANet 的 Model Encoder 共享权重但运行三次。因此需要区分：
- **Pass 1** (→ M1): block 0-6, 用于 Pointer 的第一个输入
- **Pass 2** (→ M2): block 0-6, 用于 Pointer 的第二个输入
- **Pass 3** (→ M3): block 0-6, 用于 Pointer 的第三个输入

总共有 **7 blocks × 4 sublayers × 3 passes = 84** 个恢复点。

### 1.3 评估样本

- 从 dev set 中随机抽取 **200-500 个样本**
- 对每个样本计算 IE，然后取平均得到 AIE (Average Indirect Effect)
- 报告 95% 置信区间

### 1.4 可视化输出

1. **因果热力图** (主图)
   - X 轴: 层位置 (Pass1-Block0 到 Pass3-Block6, 共 21 列)
   - Y 轴: 子组件类型 (Conv0, Conv1, Self-Attn, FFN, 共 4 行)
   - 颜色: AIE 值（使用 viridis 或 RdBu 色图）

2. **组件级别汇总条形图**
   - 将所有层的 AIE 按组件类型汇总
   - Conv (所有层 conv 的 AIE 之和) vs Self-Attention vs FFN

3. **按 Pass 分解的折线图**
   - 3 条线 (Pass 1/2/3)，X 轴为 Block 0-6
   - 分别画 Conv / Self-Attn / FFN 的 AIE 变化趋势

### 1.5 预期分析维度

- 哪个组件类型总体上因果效应最强？
- 因果效应是否在某些层集中？是否呈现"early site / late site"的双峰模式？
- 三次 pass 中，组件的重要性是否不同？（预测 Pass 3 更重要，因为 M3 直接影响 p2）
- 结果是否与 QANet 论文的消融实验结论一致？（论文称去掉 Conv → F1 降 2.7，去掉 Attn → F1 降 1.3）

---

## 实验二：Context vs. Question 双流因果追踪

### 2.1 研究假设

**H2**: Context 流和 Question 流在 QANet 中承担不同的因果角色。Context 的低层局部编码对答案定位更关键，而 Question 的全局语义编码对答案理解更关键。

### 2.2 实验设计

#### 腐蚀方案对比

| 条件 | 腐蚀目标 | 恢复目标 |
|------|---------|---------|
| A | Context embeddings | 逐组件恢复 Embedding Encoder + Model Encoder |
| B | Question embeddings | 逐组件恢复 Embedding Encoder + Model Encoder |
| C | Context + Question 同时腐蚀 | 逐组件恢复 |

#### 腐蚀实现

```python
# 条件 A：仅腐蚀 Context
C_corrupt = C + torch.randn_like(C) * noise_std
Q_clean = Q  # 保持不变

# 条件 B：仅腐蚀 Question
C_clean = C  # 保持不变
Q_corrupt = Q + torch.randn_like(Q) * noise_std

# 条件 C：同时腐蚀
C_corrupt = C + torch.randn_like(C) * noise_std
Q_corrupt = Q + torch.randn_like(Q) * noise_std
```

#### 恢复范围

本实验重点关注 **Embedding Encoder** 和 **CQ Attention** 层的因果效应：

- **Embedding Encoder** 中有独立的 Context 和 Question 处理路径
  - 恢复 Embedding Encoder 中 Context 通路的各子组件
  - 恢复 Embedding Encoder 中 Question 通路的各子组件
- **CQ Attention 层** 作为两流融合的关键节点
  - 恢复 CQ Attention 的输出

### 2.3 分析重点

- **信息融合时机**: CQ Attention 恢复后的 IE 有多大？这表明双流信息融合对答案预测的重要性
- **单流充分性**: 仅恢复 Context 或 Question 通路的 Embedding Encoder 能恢复多少 TE？
- **交互效应**: 恢复 CQ Attention 输出是否比分别恢复两流的效果更显著？（非线性交互证据）

### 2.4 可视化

1. **双流因果对比图**
   - 并排热力图：左=腐蚀 Context 的 AIE, 右=腐蚀 Question 的 AIE
   
2. **信息融合瀑布图**
   - X 轴: 模型阶段 (Embedding → Emb Encoder → CQ Attn → Model Encoder blocks)
   - Y 轴: 累积 AIE
   - 标注 CQ Attention 处的"跳跃"有多大

---

## 实验三：Answer-Token 位置特异性分析

### 3.1 研究假设

**H3**: QANet 中不同组件对 answer span 边界 (start vs. end) 的因果贡献不同。Conv 对精确定位 span 边界（start token）更重要，Self-Attention 对确定 span 结束位置（end token，需要更远的上下文理解）更重要。

### 3.2 实验设计

#### 核心思路

分别追踪 **p1 (start probability)** 和 **p2 (end probability)** 的因果效应：

$$\text{IE}_{p1}(c, l) = P_{\text{restore}}(y_1) - P_{\text{corrupt}}(y_1)$$
$$\text{IE}_{p2}(c, l) = P_{\text{restore}}(y_2) - P_{\text{corrupt}}(y_2)$$

#### 补充分析：按 answer 长度分层

将样本按 answer span 长度分组：
- **短答案** (1-2 tokens)
- **中等答案** (3-5 tokens)
- **长答案** (6+ tokens)

**假设**: 对于长答案，Self-Attention 的因果效应应显著增强（需要更远的上下文来确定 end position）。

### 3.3 评估指标

对每个 (component, layer, pass) 组合，报告：
- `AIE_p1`: 对 start position 预测的平均间接效应
- `AIE_p2`: 对 end position 预测的平均间接效应
- `AIE_ratio = AIE_p1 / AIE_p2`: 偏向 start 还是 end 的比率

### 3.4 可视化

1. **双通道热力图**
   - 上图: AIE for p1 (start)
   - 下图: AIE for p2 (end)
   - 差异图: AIE_p1 - AIE_p2 (红色 = 偏向 start, 蓝色 = 偏向 end)

2. **按 answer 长度分层的折线图**
   - 3 组 (短/中/长) × 3 组件类型
   - 展示不同答案长度下组件重要性的变化

---

## 实验四：Stochastic Depth 与因果重要性的关系

### 4.1 研究假设

**H4**: QANet 中使用的 Stochastic Depth (Layer Dropout) 正则化方法的生存概率分布，与通过因果追踪发现的组件重要性分布存在负相关。即：因果效应越高的层，越不应该被 drop。

### 4.2 实验设计

#### Step 1: 获取因果重要性分布

使用实验一的结果，得到每个 sublayer 的 AIE 值。

#### Step 2: 对比现有 Stochastic Depth 分布

QANet 的 stochastic depth 生存概率为：
$$p_l = 1 - \frac{l}{L}(1 - p_L), \quad p_L = 0.9$$

即线性递减：浅层生存率高，深层生存率低。

#### Step 3: 设计改进的 Stochastic Depth 策略

基于因果追踪结果，提出 **Causal-Aware Stochastic Depth**：
- 因果效应高的层获得更高的生存概率
- 因果效应低的层获得更低的生存概率

$$p_l^{\text{causal}} = 1 - \alpha \cdot (1 - \text{norm}(\text{AIE}_l))$$

其中 $\alpha$ 控制最大 drop 概率，$\text{norm}(\text{AIE}_l)$ 是归一化后的因果重要性。

#### Step 4: 训练对比实验

| 配置 | Stochastic Depth 策略 | 训练步数 |
|------|---------------------|---------|
| Baseline | 线性递减（论文原版） | 相同 |
| Causal-Aware | 基于 AIE 的非均匀分布 | 相同 |
| Inverse-Causal | 反转 AIE 分布（消融对照） | 相同 |

### 4.3 评估指标

- Dev set F1 和 EM
- 训练收敛曲线 (loss vs. steps)
- 关注 causal-aware 策略是否优于线性策略

### 4.4 分析维度

- 因果重要性与最优 drop 概率之间的相关性
- Causal-aware stochastic depth 是否提升性能？
- 这种提升是否具有统计显著性？（多次训练取平均和标准差）

---

## 实验五：M1/M2/M3 三次 Pass 的角色分化

### 5.1 研究假设

**H5**: QANet Model Encoder 的三次重复执行 (M1, M2, M3) 虽然共享权重，但在信息处理中承担不同角色。M1 负责基础语义编码，M2 负责答案候选精炼，M3 负责最终决策。

### 5.2 背景

QANet 的 Pointer 输出层使用：
```python
p1 = W1 · [M1; M2]  # start position
p2 = W2 · [M1; M3]  # end position
```

- M1 同时参与 p1 和 p2 → 可能编码答案的通用表征
- M2 仅参与 p1 → 可能专门精炼 start position
- M3 仅参与 p2 → 可能专门精炼 end position

### 5.2 实验设计

#### 方法 A: Pass-Level 因果追踪

对三次 pass 分别计算总 AIE：
$$\text{AIE}_{\text{pass}_k} = \sum_{b=0}^{6} \sum_{c \in \{conv, attn, ffn\}} \text{AIE}(k, b, c)$$

#### 方法 B: 单 Pass 恢复实验

- 仅恢复 Pass 1 的所有组件 → 测量 span 预测恢复程度
- 仅恢复 Pass 2 → ...
- 仅恢复 Pass 3 → ...

#### 方法 C: Cross-Pass 分析

研究 Pass 间的信息传递：
- 腐蚀后仅恢复 Pass 1 → 观察 Pass 2、3 是否也部分恢复（信息传播）
- 腐蚀后仅恢复 Pass 3 → 观察对 p1 (使用 M1, M2) 的影响（应该无直接效果）

### 5.3 可视化

1. **三 Pass 因果效应对比条形图**
   - 3 组 bar (Pass 1/2/3)
   - 进一步按组件类型 (Conv/Attn/FFN) 分色

2. **Pass 间信息传播矩阵**
   - 3×2 矩阵: 恢复 Pass i → 对 p1, p2 的 IE
   - 热力图形式

---

## 统一实验配置

### 基础模型

- 使用 Stage 1 & 2 修复后、完整训练的 **best checkpoint**
- 模型必须处于 `eval()` 模式（禁用 dropout 和 layer dropout）
- 所有实验使用 `torch.no_grad()`

### 噪声参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 噪声类型 | 高斯噪声 $\epsilon \sim \mathcal{N}(0, \nu^2)$ | ROME 论文 |
| 噪声强度 $\nu$ | $3 \times \text{std}(\text{embeddings})$ | ROME 论文 Section 2.1 |
| 噪声注入位置 | `proj_conv` 输出之后 | 对应 QANet 的 embedding 层输出 |

### 评估样本

| 参数 | 值 |
|------|-----|
| 样本来源 | SQuAD v1.1 dev set |
| 样本数量 | 200–500（视 GPU 内存和计算时间） |
| 选取方式 | 随机抽样 + 按 answer 长度分层 |
| 每个样本重复 | 3 次（不同噪声采样），取均值 |

### 统计显著性

- 所有 AIE 报告 **均值 ± 95% 置信区间**
- 实验四的训练对比使用 **3 个不同种子**，报告均值 ± 标准差
- 使用 Welch's t-test 检验组间差异

---

## 实验优先级与工作量估计

| 优先级 | 实验 | 估计工作量 | 难度 |
|--------|------|-----------|------|
| **P0 (必做)** | 实验一：子组件因果追踪 | 实现 hook 系统 + 运行 + 可视化 ≈ 2-3 天 | ★★★ |
| **P0 (必做)** | 实验三：Start vs. End 分析 | 与实验一共享代码，额外分析 ≈ 0.5 天 | ★★ |
| **P1 (推荐)** | 实验五：M1/M2/M3 角色分化 | 扩展实验一的 hook 到三次 pass ≈ 1 天 | ★★ |
| **P2 (加分)** | 实验二：双流分析 | 需修改 corruption 目标 ≈ 1 天 | ★★★ |
| **P3 (加分)** | 实验四：Causal-Aware Stochastic Depth | 需重新训练模型 ≈ 2-3 天 | ★★★★ |

---

## 代码组织建议

```
experiments/
├── causal_tracing/
│   ├── tracer.py           # CausalTracer 核心类
│   ├── hooks.py            # Hook 注册与管理
│   ├── corruption.py       # 噪声注入策略
│   ├── metrics.py          # IE / AIE 计算
│   ├── run_exp1.py         # 实验一入口
│   ├── run_exp2.py         # 实验二入口
│   ├── run_exp3.py         # 实验三入口
│   ├── run_exp4.py         # 实验四入口
│   ├── run_exp5.py         # 实验五入口
│   └── visualize.py        # 热力图 / 条形图 / 折线图
├── causal_tracing_plan.md  # 本文档
└── results/                # 实验结果输出
    ├── exp1/
    ├── exp2/
    └── ...
```
