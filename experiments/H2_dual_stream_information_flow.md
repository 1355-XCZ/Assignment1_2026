# H2: Context vs Question 双流信息流与 CQ Attention 融合机制

## 1. 研究问题

> QANet 的双流架构中，Context 编码和 Question 编码对答案预测的因果贡献如何？CQ Attention 作为两流的唯一融合点，其因果责任有多大？信息处理的关键阶段是融合前的独立编码还是融合后的联合推理？

## 2. 研究假设

**H2**: Context 流和 Question 流在 QANet 中承担不同的因果角色，且 CQ Attention 层作为信息瓶颈承担了超比例的因果责任。

- **预测 1**: 腐蚀 Question 对答案预测的破坏性（Total Effect）大于腐蚀 Context。
  - 理由: 没有问题语义，模型根本不知道要找什么；而即使 context 有噪声，问题仍可引导模型在大致区域搜索。
- **预测 2**: 恢复 CQ Attention 输出能恢复大部分 Total Effect（> 70%），说明 CQ Attention 是关键的信息瓶颈。
  - 理由: CQ Attention 是唯一将 Context 和 Question 信息交叉的层，其输出质量决定了后续 Model Encoder 的上限。
- **预测 3**: 在 Embedding Encoder 内部，Question 通路中 Self-Attention 的因果效应高于 Conv 的因果效应（因为问题理解更依赖全局语义），而 Context 通路中 Conv 的因果效应更高（因为答案定位更依赖局部 pattern）。

## 3. 背景：QANet 的双流结构

```
Context tokens                    Question tokens
     │                                 │
  word_emb + char_emb              word_emb + char_emb
     │                                 │
  Embedding(char_conv+highway)     Embedding(char_conv+highway)
     │                                 │
  proj_conv (500→d_model)          proj_conv (500→d_model)
     │                                 │
  ┌──────────────────────────────────────────────┐
  │  Embedding Encoder (共享权重)                  │
  │  1 block: 4 conv + self_attn + ffn           │
  │  Ce = emb_enc(C, cmask)                      │
  │  Qe = emb_enc(Q, qmask)                     │
  └──────────────────────────────────────────────┘
     │                                 │
     └──────────┬──────────────────────┘
                │
         CQ Attention (唯一融合点)
         输出: [C, A, C⊙A, C⊙B] → [B, 4*d_model, Lc]
                │
         cq_resizer (4*d_model → d_model)
                │
         Model Encoder × 3 passes → M1, M2, M3
                │
           Pointer → p1, p2
```

关键观察：**CQ Attention 之前，两流完全独立**。CQ Attention 之后，只剩 Context 维度的表征。

## 4. 实验设计

### 4.1 实验矩阵

本实验包含 **三组腐蚀条件** × **多个恢复目标**：

#### 腐蚀条件

| 条件 ID | 腐蚀对象 | 实现方式 | 目的 |
|---------|---------|---------|------|
| **CORRUPT-C** | Context embedding | 在 `proj_conv(C)` 后加噪声 | 测量 Context 编码的因果贡献 |
| **CORRUPT-Q** | Question embedding | 在 `proj_conv(Q)` 后加噪声 | 测量 Question 编码的因果贡献 |
| **CORRUPT-CQ** | 两者同时 | 两者都加噪声 | 测量总破坏 + 评估各恢复点的绝对效应 |

噪声标准差统一为 `3 × std(target)`，其中 target 是被腐蚀的张量。

#### 恢复目标

| 恢复点 ID | 位置 | 恢复内容 | 维度 |
|-----------|------|---------|------|
| **EMB-ENC-C** | Embedding Encoder Context 输出 | Ce (clean) | [B, d_model, Lc] |
| **EMB-ENC-Q** | Embedding Encoder Question 输出 | Qe (clean) | [B, d_model, Lq] |
| **EMB-ENC-C-conv[i]** | Emb Encoder 内 Context 通路的第 i 个 conv | conv output (clean) | [B, d_model, Lc] |
| **EMB-ENC-C-attn** | Emb Encoder 内 Context 通路的 self-attn | attn output (clean) | [B, d_model, Lc] |
| **EMB-ENC-C-ffn** | Emb Encoder 内 Context 通路的 FFN | ffn output (clean) | [B, d_model, Lc] |
| **EMB-ENC-Q-conv[i]** | Emb Encoder 内 Question 通路的子组件 | 同上，对 Q 通路 | [B, d_model, Lq] |
| **EMB-ENC-Q-attn** | ... | ... | ... |
| **EMB-ENC-Q-ffn** | ... | ... | ... |
| **CQ-ATT** | CQ Attention 输出 | cq_att(Ce, Qe, ...) 的输出 (clean) | [B, 4*d_model, Lc] |
| **CQ-RESIZED** | cq_resizer 输出 | cq_resizer(X) (clean) | [B, d_model, Lc] |

### 4.2 实现细节

#### 腐蚀 Context 的实现

```python
def forward_corrupt_context(model, Cwid, Ccid, Qwid, Qcid, noise_std_scale=3.0):
    cmask = (Cwid == 0)
    qmask = (Qwid == 0)
    
    Cw, Cc = model.word_emb(Cwid), model.char_emb(Ccid)
    Qw, Qc = model.word_emb(Qwid), model.char_emb(Qcid)
    
    C, Q = model.emb(Cc, Cw), model.emb(Qc, Qw)
    C = model.proj_conv(C)
    Q = model.proj_conv(Q)
    
    # >>> 腐蚀 Context <<<
    noise_std = noise_std_scale * C.std().item()
    C = C + torch.randn_like(C) * noise_std
    # Q 保持 clean
    
    # 继续正常执行...
    emb_total = model.emb_enc.conv_num + 2
    Ce = model.emb_enc(C, cmask, l=1, total_layers=emb_total)
    Qe = model.emb_enc(Q, qmask, l=1, total_layers=emb_total)
    
    X = model.cq_att(Ce, Qe, cmask, qmask)
    # ... (后续正常)
```

#### 恢复 CQ Attention 输出的实现

```python
def forward_corrupt_cq_restore_cq(model, Cwid, Ccid, Qwid, Qcid,
                                    clean_cq_output, noise_std_scale=3.0):
    # ... (前面与 corrupt 版本相同) ...
    
    # CQ Attention 正常计算（但输入被腐蚀了，所以结果不好）
    X_corrupt = model.cq_att(Ce_corrupt, Qe_corrupt, cmask, qmask)
    
    # >>> 恢复 CQ Attention 输出 <<<
    X = clean_cq_output  # 替换为 clean 版本
    
    M1 = model.cq_resizer(X)
    # ... (后续正常)
```

#### Embedding Encoder 内部恢复

由于 Embedding Encoder 分别处理 C 和 Q（共享权重，但独立执行），恢复子组件需要用与 H1 相同的 `encoder_block_forward_traced` 方法，但分别对 Context 和 Question 通路执行。

```python
# 恢复 Embedding Encoder 中 Context 通路的 self_attn
Ce = encoder_block_forward_traced(
    model.emb_enc, C_corrupt, cmask, l=1, total_layers=emb_total,
    clean_acts=clean_emb_enc_C_acts,  # 预先收集的 clean 激活
    restore_target="self_attn"
)
# Question 通路正常执行（可能也被腐蚀了，取决于条件）
Qe = model.emb_enc(Q, qmask, l=1, total_layers=emb_total)
```

### 4.3 实验运行矩阵

完整的实验矩阵（每个 cell 是一次 forward pass 系列）：

| 腐蚀条件 | 恢复目标 | 产出 |
|---------|---------|------|
| CORRUPT-C | 无 | TE_C (Context 腐蚀的总效应) |
| CORRUPT-C | EMB-ENC-C (整体) | IE: 恢复整个 Context 编码的效果 |
| CORRUPT-C | EMB-ENC-C-conv[0..3], attn, ffn (逐组件) | IE: Embedding Encoder 内各组件对 C 编码的贡献 |
| CORRUPT-C | CQ-ATT | IE: CQ Attention 在 C 被腐蚀时能恢复多少 |
| CORRUPT-Q | 无 | TE_Q (Question 腐蚀的总效应) |
| CORRUPT-Q | EMB-ENC-Q (整体) | IE: 恢复整个 Question 编码的效果 |
| CORRUPT-Q | EMB-ENC-Q-conv[0..3], attn, ffn (逐组件) | IE: Embedding Encoder 内各组件对 Q 编码的贡献 |
| CORRUPT-Q | CQ-ATT | IE: CQ Attention 在 Q 被腐蚀时能恢复多少 |
| CORRUPT-CQ | 无 | TE_CQ (双流同时腐蚀的总效应) |
| CORRUPT-CQ | EMB-ENC-C (整体) | IE: 仅恢复 C 编码能挽回多少 |
| CORRUPT-CQ | EMB-ENC-Q (整体) | IE: 仅恢复 Q 编码能挽回多少 |
| CORRUPT-CQ | EMB-ENC-C + EMB-ENC-Q (同时) | IE: 同时恢复两流能挽回多少 |
| CORRUPT-CQ | CQ-ATT | IE: 恢复融合输出能挽回多少 |
| CORRUPT-CQ | CQ-RESIZED | IE: 恢复 resize 后的输出能挽回多少（应≈CQ-ATT） |

## 5. 控制变量

| 变量 | 设定 |
|------|------|
| 模型 | 与 H1 相同的 checkpoint |
| 模型模式 | `eval()` |
| 噪声强度 | `3 × std(target)` (Context 和 Question 分别计算各自的 std) |
| 随机种子 | 固定 seed=42 |
| 评估样本 | 与 H1 相同的 300 个样本（结果可直接对比） |
| 噪声重复 | 每样本 3 次 |

## 6. 评估指标

### 6.1 主指标

与 H1 相同的 span 概率指标：

$$P_{\text{span}} = \text{softmax}(p_1)[y_1] \cdot \text{softmax}(p_2)[y_2]$$

$$\text{IE}(\text{condition}, \text{restore}) = P_{\text{restored}} - P_{\text{corrupt}}$$

$$\text{NIE}(\text{condition}, \text{restore}) = \text{IE} / \text{TE}$$

### 6.2 补充指标

**CQ Attention 恢复比率** (CQ Recovery Ratio):

$$R_{CQ} = \frac{\text{IE}(\text{CORRUPT-CQ}, \text{CQ-ATT})}{\text{TE}_{CQ}}$$

如果 $R_{CQ}$ 接近 1.0，说明 CQ Attention 输出几乎携带了所有必要信息。

**双流贡献比** (Stream Contribution Ratio):

$$\text{SCR} = \frac{\text{TE}_C}{\text{TE}_Q}$$

如果 SCR < 1，说明腐蚀 Q 的破坏性更大（Question 更关键）。

**加性检验** (Additivity Test):

$$\text{IE}(\text{CORRUPT-CQ}, \text{restore-C}) + \text{IE}(\text{CORRUPT-CQ}, \text{restore-Q}) \stackrel{?}{\approx} \text{IE}(\text{CORRUPT-CQ}, \text{restore-C+Q})$$

如果左边 < 右边，说明两流之间存在超加性交互（信息整合的非线性增益）。
如果左边 ≈ 右边，说明两流信息近似独立。

## 7. 计算量估算

- CORRUPT-C 系列: 1 (TE) + 1 (整体恢复) + 6 (Emb Enc 子组件) + 2 (CQ-ATT, CQ-RESIZED) = 10
- CORRUPT-Q 系列: 同上 = 10
- CORRUPT-CQ 系列: 1 + 2 + 1 + 1 + 1 = 6
- Clean run: 1
- 每样本总计: (10 + 10 + 6 + 1) × 3 (噪声重复) = **81 次 forward pass**
- 300 样本: 300 × 81 = **24,300 次 forward pass**
- 约 2-4 分钟 (GPU)

## 8. 可视化方案

### 8.1 主图：三种腐蚀条件下的 Total Effect 对比

```
条形图:
  TE_C  |████████      |
  TE_Q  |██████████████|
  TE_CQ |███████████████|
```

直观展示 Context vs Question 腐蚀的破坏性差异。

### 8.2 Embedding Encoder 子组件因果效应对比

并排条形图，分两组（Context 通路 / Question 通路），每组 6 个 bar（conv_0, conv_1, conv_2, conv_3, attn, ffn）：

```
CORRUPT-C:                    CORRUPT-Q:
  Conv0-C |██  |               Conv0-Q |███ |
  Conv1-C |█   |               Conv1-Q |██  |
  Conv2-C |███ |               Conv2-Q |██  |
  Conv3-C |████|               Conv3-Q |█   |
  Attn-C  |██  |               Attn-Q  |████|
  FFN-C   |█   |               FFN-Q   |███ |
```

检验预测 3：Context 通路是否 Conv 主导，Question 通路是否 Attn 主导。

### 8.3 信息流瀑布图

```
模型阶段:
  Input → Embedding → Emb Encoder → CQ Attention → CQ Resizer → Model Encoder → Output
  
累积 NIE:
  0% ──── 15% ────── 35% ────────── 85% ──────── 87% ────────── 100%
                                     ↑
                              CQ Attention 贡献了 50%
```

X 轴为模型阶段，Y 轴为累积 NIE，展示信息在流水线中的"增值"过程。

### 8.4 加性检验散点图

每个样本一个点：
- X 轴: IE(restore-C) + IE(restore-Q)
- Y 轴: IE(restore-C+Q)
- 对角线 = 完美加性
- 点在对角线上方 = 超加性交互

## 9. 预期结果与分析框架

### 预测 1 验证: TE_C vs TE_Q

- 预期 TE_Q > TE_C（Question 腐蚀更致命）
- 如果 TE_C > TE_Q → 出乎意料，需要讨论：可能因为 context 长度远大于 question（400 vs 50），更多的噪声维度导致更大的破坏
- 需要控制：可以尝试按噪声总能量归一化后再比较

### 预测 2 验证: CQ Recovery Ratio

- 预期 $R_{CQ} > 0.7$
- 如果 $R_{CQ}$ 很低（< 0.3）→ 说明 CQ Attention 不是瓶颈，Model Encoder 也在进行关键的信息处理
- 这与 H1 的结果形成交叉验证

### 预测 3 验证: 双流内部组件差异

- 预期 Context 通路中 Conv AIE > Attn AIE
- 预期 Question 通路中 Attn AIE > Conv AIE
- 如果两条通路的组件分布相同 → 共享权重的 Embedding Encoder 没有根据输入类型调整行为

### 加性检验讨论

- 超加性 → CQ Attention 在两流信息交叉时产生了非线性增益，两流信息互补
- 亚加性 → 两流存在冗余信息
- 近似加性 → 两流近似独立处理

## 10. 与 H1 的关系

- **H1 覆盖 Model Encoder (CQ Attention 之后)**: 研究 Conv/Attn/FFN 在联合表征中的角色
- **H2 覆盖 Embedding Encoder + CQ Attention (Model Encoder 之前)**: 研究双流独立编码和融合的角色
- 两者共同构成 QANet **全流水线**的因果图谱
- H2 使用**不同的腐蚀目标**（Context vs Question vs Both），所以不是 H1 的简单重复

## 11. 潜在风险与应对

| 风险 | 可能性 | 应对 |
|------|--------|------|
| TE_Q 接近 1.0（Question 腐蚀后模型完全失效） | 高 | 降低噪声倍数到 1×std；或改用 partial corruption（只腐蚀部分 question tokens） |
| Embedding Encoder 子组件的 IE 差异不显著 | 中 | Embedding Encoder 只有 1 个 block，层数较少可能导致分辨率不足。可补充分析整体 vs 子组件恢复的差距 |
| CQ Attention 恢复等价于直接恢复两流编码 | 中 | 这不是坏结果——它说明 CQ Attention 是一个 faithful 的信息传递器，没有丢失或添加信息 |
| Context/Question 长度差异影响噪声效果 | 中 | 噪声强度按各自的 std 独立计算，但总噪声能量仍不同。在分析中讨论此因素 |

## 12. 参考文献

- Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 2022.
- Yu, A. W., et al. (2018). QANet: Combining Local Convolution with Global Self-Attention. ICLR 2018.
- Seo, M. J., et al. (2016). Bidirectional Attention Flow for Machine Comprehension. (CQ Attention 的来源)
- Xiong, C., et al. (2016). Dynamic Coattention Networks for Question Answering. (DCN Attention 的来源)
