# Chief Engineer Bug Report

> Run: `chief-claude4` | Generated: 2026-03-14 12:53

## Executive Summary

### Chief A (Claude Opus)
This QANet codebase is heavily seeded with bugs at every level: over 40 unique, confirmed defects spanning data loading, model architecture, optimizers, schedulers, activations, normalizations, and the training loop. Multiple Stage I bugs (Namespace init, loss.item().backward(), Conv1d unfold, PosEncoder broadcasting, embedding swaps, depthwise-separable ordering) prevent any successful training run, while numerous Stage II mechanism bugs (inverted ReLU/LeakyReLU, wrong Adam moments, broken dropout scaling, discarded attention, wrong Kaiming/Xavier formulas) would produce garbage results even if the pipeline ran. The codebase requires comprehensive fixes across virtually every module before it can function.

### Chief B (GPT-5.4 PRO)
The codebase is pervasively broken across all major components, with 40+ distinct confirmed bugs spanning data loading, model architecture, optimization, scheduling, normalization, initialization, activation functions, and the training/evaluation loops. Multiple Stage-I crash bugs prevent any execution (e.g., `loss.item().backward()`, Namespace positional arg, ckpt key mismatch, Conv1d unfold on wrong dim, embedding lookup swap, MHA permute, encoder norm indexing). Even if all crash bugs were resolved, Stage-II mechanism bugs (inverted ReLU/LeakyReLU, wrong dropout scaling, broken Adam second-moment and bias-correction, defective LR schedulers, discarded attention output) would prevent meaningful learning. Two additional bugs were missed by audit teams: a critical embedding permute error causing a channel-mismatch crash, and an Adam bias-correction exponentiation error that flips gradient direction after step 1.

---

## Statistics

| Metric | Count |
|--------|-------|
| Confirmed bugs | 74 |
| Rejected (false positives) | 1 |
| Newly discovered by chiefs | 6 |
| Stage I bugs | 33 |
| Stage II bugs | 41 |
| Must-fix | 52 |
| Should-fix | 18 |
| Nice-to-fix | 3 |
| Ignorable | 1 |

---

## Coverage Assessment

### Chief A
**Well-covered areas:** Optimizers (adam.py, sgd.py, sgd_momentum.py), schedulers, conv.py (DepthwiseSeparableConv ordering and padding), encoder.py (norms indexing, attention discard), heads.py, loss.py, train_utils.py, activations, normalizations, and preprocessing were all thoroughly examined by multiple teams. Many true bugs received 2-6 independent reports.

**Under-examined areas:** (1) Models/embedding.py — the `ch_emb.permute(0,2,1,3)` bug in Embedding.forward was completely missed; only the Highway transpose was caught. (2) Optimizers/adam.py — the bias correction formula (`beta1 * t` instead of `beta1 ** t`) was missed because BUG-023's KeyError masks it. (3) Models/encoder.py MultiHeadAttention — the mask `repeat(H,1,1)` ordering issue (which becomes critical once BUG-049's input permute is fixed) was not flagged. (4) Models/Initializations/xavier.py — the `xavier_uniform_` function has the same `fan_in * fan_out` bug as `xavier_normal_` but was not independently reported. (5) Cross-file interaction between evaluate.py and save_checkpoint was well-covered for the key mismatch, but the qa_ce_loss incompatibility with model log_softmax output was unexamined.

### Chief B
**Well-covered areas:** Optimizers (adam, sgd, sgd_momentum — key mismatches, weight decay signs, second moment all caught), LR schedulers (all three scheduler bugs caught), Conv layers (unfold dim, padding height, depthwise-separable order), encoder normalization indexing and attention-residual discard, loss function argument swap, training loop (Namespace, loss.item().backward, gradient clip ordering), activation functions, initialization formulas, and preprocessing answer-index inconsistencies.

**Under-examined areas:** (1) Embedding.forward `ch_emb.permute(0, 2, 1, 3)` — a cross-file shape-flow bug requiring tracing from qanet.py through embedding.py into conv.py's Conv2d, missed because teams focused on the Highway transpose bug in the same file and may have trusted the misleading code comment. (2) Adam bias-correction formula (`beta * t` vs `beta ** t`) — teams found the key mismatch and second-moment bugs but overlooked this subtle arithmetic error that looks plausible at first glance. (3) The CQ-attention mask swap (BUG-052) was correctly identified as a bug but misclassified as Stage-II; it actually causes a Stage-I crash because `para_limit ≠ ques_limit` makes the mask non-broadcastable. (4) Xavier uniform initialization shares the same `fan_in * fan_out` bug as xavier normal but was not explicitly called out. (5) The interaction between multiple co-occurring bugs (e.g., BUG-022 masking the embedding permute bug) was not analyzed by any team.

---

## Must Fix (52)

### BUG-001: `EvaluateTools/evaluate.py` L107

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | checkpoint |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | gpt-5.4-pro[reason:xhigh] |

**Symptom**: KeyError: 'model' when loading the checkpoint because the saved key is 'model_state'.

**Root Cause**: `ckpt["model"]` does not match the key `"model_state"` used in `save_checkpoint`.

**Fix**: Change `ckpt["model"]` to `ckpt["model_state"]`.

**Chief Reasoning**:
- *chief_a*: evaluate.py line ~107: `model.load_state_dict(ckpt['model'])`. train_utils.py save_checkpoint stores weights as `'model_state': model.state_dict()`. KeyError: 'model' is guaranteed when loading any checkpoint produced by training.
- *chief_b*: evaluate.py line ~107: `model.load_state_dict(ckpt['model'])` but save_checkpoint in train_utils.py uses `'model_state': model.state_dict()`. KeyError: 'model' at runtime. Confirmed by inspecting both files.

---

### BUG-002: `EvaluateTools/evaluate.py` L124

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | checkpoint |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: evaluate() raises a KeyError when loading a checkpoint produced by the training code.

**Root Cause**: Checkpoint loading looks for `ckpt["model"]`, but `save_checkpoint()` stores the model weights under `"model_state"`.

**Fix**: Load the state dict from the saved key name, e.g. `model.load_state_dict(ckpt["model_state"])`.

**Chief Reasoning**:
- *chief_a*: Exact duplicate of BUG-004. Same root cause (ckpt['model'] vs ckpt['model_state']), same file, same line.
- *chief_b*: Exact duplicate of BUG-004: same root cause (ckpt key 'model' vs 'model_state'), same file, same fix.

---

### BUG-003: `Losses/loss.py` L6

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | loss_function |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | gemini-3.1-pro-preview[think:high] |

**Symptom**: F.nll_loss receives target tensor (y1) as input and log-prob tensor (p1) as target, causing a shape/dtype mismatch crash.

**Root Cause**: Arguments to the first F.nll_loss call are swapped: F.nll_loss(y1, p1) instead of F.nll_loss(p1, y1).

**Fix**: Change `F.nll_loss(y1, p1)` to `F.nll_loss(p1, y1)` so input (log-probs) is the first argument and target (indices) is the second.

**Chief Reasoning**:
- *chief_a*: Losses/loss.py line 6: `F.nll_loss(y1, p1)` — F.nll_loss signature is (input, target). y1 is a LongTensor of target indices, p1 is a float log-prob tensor. Passing y1 as input triggers a dtype/shape crash. Second call `F.nll_loss(p2, y2)` is correct. Only the first call is swapped.
- *chief_b*: Losses/loss.py line 6: `F.nll_loss(y1, p1)` passes target y1 (LongTensor of indices) as input and log-probs p1 as target. F.nll_loss signature is (input, target). This crashes with dtype/shape mismatch. The second call `F.nll_loss(p2, y2)` is correct. Fix: swap to `F.nll_loss(p1, y1)`.

---

### BUG-004: `Models/Normalizations/layernorm.py` L36

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | normalization |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] |

**Symptom**: Crash during normalization due to shape mismatch when broadcasting x - mean.

**Root Cause**: keepdim=False removes the reduced dimensions, causing broadcasting to fail.

**Fix**: Set keepdim=True in mean and var calculations.

**Chief Reasoning**:
- *chief_a*: Models/Normalizations/layernorm.py line 36-37: `mean = x.mean(dim=dims, keepdim=False)`. For x=[B,C,L] and dims=(-2,-1), mean becomes [B] instead of [B,1,1]. Then `x - mean` tries to broadcast [B,C,L] - [B], which fails (ambiguous broadcast). Setting keepdim=True yields [B,1,1] which broadcasts correctly.
- *chief_b*: Models/Normalizations/layernorm.py lines 36-37: `mean = x.mean(dim=dims, keepdim=False)` and `var = x.var(dim=dims, keepdim=False)`. With keepdim=False, reduced dims are removed. Subsequent `x - mean` cannot broadcast correctly (e.g., x is [B,C,L] but mean is [B] after reducing over (-2,-1)). Fix: keepdim=True.

---

### BUG-005: `Models/attention.py` L31

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | attention |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] |

**Symptom**: RuntimeError: size mismatch for batched matrix multiplication (bmm).

**Root Cause**: The arguments to torch.bmm are in the wrong order. Q has shape [B, Lq, C] and S1 has shape [B, Lc, Lq], so Q x S1 is an invalid matrix multiplication.

**Fix**: Change the order of arguments to torch.bmm(S1, Q).

**Chief Reasoning**:
- *chief_a*: Models/attention.py line 31: `A = torch.bmm(Q, S1)`. Q=[B,Lq,C], S1=[B,Lc,Lq]. bmm requires inner dims to match: Q's last dim C ≠ S1's second dim Lc. Correct: `A = torch.bmm(S1, Q)` giving [B,Lc,Lq]@[B,Lq,C]=[B,Lc,C].
- *chief_b*: Models/attention.py line 31: `A = torch.bmm(Q, S1)` where Q is [B, Lq, C] and S1 is [B, Lc, Lq]. Inner dims C vs Lc don't match. Correct: `A = torch.bmm(S1, Q)` giving [B, Lc, C].

---

### BUG-006: `Models/conv.py` L44

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | shape_mismatch |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: Conv1d unfold operates on channel dimension (dim=1) instead of spatial dimension (dim=2), producing wrong output shape and incorrect convolution

**Root Cause**: `x.unfold(1, self.kernel_size, 1)` unfolds along C_in instead of L; should be `x.unfold(2, self.kernel_size, 1)`

**Fix**: Change `x.unfold(1, ...)` to `x.unfold(2, ...)`

**Chief Reasoning**:
- *chief_a*: Models/conv.py line 44: `x.unfold(1, self.kernel_size, 1)` unfolds along dim 1 (C_in) instead of dim 2 (L). For x=[B,C_in,L], this yields [B,C_in-k+1,L,k] instead of [B,C_in,L_out,k]. Subsequent view(B,G,C_in_g,L_out,k) crashes because C_in-k+1 ≠ C_in in general.
- *chief_b*: Models/conv.py line 44: `x.unfold(1, self.kernel_size, 1)` unfolds along dim 1 (C_in) instead of dim 2 (L). For x=[B,C_in,L], this produces [B, C_in-k+1, L, k] instead of [B, C_in, L-k+1, k]. The subsequent view to [B,G,C_in_g,L_out,k] fails because element counts don't match.

---

### BUG-007: `Models/conv.py` L96

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | shape_mismatch |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: 2D convolutions with padding crash during width padding because the width-pad tensor height does not match the already height-padded input.

**Root Cause**: `pad_w` is allocated with the original height `H` instead of the current padded height `x.size(2)`.

**Fix**: Allocate `pad_w` with the updated height after height padding, e.g. `x.new_zeros(B, C_in, x.size(2), p)`.

**Chief Reasoning**:
- *chief_a*: Models/conv.py line 116: `pad_w = x.new_zeros(B, C_in, H, p)` uses original H after x was already height-padded to H+2p. torch.cat along dim=3 requires matching dim=2 sizes: x has H+2p but pad_w has H. Fix: use x.size(2) instead of H.
- *chief_b*: Models/conv.py line ~116: after height padding `x = torch.cat([pad_h, x, pad_h], dim=2)` makes x [B,C_in,H+2p,W], but `pad_w = x.new_zeros(B, C_in, H, p)` still uses original H. torch.cat along dim 3 fails because dim 2 is H+2p vs H.

---

### BUG-008: `Models/conv.py` L116

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | shape_mismatch |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] |

**Symptom**: Conv2d crashes with a shape-mismatch error during width-padding concatenation because pad_w height doesn't match the already height-padded tensor.

**Root Cause**: pad_w is created with the original height H instead of the post-padding height H+2p (after x was already padded along dim 2).

**Fix**: Change x.new_zeros(B, C_in, H, p) to x.new_zeros(B, C_in, H + 2*p, p) or use x.size(2).

**Chief Reasoning**:
- *chief_a*: Exact duplicate of BUG-010. Same root cause: Conv2d pad_w uses original H instead of post-padding x.size(2).
- *chief_b*: Exact duplicate of BUG-010: same root cause (pad_w height mismatch after height padding), same file, same fix.

---

### BUG-009: `Models/conv.py` L136

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | other |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Layers where `in_ch != out_ch` fail with channel/group mismatches, and even when shapes happen to match the separable convolution computes the wrong operation.

**Root Cause**: `DepthwiseSeparableConv.forward` applies pointwise convolution before depthwise convolution, reversing the intended order.

**Fix**: Apply depthwise first and pointwise second: `return self.pointwise_conv(self.depthwise_conv(x))`.

**Chief Reasoning**:
- *chief_a*: Models/conv.py line ~152: `return self.depthwise_conv(self.pointwise_conv(x))` applies pointwise(in→out) first, then depthwise(in→in, groups=in). When in_ch≠out_ch (e.g., context_conv: 364→96), the depthwise receives 96 channels but has groups=364 → crash. Even when in_ch==out_ch, the computation is semantically wrong (standard DSConv is depthwise-then-pointwise).
- *chief_b*: Models/conv.py line ~152: `return self.depthwise_conv(self.pointwise_conv(x))` applies pointwise first then depthwise. For context_conv where in_ch=d_word+d_char=364 and out_ch=d_model=96, pointwise changes channels to 96, then depthwise (groups=364, expecting 364 channels) receives 96 channels → crash. Fix: swap to pointwise(depthwise(x)).

---

### BUG-010: `Models/conv.py` L152

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | other |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: DepthwiseSeparableConv applies pointwise then depthwise; when in_channels != out_channels the depthwise conv receives a tensor with the wrong number of channels and crashes.

**Root Cause**: Forward calls self.depthwise_conv(self.pointwise_conv(x)) — the order is reversed; depthwise should precede pointwise.

**Fix**: Change to return self.pointwise_conv(self.depthwise_conv(x)).

**Chief Reasoning**:
- *chief_a*: Exact duplicate of BUG-012. Same reversed depthwise/pointwise order in DepthwiseSeparableConv.forward.
- *chief_b*: Duplicate of BUG-012. Same root cause, same code location, same fix.

---

### BUG-011: `Models/conv.py` L174

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | other |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | gpt-5.4-pro[reason:xhigh] |

**Symptom**: DepthwiseSeparableConv applies pointwise convolution before depthwise, reversing the intended order. When in_channels != out_channels the depthwise grouped conv receives the wrong number of channels and crashes.

**Root Cause**: return self.depthwise_conv(self.pointwise_conv(x)) applies pointwise first; correct order is depthwise then pointwise.

**Fix**: Change to return self.pointwise_conv(self.depthwise_conv(x)).

**Chief Reasoning**:
- *chief_a*: Exact duplicate of BUG-012.
- *chief_b*: Duplicate of BUG-012. Same root cause, same code location, same fix.

---

### BUG-012: `Models/embedding.py` L17

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | embedding |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: The highway layers receive batch size as the feature dimension, causing linear-layer shape errors or cross-example mixing if sizes happen to match.

**Root Cause**: Highway.forward uses x.transpose(0, 2), which swaps batch and length instead of channel and length.

**Fix**: Transpose dimensions 1 and 2 (or permute to [B, L, C]) before the linear layers, then transpose back.

**Chief Reasoning**:
- *chief_a*: Models/embedding.py line 17: `x.transpose(0, 2)` on [B,C,L] gives [L,C,B], not [B,L,C]. Linear layers then operate on the B dimension as features. If batch_size ≠ d_word+d_char, it crashes; if they happen to match, it cross-contaminates batch elements. Return `x.transpose(1,2)` also yields wrong shape [L,B,C] instead of [B,C,L]. Fix: use transpose(1,2).
- *chief_b*: Models/embedding.py Highway.forward: `x.transpose(0, 2)` on [B,C,L] produces [L,C,B], feeding batch dim (B) into linear layers expecting feature dim (C). When B≠C (almost always), linear layer raises a size mismatch. Fix: `x.transpose(1, 2)` → [B,L,C].

---

### BUG-013: `Models/embedding.py` L31

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | embedding |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Highway network linear layers receive the batch dimension as features, crashing when batch_size != d_word+d_char.

**Root Cause**: x.transpose(0, 2) swaps batch and length dims, producing [L,C,B] instead of [B,L,C]; should swap dims 1 and 2.

**Fix**: Change x.transpose(0, 2) to x.transpose(1, 2).

**Chief Reasoning**:
- *chief_a*: Exact duplicate of BUG-015. Same Highway transpose bug.
- *chief_b*: Duplicate of BUG-015. Same transpose(0,2) → transpose(1,2) fix in Highway.forward.

---

### BUG-014: `Models/embedding.py` L31

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | embedding |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | ? |
| Discovered by | claude-opus-4-6[think:adaptive] (chief) |

**Symptom**: Embedding.forward crashes with a Conv2d channel mismatch: ch_emb has char_limit (16) channels in dim 1 but Conv2d expects d_char (64) channels.

**Root Cause**: `ch_emb.permute(0, 2, 1, 3)` on [B, L, char_limit, d_char] produces [B, char_limit, L, d_char], not the intended [B, d_char, L, char_limit]. The comment claims [B, d_char, L, char_len] but the permute indices are wrong: (0,2,1,3) swaps dims 1↔2, while the correct permute (0,3,1,2) moves dim 3 (d_char) to position 1.

**Fix**: Change `ch_emb.permute(0, 2, 1, 3)` to `ch_emb.permute(0, 3, 1, 2)`.

**Why Missed by Teams**: Teams focused on the Highway.forward transpose bug (BUG-015/016) in the same file but overlooked the separate permute in Embedding.forward. The misleading inline comment ('# [B, d_char, L, char_len]') made the line look correct at a glance. Additionally, earlier crash bugs (BUG-022 swapped embeddings, BUG-017 PosEncoder) prevent execution from reaching this point.

---

### BUG-015: `Models/embedding.py` L31

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | embedding |
| Assignment | Stage I - Task 7: Embedding |
| Confidence | ? |
| Discovered by | claude-opus-4-6[think:adaptive] (chief) |

**Symptom**: Conv2d in Embedding.forward receives a tensor with char_limit (16) channels instead of d_char (64) channels, causing a shape-mismatch crash in the depthwise-separable convolution.

**Root Cause**: ch_emb.permute(0, 2, 1, 3) on input [B, L, char_limit, d_char] produces [B, char_limit, L, d_char], but Conv2d (with in_channels=d_char=64) expects [B, d_char, L, char_limit]. The permute swaps the wrong pair of dimensions.

**Fix**: Change `ch_emb.permute(0, 2, 1, 3)` to `ch_emb.permute(0, 3, 1, 2)` to produce [B, d_char, L, char_limit].

**Why Missed by Teams**: Teams focused on the Highway transpose bug (BUG-015/016) in the same file and likely trusted the misleading code comment '# [B, d_char, L, char_len]' which states the desired (but not actual) output shape. Verifying the actual permutation output requires tracing 4D tensor shapes across qanet.py → embedding.py → conv.py, a cross-file analysis that subset-viewing teams could not perform. Additionally, BUG-022 (embedding lookup swap) causes an earlier crash that masks this bug in end-to-end testing.

---

### BUG-016: `Models/encoder.py` L26

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | shape_mismatch |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: EncoderBlock construction usually fails while building positional encodings because the frequency tensor does not broadcast against the [C, L] position grid.

**Root Cause**: freqs is unsqueezed on dimension 0, producing shape [1, C] instead of [C, 1].

**Fix**: Change freqs.unsqueeze(0) to freqs.unsqueeze(1).

**Chief Reasoning**:
- *chief_a*: Models/encoder.py line 26: `freqs.unsqueeze(0)` on [d_model] gives [1,d_model] instead of [d_model,1]. Then `pos * freqs` is [d_model,length]*[1,d_model]. Broadcasting requires dim1: length vs d_model. With defaults (96≠400), this crashes. Fix: unsqueeze(1) → [d_model,1] for correct [d_model,length]*[d_model,1] broadcast.
- *chief_b*: Models/encoder.py line ~26: `freqs.unsqueeze(0)` produces [1, d_model]. pos is [d_model, length]. Multiplication [d_model, length] * [1, d_model] fails because last dims (length vs d_model) are incompatible. Fix: `freqs.unsqueeze(1)` → [d_model, 1] broadcasts correctly.

---

### BUG-017: `Models/encoder.py` L87

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | normalization |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] |

**Symptom**: IndexError: self.norms[i+1] goes out of bounds on the last conv iteration (i = conv_num-1 → index conv_num, but list has conv_num elements indexed 0..conv_num-1).

**Root Cause**: Off-by-one in norm indexing: self.norms[i + 1] should be self.norms[i].

**Fix**: Change self.norms[i + 1] to self.norms[i].

**Chief Reasoning**:
- *chief_a*: Models/encoder.py line 87: `self.norms[i + 1]` in a loop i=0..conv_num-1, but self.norms has exactly conv_num elements (indices 0..conv_num-1). The last iteration accesses index conv_num → IndexError. Also, self.norms[0] is never used.
- *chief_b*: Models/encoder.py line ~103: `self.norms[i + 1]` in the convolution loop. self.norms has conv_num elements (indices 0..conv_num-1). At the last iteration i=conv_num-1, index i+1=conv_num is out of bounds → IndexError. Also skips self.norms[0]. Fix: use self.norms[i].

---

### BUG-018: `Models/encoder.py` L103

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | normalization |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] | gemini-3.1-pro-preview[think:high] |

**Symptom**: The encoder crashes on the last convolution with an index-out-of-range error, and it also skips the first normalization slot before that.

**Root Cause**: self.norms has length conv_num, but the loop indexes it as self.norms[i + 1].

**Fix**: Index self.norms with i, or allocate conv_num + 1 norms and use them consistently.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-018. Same off-by-one norms indexing bug.
- *chief_b*: Duplicate of BUG-018. Same off-by-one norm indexing bug.

---

### BUG-019: `Models/encoder.py` L129

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | normalization |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Every encoder block raises `IndexError` on its last convolution step.

**Root Cause**: `self.norms` contains `conv_num` elements, but the loop indexes `self.norms[i + 1]`, so the final iteration accesses one past the end.

**Fix**: Index `self.norms[i]` instead, or allocate one extra normalization layer if that was the intent.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-018.
- *chief_b*: Duplicate of BUG-018. Same off-by-one norm indexing bug.

---

### BUG-020: `Models/heads.py` L18

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | shape_mismatch |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | gemini-3.1-pro-preview[think:high] |

**Symptom**: torch.cat([M1, M2], dim=0) concatenates along batch dim producing [2B, C, L] instead of [B, 2C, L], causing matmul dimension mismatch with w1 of size [2C].

**Root Cause**: Wrong concatenation dimension: `dim=0` should be `dim=1`.

**Fix**: Change `torch.cat([M1, M2], dim=0)` to `torch.cat([M1, M2], dim=1)`.

**Chief Reasoning**:
- *chief_a*: Models/heads.py line 18: `torch.cat([M1, M2], dim=0)` on [B,C,L] gives [2B,C,L]. Then `torch.matmul(self.w1, X1)` with w1=[2C] against [2B,C,L] tries to dot 2C with C (second-to-last dim) → size mismatch crash. Fix: dim=1 gives [B,2C,L], then matmul [2C]@[B,2C,L]→[B,L] correctly.
- *chief_b*: Models/heads.py line 18: `torch.cat([M1, M2], dim=0)` on [B,C,L] tensors → [2B,C,L]. Then `torch.matmul(self.w1, X1)` with w1 of shape [2C] fails: matmul of [2C] with [2B,C,L] requires last-two dims [C,L], but w1 is [2C]≠C. Fix: dim=1 → [B,2C,L], then w1·X1 works as [2C]@[2C,L]=[L] per batch.

---

### BUG-021: `Models/qanet.py` L63

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | embedding |
| Assignment | Stage I - Task 3: Model Architecture |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: Context embedding lookup can raise index errors on the char table, and even when indices are in range the context char/word tensors reach Embedding with swapped feature dimensions.

**Root Cause**: Context word ids and char ids are looked up in the wrong embedding tables: Cwid goes through self.char_emb and Ccid goes through self.word_emb.

**Fix**: Use self.word_emb(Cwid) for context words and self.char_emb(Ccid) for context chars before calling self.emb.

**Chief Reasoning**:
- *chief_a*: Models/qanet.py line 63: `Cw, Cc = self.char_emb(Cwid), self.word_emb(Ccid)`. Cwid contains word indices (up to ~50k), char_emb has vocab ~200 → IndexError. Even if sizes matched, the semantics are completely wrong (word IDs looked up in char table). Fix: Cw=self.word_emb(Cwid), Cc=self.char_emb(Ccid).
- *chief_b*: Models/qanet.py line 63: `Cw, Cc = self.char_emb(Cwid), self.word_emb(Ccid)` feeds word IDs into char_emb and char IDs into word_emb. Cwid contains word vocabulary indices that likely exceed char_emb's vocab size → IndexError. Fix: `Cw = self.word_emb(Cwid)`, `Cc = self.char_emb(Ccid)`.

---

### BUG-022: `Optimizers/adam.py` L62

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | optimizer |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] |

**Symptom**: KeyError: 'm' when trying to access the first moment buffer.

**Root Cause**: The state dictionary keys are initialized as 'exp_avg' and 'exp_avg_sq' but accessed as 'm' and 'v'.

**Fix**: Use consistent keys, e.g., 'm' and 'v', for both initialization and access.

**Chief Reasoning**:
- *chief_a*: Optimizers/adam.py: State initialized as state['exp_avg'] and state['exp_avg_sq'] at line ~62, but accessed as state['m'] and state['v'] at line ~65. KeyError: 'm' on the first optimizer step.
- *chief_b*: Optimizers/adam.py: state buffers initialized as `state['exp_avg']` and `state['exp_avg_sq']` (line ~60) but accessed as `state['m']` and `state['v']` (line ~62). KeyError: 'm' on first step. Fix: use consistent key names.

---

### BUG-023: `Optimizers/sgd_momentum.py` L43

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | optimizer |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: KeyError: 'velocity' on every step because the buffer is stored under 'vel' but accessed as 'velocity'

**Root Cause**: Initialization sets state['vel'] but the read on the next line accesses state['velocity']

**Fix**: Change state['vel'] to state['velocity'] (or vice versa, use a consistent key name)

**Chief Reasoning**:
- *chief_a*: Optimizers/sgd_momentum.py line 43-44: Velocity buffer stored as state['vel'] but accessed as state['velocity']. KeyError on first step.
- *chief_b*: Optimizers/sgd_momentum.py line 43: `state['vel'] = torch.zeros_like(p)` then line 45: `v = state['velocity']`. KeyError: 'velocity'. Fix: use consistent key name ('velocity' in both places).

---

### BUG-024: `Schedulers/cosine_scheduler.py` L25

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | lr_scheduler |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] |

**Symptom**: AttributeError: module 'math' has no attribute 'PI'.

**Root Cause**: The constant for pi in the math module is math.pi, not math.PI.

**Fix**: Change math.PI to math.pi.

**Chief Reasoning**:
- *chief_a*: Schedulers/cosine_scheduler.py line 25: `math.PI` does not exist; Python's math module uses `math.pi` (lowercase). AttributeError on first get_lr() call.
- *chief_b*: Schedulers/cosine_scheduler.py line 25: `math.PI` → AttributeError. Python's math module uses lowercase `math.pi`. Fix: change to `math.pi`.

---

### BUG-025 ✅: `TrainTools/train.py` L88

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | training_loop |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Status | ✅ Fixed |
| Discovered by | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | gemini-3.1-pro-preview[think:high] |

**Symptom**: TypeError: Namespace.__init__() takes 1 positional argument but 2 were given, because a dict is passed as a positional arg.

**Root Cause**: `argparse.Namespace({k: v ...})` passes a dict positionally; Namespace expects **kwargs.

**Fix**: Change to `argparse.Namespace(**{k: v for k, v in locals().items()})`.

**BUG Impact (if not fixed)**: Training fails immediately at startup with a `TypeError`, so the pipeline cannot reach cache checks, model initialization, forward/backward propagation, optimizer updates, checkpoint saves, or validation evaluation.

**FIX Impact (after fixed)**: The training entrypoint can construct a valid runtime namespace and proceed into data checks, model build, optimization steps, checkpointing, and evaluation as expected for Stage I functionality recovery.

**Chief Reasoning**:
- *chief_a*: TrainTools/train.py line 88: `argparse.Namespace({k: v ...})` passes a dict as a positional argument. Namespace.__init__ only accepts **kwargs. TypeError on every training invocation. Fix: `Namespace(**{k: v ...})`.
- *chief_b*: TrainTools/train.py line 88: `argparse.Namespace({k: v ...})` passes a dict as a positional argument. Namespace.__init__ only accepts **kwargs, causing TypeError. Fix: `argparse.Namespace(**{k: v for k, v in locals().items()})`.

---

### BUG-026: `TrainTools/train_utils.py` L30

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | training_loop |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | gemini-3.1-pro-preview[think:high] |

**Symptom**: AttributeError: 'float' object has no attribute 'backward' — loss.item() returns a Python float, which cannot be back-propagated.

**Root Cause**: loss.item().backward() calls .item() before .backward(), detaching the computation graph.

**Fix**: Change `loss.item().backward()` to `loss.backward()`.

**Chief Reasoning**:
- *chief_a*: TrainTools/train_utils.py line 30: `loss.item().backward()`. loss.item() returns a Python float, severing the computation graph. float has no .backward() method → AttributeError. Fix: `loss.backward()`.
- *chief_b*: TrainTools/train_utils.py line 30: `loss.item().backward()`. loss.item() returns a Python float which has no .backward() method → AttributeError. Fix: `loss.backward()`.

---

### BUG-034: `EvaluateTools/eval_utils.py` L100

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | critical |
| Category | evaluation |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: argmax over dim=0 (batch dimension) produces a tensor of shape (seq_len,) instead of (batch_size,), yielding completely wrong answer indices and shape mismatches with the batch of ids.

**Root Cause**: `torch.argmax(p1, dim=0)` and `torch.argmax(p2, dim=0)` use dim=0 (batch) instead of dim=1 (sequence length). p1/p2 have shape (batch, seq_len).

**Fix**: Change `dim=0` to `dim=1` in both `torch.argmax` calls.

**Chief Reasoning**:
- *chief_a*: EvaluateTools/eval_utils.py line 100: `torch.argmax(p1, dim=0)` on p1=[B,seq_len] reduces over batch dim, yielding [seq_len] instead of [B]. Downstream code expects per-example predictions. Fix: dim=1. Classified as Stage II because evaluation can technically run (argmax still produces a tensor), but answers are completely wrong.
- *chief_b*: EvaluateTools/eval_utils.py line ~100: `torch.argmax(p1, dim=0)` reduces over batch dim (0) instead of sequence dim (1). p1/p2 are [B, seq_len], so dim=0 produces [seq_len] instead of [B]. All predicted spans are wrong and subsequent shape operations fail. Fix: dim=1.

---

### BUG-035: `Models/Activations/leakeyReLU.py` L19

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | critical |
| Category | activation |
| Assignment | Stage II - Task 6: Activation |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] |

**Symptom**: Positive inputs are scaled by negative_slope and negative inputs pass through unscaled – the exact inverse of LeakyReLU

**Root Cause**: torch.where(x < 0, x, self.negative_slope * x) returns x when condition is True (x<0) and scaled x when False (x>=0); the two value arguments are swapped

**Fix**: Change to torch.where(x < 0, self.negative_slope * x, x)

**Chief Reasoning**:
- *chief_a*: Models/Activations/leakeyReLU.py line 19: `torch.where(x < 0, x, self.negative_slope * x)`. When x<0 (True), returns x unscaled; when x≥0 (False), returns negative_slope*x. This inverts LeakyReLU: positives are attenuated, negatives pass through. Fix: swap the two value arguments.
- *chief_b*: Models/Activations/leakeyReLU.py line 19: `torch.where(x < 0, x, self.negative_slope * x)`. torch.where(cond, val_if_true, val_if_false): when x<0 returns x (unscaled), when x≥0 returns slope*x (scaled). This is inverted — positive values get damped, negatives pass through. Fix: swap the two value arguments.

---

### BUG-036: `Models/Activations/relu.py` L12

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | critical |
| Category | activation |
| Assignment | Stage II - Task 6: Activation |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: ReLU zeros out all positive values and keeps negatives, producing the opposite of ReLU

**Root Cause**: x.clamp(max=0.0) clamps values to be at most 0; correct ReLU requires x.clamp(min=0.0)

**Fix**: Change x.clamp(max=0.0) to x.clamp(min=0.0)

**Chief Reasoning**:
- *chief_a*: Models/Activations/relu.py line 12: `x.clamp(max=0.0)` caps all values at 0, keeping negatives and zeroing positives — the exact opposite of ReLU. Fix: x.clamp(min=0.0).
- *chief_b*: Models/Activations/relu.py line 12: `x.clamp(max=0.0)` clamps all values to ≤0, zeroing positives and keeping negatives — the exact opposite of ReLU. Fix: `x.clamp(min=0.0)`.

---

### BUG-037: `Models/Normalizations/groupnorm.py` L42

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | critical |
| Category | normalization |
| Assignment | Stage II - Task 4: Regularization |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: Channels are split as (B, C//G, G, *spatial) so normalization is computed across interleaved channels rather than contiguous groups, producing wrong statistics and garbled output

**Root Cause**: Reshape order is (B, C//G, G, *spatial) instead of the correct (B, G, C//G, *spatial)

**Fix**: Change x.view(B, C // self.G, self.G, *spatial) to x.view(B, self.G, C // self.G, *spatial)

**Chief Reasoning**:
- *chief_a*: Models/Normalizations/groupnorm.py line 42: `x.view(B, C // self.G, self.G, *spatial)` puts C//G before G. The normalization dims tuple(range(2,...)) then normalizes over (G, *spatial) instead of (C//G, *spatial), mixing groups with spatial stats. Fix: x.view(B, self.G, C // self.G, *spatial).
- *chief_b*: Models/Normalizations/groupnorm.py line 42: `x.view(B, C // self.G, self.G, *spatial)` puts channels-per-group before groups. Correct is `x.view(B, self.G, C // self.G, *spatial)`. The wrong reshape interleaves channels from different groups, computing statistics over the wrong channel subsets.

---

### BUG-038: `Models/encoder.py` L117

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | critical |
| Category | attention |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: Self-attention output is completely discarded; the encoder block's attention sub-layer contributes nothing

**Root Cause**: `out = self.self_att(out, mask)` is immediately overwritten by `out = res` on the next line, destroying the attention result instead of forming a residual connection `out = out + res`

**Fix**: Change `out = res` to `out = out + res`

**Chief Reasoning**:
- *chief_a*: Models/encoder.py lines 117-118: `out = self.self_att(out, mask)` then immediately `out = res`. The attention output is discarded and replaced by the residual. The attention sublayer contributes nothing. Fix: `out = out + res` (or `out = self.drop(out) + res`).
- *chief_b*: Models/encoder.py lines 117-118: `out = self.self_att(out, mask)` immediately followed by `out = res`. The attention output is completely discarded and replaced by the pre-attention residual. The attention sublayer contributes nothing. Fix: `out = out + res` (or `out = self.drop(out) + res`).

---

### BUG-039: `Optimizers/adam.py` L68

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | critical |
| Category | optimizer |
| Assignment | Stage II - Task 1: Optimizer |
| Confidence | ? |
| Discovered by | claude-opus-4-6[think:adaptive] (chief) |

**Symptom**: After step 1, Adam bias correction becomes negative (e.g., at t=2 with beta1=0.8: 1-0.8*2=-0.6), flipping the sign of m_hat and v_hat. This reverses the update direction and causes divergence.

**Root Cause**: Bias correction uses multiplication (`1.0 - beta1 * t`) instead of exponentiation (`1.0 - beta1 ** t`). At t=1 these are identical (0.8*1 = 0.8^1), but at t≥2 they diverge dramatically. With beta1=0.8, t=2: code gives 1-1.6=-0.6 vs correct 1-0.64=0.36.

**Fix**: Change `1.0 - beta1 * t` to `1.0 - beta1 ** t` and `1.0 - beta2 * t` to `1.0 - beta2 ** t`.

**Why Missed by Teams**: Masked by BUG-023 (KeyError on state['m'] prevents execution from ever reaching the bias correction lines). Audit teams likely never traced code past the KeyError. The formula also looks superficially correct since * and ** give identical results at t=1.

---

### BUG-040: `Optimizers/adam.py` L70

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | critical |
| Category | optimizer |
| Assignment | Stage II - Task 9: Optimizer |
| Confidence | ? |
| Discovered by | claude-opus-4-6[think:adaptive] (chief) |

**Symptom**: After step 1, Adam's bias correction becomes negative (e.g., at step 2 with beta1=0.8: 1 - 0.8*2 = -0.6), flipping the sign of the corrected first moment m_hat. This causes parameter updates in the wrong direction, leading to immediate divergence.

**Root Cause**: Bias correction uses multiplication instead of exponentiation: `bias_correction1 = 1.0 - beta1 * t` and `bias_correction2 = 1.0 - beta2 * t`. The correct Adam formula is `1.0 - beta1 ** t` and `1.0 - beta2 ** t`.

**Fix**: Change `1.0 - beta1 * t` to `1.0 - beta1 ** t` and `1.0 - beta2 * t` to `1.0 - beta2 ** t`.

**Why Missed by Teams**: Teams found three other Adam bugs (BUG-023 key mismatch, BUG-053 weight decay sign, BUG-054 missing grad squaring) but overlooked this subtle arithmetic error. At step 1, `beta * t` and `beta ** t` give identical results (0.8*1 = 0.8^1 = 0.8), so the bug only manifests from step 2 onward. The `*` vs `**` visual difference is easy to miss during code review, especially when multiple other bugs in the same function demand attention.

---

### BUG-041: `Models/Initializations/kaiming.py` L20

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | initialization |
| Assignment | Stage II - Task 5: Initialization |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] |

**Symptom**: Kaiming normal initialization variance is too small by a factor of 2, leading to vanishing gradients.

**Root Cause**: The formula uses 1.0 / fan instead of 2.0 / fan.

**Fix**: Change 1.0 / fan to 2.0 / fan.

**Chief Reasoning**:
- *chief_a*: Models/Initializations/kaiming.py line 20: `std = math.sqrt(1.0 / fan)`. He (2015) formula for ReLU is sqrt(2/fan). The code uses 1/fan instead of 2/fan, halving the variance. This leads to signal attenuation in deep ReLU networks.
- *chief_b*: Models/Initializations/kaiming.py line 20: `std = math.sqrt(1.0 / fan)`. He initialization for ReLU requires sqrt(2.0 / fan). The factor-of-2 difference halves the variance, potentially causing signal to vanish in deep networks.

---

### BUG-042: `Models/Initializations/kaiming.py` L37

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | initialization |
| Assignment | Stage II - Task 5: Initialization |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Kaiming uniform initialization has wrong bound, leading to vanishing activations after ReLU layers

**Root Cause**: std = sqrt(1.0 / fan) instead of the correct He formula std = sqrt(2.0 / fan)

**Fix**: Change `math.sqrt(1.0 / fan)` to `math.sqrt(2.0 / fan)` in kaiming_uniform_

**Chief Reasoning**:
- *chief_a*: Same bug as BUG-041, in kaiming_uniform_ (line 37). Both use sqrt(1.0/fan) instead of sqrt(2.0/fan).
- *chief_b*: Same bug in kaiming_uniform_ (line 37): `math.sqrt(1.0 / fan)` instead of `math.sqrt(2.0 / fan)`. Same file, same mathematical error, same fix as BUG-041.

---

### BUG-043: `Models/Initializations/xavier.py` L19

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | initialization |
| Assignment | Stage II - Task 5: Initialization |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] |

**Symptom**: Xavier normal initialization variance is incorrect, leading to poor convergence.

**Root Cause**: The formula uses fan_in * fan_out instead of fan_in + fan_out.

**Fix**: Change fan_in * fan_out to fan_in + fan_out.

**Chief Reasoning**:
- *chief_a*: Models/Initializations/xavier.py line 19: `std = gain * math.sqrt(2.0 / (fan_in * fan_out))`. Glorot (2010) uses fan_in + fan_out in the denominator, not fan_in * fan_out. This dramatically reduces the scale for typical layer sizes. Note: xavier_uniform_ at line 30 has the identical bug.
- *chief_b*: Models/Initializations/xavier.py line 19: `math.sqrt(2.0 / (fan_in * fan_out))` should be `math.sqrt(2.0 / (fan_in + fan_out))`. Product instead of sum drastically underestimates the correct std for layers with many units. NOTE: xavier_uniform_ (line ~30) has the exact same bug but was not explicitly called out — both must be fixed.

---

### BUG-044: `Models/Normalizations/layernorm.py` L40

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | normalization |
| Assignment | Stage II - Task 4: Regularization |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] |

**Symptom**: LayerNorm applies the affine transformation incorrectly, multiplying by bias and adding weight.

**Root Cause**: The formula uses x_norm * self.bias + self.weight instead of x_norm * self.weight + self.bias.

**Fix**: Change to x_norm * self.weight + self.bias.

**Chief Reasoning**:
- *chief_a*: Models/Normalizations/layernorm.py line 40: `return x_norm * self.bias + self.weight`. The affine transform is swapped: weight (scale) is used as additive and bias (shift) as multiplicative. Correct: `x_norm * self.weight + self.bias`.
- *chief_b*: Models/Normalizations/layernorm.py line 40: `x_norm * self.bias + self.weight`. Affine transform should be `x_norm * self.weight + self.bias`. Weight (gamma) scales, bias (beta) shifts. Swapping them makes the scale additive and the shift multiplicative, breaking normalization semantics.

---

### BUG-045: `Models/conv.py` L142

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | other |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] |

**Symptom**: DepthwiseSeparableConv applies pointwise conv before depthwise conv, inverting the expected order and changing the computation semantics.

**Root Cause**: Forward method calls `self.depthwise_conv(self.pointwise_conv(x))` but should call `self.pointwise_conv(self.depthwise_conv(x))`.

**Fix**: Change forward to `return self.pointwise_conv(self.depthwise_conv(x))`.

**Chief Reasoning**:
- *chief_a*: Same as BUG-012 (depthwise/pointwise order reversed). Framed here as a Stage II semantic bug for in_ch==out_ch cases.
- *chief_b*: Duplicate of BUG-012. Same DepthwiseSeparableConv order bug, just classified as stage2 (mechanism) rather than stage1 (crash). When in_ch == out_ch it doesn't crash but still computes the wrong operation.

---

### BUG-046: `Models/conv.py` L159

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | other |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: DepthwiseSeparableConv applies pointwise before depthwise, reversing the intended factorization and changing the learned function

**Root Cause**: Forward returns `self.depthwise_conv(self.pointwise_conv(x))` — standard depthwise-separable conv applies depthwise first, then pointwise

**Fix**: Change to `return self.pointwise_conv(self.depthwise_conv(x))`

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-012.
- *chief_b*: Duplicate of BUG-012.

---

### BUG-047: `Models/conv.py` L172

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | other |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] |

**Symptom**: DepthwiseSeparableConv applies pointwise before depthwise, reversing the standard depthwise-separable convolution order and changing the learned function.

**Root Cause**: forward returns self.depthwise_conv(self.pointwise_conv(x)) instead of self.pointwise_conv(self.depthwise_conv(x)).

**Fix**: Swap to return self.pointwise_conv(self.depthwise_conv(x)).

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-012.
- *chief_b*: Duplicate of BUG-012.

---

### BUG-048: `Models/dropout.py` L15

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | regularization |
| Assignment | Stage II - Task 4: Regularization |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: Activations are scaled incorrectly during training, altering the expected value and leading to poor convergence or exploding gradients.

**Root Cause**: Inverted dropout scales surviving elements by `1 / p` instead of `1 / (1 - p)`.

**Fix**: Change the scaling factor to divide by `(1.0 - self.p)`.

**Chief Reasoning**:
- *chief_a*: Models/dropout.py line 15: `return x * mask / self.p`. Inverted dropout should scale by 1/(1-p) to preserve expected value. With p=0.1, correct scale is ~1.11 but code uses 1/0.1=10, amplifying surviving activations 10x. Fix: divide by (1.0 - self.p).
- *chief_b*: Models/dropout.py line 15: `return x * mask / self.p`. Inverted dropout should scale by 1/(1-p). With p=0.1, surviving elements are scaled by 10x instead of ~1.11x, causing activations to explode. Fix: divide by `(1.0 - self.p)`.

---

### BUG-049: `Models/encoder.py` L61

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | attention |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] |

**Symptom**: The model mixes data across different batch elements, destroying batch independence and ruining gradients.

**Root Cause**: The `permute(2, 0, 1, 3)` call creates an [H, B, L, d_k] layout, but the subsequent `view` operations assume a [B, H, L, d_k] layout, causing batch and head dimensions to be interleaved.

**Fix**: Use `permute(0, 2, 1, 3)` for q, k, and v, and `permute(0, 2, 1, 3)` for the output tensor before reshaping.

**Chief Reasoning**:
- *chief_a*: Models/encoder.py line 61: `q.permute(2, 0, 1, 3)` on [B,L,H,d_k] produces [H,B,L,d_k], not [B,H,L,d_k]. After contiguous().view(B*H,L,d_k), the data is H-major. The attention computation itself happens to work (mask repeat is also H-major), BUT the output view(B,H,L,d_k) assumes B-major, scrambling batch and head dimensions. Output permute(1,2,0,3) is also wrong (should be (0,2,1,3)). Note: fixing input permute to (0,2,1,3) also requires fixing the mask expansion from repeat(H,1,1) to a B-major scheme.
- *chief_b*: Models/encoder.py line ~61: `q.permute(2, 0, 1, 3)` on [B,L,H,d_k] produces [H,B,L,d_k]. After view(B*H,L,d_k), batch and head dims are interleaved. The output permute(1,2,0,3) on [B,H,L,d_k] similarly produces [H,L,B,d_k], mixing batch elements. Both should be permute(0,2,1,3). This corrupts attention by mixing data across batch elements.

---

### BUG-050: `Models/encoder.py` L132

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | attention |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: The self-attention sublayer has no effect; the block effectively drops back to the residual path only.

**Root Cause**: Immediately after `self.self_att(out, mask)`, the code overwrites the result with `out = res` instead of adding the residual connection.

**Fix**: Keep the attention output and combine it with the residual, e.g. `out = self.drop(out) + res` (or equivalent layer-drop logic).

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-037. Same attention output overwritten by residual.
- *chief_b*: Duplicate of BUG-037. Same attention residual overwrite: `out = res` discards self-attention output.

---

### BUG-051: `Models/qanet.py` L80

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | attention |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: Context-query attention uses the wrong masks: context PAD tokens are treated as query PADs and vice versa, producing incorrect attention weights

**Root Cause**: Mask arguments are swapped in `self.cq_att(Ce, Qe, qmask, cmask)` — the CQAttention signature expects `(C, Q, cmask, qmask)`

**Fix**: Change to `self.cq_att(Ce, Qe, cmask, qmask)`

**Chief Reasoning**:
- *chief_a*: Models/qanet.py line 80: `self.cq_att(Ce, Qe, qmask, cmask)`. CQAttention.forward signature is (C, Q, cmask, qmask). Passing qmask as cmask and cmask as qmask swaps which positions are treated as padding. With para_limit≠ques_limit, this also causes shape mismatches in the mask broadcasting.
- *chief_b*: Models/qanet.py line 80: `self.cq_att(Ce, Qe, qmask, cmask)` but CQAttention signature is (C, Q, cmask, qmask). This swaps the masks. Inside CQAttention, qmask_param (actually cmask [B,400]) is unsqueezed to [B,1,400] and used in mask_logits with S [B,400,50]. Broadcasting [B,400,50] with [B,1,400] fails on dim 2 (50 vs 400) → RuntimeError. NOTE: This is actually a Stage-I crash bug (not stage2 as originally classified) because para_limit≠ques_limit makes the shapes incompatible.

---

### BUG-052: `Optimizers/adam.py` L53

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | optimizer |
| Assignment | Stage II - Task 1: Optimizer |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: If weight_decay > 0, the optimizer pushes parameters away from zero instead of regularizing them toward zero.

**Root Cause**: The L2 term is added with alpha=-wd, forming grad - wd * p instead of grad + wd * p.

**Fix**: Add the weight-decay contribution with a positive sign, or implement decoupled decay separately.

**Chief Reasoning**:
- *chief_a*: Optimizers/adam.py line 53: `grad = grad.add(p, alpha=-wd)` computes grad - wd*p. L2 regularization requires grad + wd*p. The negative sign pushes weights away from zero (anti-regularization). Fix: alpha=wd (positive).
- *chief_b*: Optimizers/adam.py line 53: `grad = grad.add(p, alpha=-wd)` computes grad - wd*p. L2 regularization requires grad + wd*p (pushing parameters toward zero). The negative sign pushes parameters away from zero. Fix: alpha=wd (positive).

---

### BUG-053: `Optimizers/adam.py` L67

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | optimizer |
| Assignment | Stage II - Task 1: Optimizer |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] |

**Symptom**: Adam optimizer fails to converge because the second moment estimate uses the gradient instead of the squared gradient.

**Root Cause**: grad is added to v instead of grad**2.

**Fix**: Change grad to grad.pow(2) or grad * grad in the second moment update.

**Chief Reasoning**:
- *chief_a*: Optimizers/adam.py line 67: `v.mul_(beta2).add_(grad, alpha=1-beta2)`. The second moment should track grad**2, not grad. Using raw grad makes v an EMA of the gradient (like m), destroying Adam's adaptive learning rate mechanism. Fix: add grad.pow(2) instead of grad.
- *chief_b*: Optimizers/adam.py line ~67: `v.mul_(beta2).add_(grad, alpha=1.0 - beta2)`. The second moment should track grad**2, not grad. Without squaring, v tracks the same quantity as m (first moment), breaking Adam's adaptive learning rate. Fix: `v.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)`.

---

### BUG-054: `Optimizers/sgd.py` L38

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | optimizer |
| Assignment | Stage II - Task 1: Optimizer |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: Weight decay subtracts wd*p from gradient instead of adding it, implementing negative L2 regularization

**Root Cause**: grad.add(p, alpha=-wd) uses a negative alpha; should be positive for L2 regularization

**Fix**: Change alpha=-wd to alpha=wd

**Chief Reasoning**:
- *chief_a*: Optimizers/sgd.py line 38: `grad = grad.add(p, alpha=-wd)` gives grad - wd*p. Then p.add_(grad, alpha=-lr) gives p - lr*(grad-wd*p) = p - lr*grad + lr*wd*p. The +lr*wd*p term increases weight magnitude — negative regularization. Fix: alpha=wd.
- *chief_b*: Optimizers/sgd.py line 38: `grad = grad.add(p, alpha=-wd)` computes grad - wd*p. Same as BUG-053: L2 regularization needs positive alpha. Fix: alpha=wd.

---

### BUG-055: `Optimizers/sgd_momentum.py` L47

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | optimizer |
| Assignment | Stage II - Task 1: Optimizer |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: Momentum update subtracts the gradient instead of adding it, reversing the effective gradient direction

**Root Cause**: v.mul_(mu).sub_(grad) computes v = mu*v - grad; the documented rule and correct formula is v = mu*v + grad

**Fix**: Change v.mul_(mu).sub_(grad) to v.mul_(mu).add_(grad)

**Chief Reasoning**:
- *chief_a*: Optimizers/sgd_momentum.py line 47: `v.mul_(mu).sub_(grad)` computes v = mu*v - grad. Then p.add_(v, alpha=-lr) gives p + lr*grad (gradient ascent). The docstring says v = mu*v + grad. Fix: change .sub_ to .add_.
- *chief_b*: Optimizers/sgd_momentum.py line 47: `v.mul_(mu).sub_(grad)` computes v = mu*v - grad. Docstring says v = mu*v + grad. With the subsequent `p.add_(v, alpha=-lr)`, i.e., p -= lr*v, using sub_ makes p += lr*grad (gradient ascent via the subtracted gradient). Fix: `.add_(grad)` instead of `.sub_(grad)`.

---

### BUG-056: `Schedulers/cosine_scheduler.py` L25

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | lr_scheduler |
| Assignment | Stage II - Task 2: LR Scheduler |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] |

**Symptom**: Learning rate does not decay to eta_min properly, it decays to a different value because the 0.5 factor is missing.

**Root Cause**: The formula is missing the 0.5 multiplier before (base_lr - self.eta_min).

**Fix**: Add 0.5 * before (base_lr - self.eta_min).

**Chief Reasoning**:
- *chief_a*: Schedulers/cosine_scheduler.py line 25: formula is `eta_min + (base_lr - eta_min) * (1 + cos(...))`. Missing the 0.5 factor. At t=0: lr = eta_min + 2*(base_lr-eta_min) = 2*base_lr - eta_min, which is ~2x the intended initial lr. Correct: `eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(...))`.
- *chief_b*: Schedulers/cosine_scheduler.py line 25: formula is `eta_min + (base_lr - eta_min) * (1 + cos(...))`. Missing the 0.5 factor. At t=0, cos(0)=1, so lr = eta_min + 2*(base_lr - eta_min) = 2*base_lr - eta_min, which exceeds the initial lr. Correct: `0.5 * (base_lr - eta_min) * (1 + cos(...))`. Note: this line also crashes first due to math.PI (BUG-025).

---

### BUG-057: `Schedulers/lambda_scheduler.py` L21

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | lr_scheduler |
| Assignment | Stage II - Task 2: LR Scheduler |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] | claude-opus-4-6[think:adaptive] | gemini-3.1-pro-preview[think:high] |

**Symptom**: The lambda output is added to the base learning rate, so a factor of 1.0 increases lr by 1 instead of leaving it unchanged.

**Root Cause**: The scheduler uses addition instead of multiplication when applying lr_lambda(t).

**Fix**: Return [base_lr * factor for base_lr in self.base_lrs].

**Chief Reasoning**:
- *chief_a*: Schedulers/lambda_scheduler.py line 21: `return [base_lr + factor ...]`. LambdaLR should multiply base_lr by the lambda output, not add. With lambda returning 1.0 and base_lr=1.0: code gives lr=2.0 instead of lr=1.0. Fix: base_lr * factor.
- *chief_b*: Schedulers/lambda_scheduler.py line 21: `return [base_lr + factor for base_lr in self.base_lrs]`. LambdaLR should multiply: `base_lr * factor`. With + and factor=1.0, lr = base_lr + 1.0 instead of base_lr * 1.0, doubling the effective learning rate for Adam (base_lr=1.0).

---

### BUG-058: `Schedulers/step_scheduler.py` L23

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | lr_scheduler |
| Assignment | Stage II - Task 2: LR Scheduler |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] |

**Symptom**: Learning rate decays linearly or becomes zero instead of decaying exponentially.

**Root Cause**: The formula multiplies by gamma * n instead of gamma ** n.

**Fix**: Change self.gamma * (t // self.step_size) to self.gamma ** (t // self.step_size).

**Chief Reasoning**:
- *chief_a*: Schedulers/step_scheduler.py line 23: `base_lr * self.gamma * (t // self.step_size)`. At t=0: gamma*0=0, so lr=0 immediately! The formula should use exponentiation: gamma ** (t // step_size). At t=0: gamma^0=1, lr=base_lr. At t=step_size: gamma^1, etc.
- *chief_b*: Schedulers/step_scheduler.py line 23: `base_lr * self.gamma * (t // self.step_size)`. Should be `gamma ** (t // step_size)`. At t=0, (t//step_size)=0, so lr = base_lr * gamma * 0 = 0. Learning rate is zero from the start! Even after step_size steps, it decays linearly instead of exponentially. Fix: use `**` instead of `*`.

---

### BUG-059: `TrainTools/train_utils.py` L31

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | training_loop |
| Assignment | Stage II - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] | gpt-5.4-pro[reason:xhigh] | gemini-3.1-pro-preview[think:high] |

**Symptom**: Gradient clipping has no effect because optimizer.step() is called before clip_grad_norm_, so parameters are updated with unclipped gradients.

**Root Cause**: optimizer.step() and clip_grad_norm_ are in the wrong order; clipping must occur after backward but before step.

**Fix**: Move `torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)` to before `optimizer.step()`.

**Chief Reasoning**:
- *chief_a*: TrainTools/train_utils.py line 31: `optimizer.step()` precedes `clip_grad_norm_()`. Parameters are updated with unclipped gradients; the subsequent clipping has no effect. Must be: backward → clip → step. Note: currently masked by BUG-032 (loss.item().backward() crash).
- *chief_b*: TrainTools/train_utils.py lines 31-32: `optimizer.step()` is called BEFORE `clip_grad_norm_()`. Gradients are applied unclipped, then clipping happens after the update (which is pointless). Fix: move clip_grad_norm_ before optimizer.step(). (Note: line 30's loss.item().backward() bug (BUG-032) prevents reaching this code, but once fixed this ordering bug manifests.)

---

## Should Fix (18)

### BUG-027: `Data/squad.py` L40

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | data_pipeline |
| Assignment | Stage I - Task 1: Data Pipeline |
| Confidence | medium |
| Discovered by | gemini-3.1-pro-preview[think:high] |

**Symptom**: Crash with `ValueError: given numpy array is not writeable` when initializing the dataset.

**Root Cause**: `np.load` on `.npz` files returns read-only memory-mapped numpy arrays. `torch.from_numpy` requires writeable arrays because it shares memory with the underlying numpy array, causing it to reject the read-only arrays.

**Fix**: Use `torch.tensor(data['...'])` or `torch.from_numpy(data['...'].copy())` to safely create the tensors.

**Chief Reasoning**:
- *chief_a*: In numpy >=1.16.3, arrays from .npz files are loaded via np.frombuffer on immutable bytes, yielding non-writable arrays. In PyTorch >=2.0, torch.from_numpy rejects non-writable arrays with RuntimeError. However, the subsequent .long() call on int32 arrays would create a copy, potentially avoiding the issue for those dtypes. Environment-dependent but a real risk. Using .copy() or torch.tensor() is the safe fix.
- *chief_b*: np.load on .npz files reads arrays into memory (not memory-mapped), so they are writable by default. Even if read-only in some edge-case numpy version, the subsequent .long() call performs a dtype conversion (npz typically stores int32, .long() produces int64), creating a new writable tensor. No crash is reproducible under standard PyTorch+NumPy versions.

---

### BUG-028: `Tools/download.py` L122

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | data_pipeline |
| Assignment | Stage I - Task 1: Data Pipeline |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Running `download_mini()` and then calling `preprocess()` with default arguments crashes with `FileNotFoundError` for the train JSON and GloVe text file.

**Root Cause**: The mini-data path produces `train-mini.json` and `glove.mini.txt`, but `preprocess()` defaults look for `train-v1.1.json` and `glove.840B.300d.txt`.

**Fix**: Make the mini download produce filenames compatible with preprocessing defaults, or provide mini-specific preprocess defaults/wrappers that point at `train-mini.json` and `glove.mini.txt`.

**Chief Reasoning**:
- *chief_a*: download_mini() produces train-mini.json and glove.mini.txt, but preprocess() defaults expect train-v1.1.json and glove.840B.300d.txt. The mini-data workflow breaks unless the user manually overrides paths. Real usability bug but not a code logic error.
- *chief_b*: download_mini() creates train-mini.json and glove.mini.txt, but preprocess() defaults expect train-v1.1.json and glove.840B.300d.txt. Users following the mini-data workflow hit FileNotFoundError. Real but lower priority since users can pass explicit paths.

---

### BUG-029: `Tools/download.py` L170

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | data_pipeline |
| Assignment | Stage I - Task 1: Data Pipeline |
| Confidence | medium |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: `download()` and `download_mini()` fail at the spaCy step on modern environments because `python -m spacy download en` exits non-zero.

**Root Cause**: The downloader hardcodes the obsolete spaCy shortcut model name `en` instead of a valid package name such as `en_core_web_sm`.

**Fix**: Replace the default model name with a current spaCy package name, or make the spaCy download optional if preprocessing no longer depends on spaCy.

**Chief Reasoning**:
- *chief_a*: Tools/download.py line 170: `spacy download en` uses the obsolete shortcut 'en' which fails in spaCy v3+. Modern spaCy requires 'en_core_web_sm'. Environment-dependent; only affects the download step, not model code.
- *chief_b*: Tools/download.py: `spacy download en` uses an obsolete shortcut name that fails on modern spaCy (>=3.0). Should use 'en_core_web_sm'. Environment-dependent but affects most current installations.

---

### BUG-030: `Tools/preproc.py` L248

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | data_pipeline |
| Assignment | Stage I - Task 1: Data Pipeline |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: The advertised mini-data workflow breaks: after `download_mini()`, calling `preprocess()` with defaults raises `FileNotFoundError` because the default raw-data paths do not exist in the mini-data layout.

**Root Cause**: `preprocess()` defaults point to `_data/squad/train-v1.1.json` and `_data/glove/glove.840B.300d.txt`, while `download_mini()` produces `_data/squad/train-mini.json` and `_data/glove/glove.mini.txt`.

**Fix**: Align `preprocess()` defaults with the mini-data filenames, or add an explicit mini-data mode that switches to `train-mini.json` and `glove.mini.txt`.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-026. Same mini-data vs default-path mismatch.
- *chief_b*: Duplicate of BUG-026. Same mini-data path mismatch from the preproc.py perspective.

---

### BUG-031: `Tools/preproc.py` L303

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | data_pipeline |
| Assignment | Stage I - Task 1: Data Pipeline |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: If callers use the dict returned by `preprocess()` as the training/eval config, downstream cache loaders immediately fail with missing attributes such as `train_npz`, `dev_npz`, `word_emb_json`, or `dev_eval_json`.

**Root Cause**: `preprocess()` publishes stale key names like `train_record_file`, `dev_record_file`, `word_emb_file`, and `char_emb_file`, while the rest of the data pipeline expects `train_npz`, `dev_npz`, `word_emb_json`, `char_emb_json`, `train_eval_json`, and `dev_eval_json`.

**Fix**: Rename the returned keys to the same `*_npz` / `*_json` names used everywhere else in the pipeline, or update downstream code to use one consistent convention.

**Chief Reasoning**:
- *chief_a*: preprocess() returns keys like 'train_record_file', 'word_emb_file' etc., but downstream consumers (Data/io.py, Data/squad.py) expect 'train_npz', 'word_emb_json' etc. Impact is limited if train()/evaluate() use their own default paths, but any code that chains preprocess() output as config will fail.
- *chief_b*: preprocess() returns keys like 'train_record_file', 'word_emb_file' etc., while Data/io.py and Data/squad.py expect 'train_npz', 'word_emb_json'. If anyone programmatically passes the return dict as config, downstream fails. However, the default workflow (preprocess writes to disk, train reads from disk using its own defaults) works. Confirmed but not a crash blocker in the intended workflow.

---

### BUG-032: `Tools/preproc.py` L336

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | data_pipeline |
| Assignment | Stage I - Task 1: Data Pipeline |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: If the caller uses the public `preprocess()` return value to configure downstream data utilities, later calls to `sanity_check_cache`, `load_word_char_mats`, or `load_*_eval` fail with missing attributes/keys such as `train_npz`, `word_emb_json`, or `dev_eval_json`.

**Root Cause**: `preprocess()` returns path keys named `train_record_file`, `dev_record_file`, `word_emb_file`, `char_emb_file`, `train_eval_file`, and `dev_eval_file`, but the data-loading helpers in `Data/io.py` and `Data/squad.py` expect the schema `train_npz`, `dev_npz`, `word_emb_json`, `char_emb_json`, `train_eval_json`, and `dev_eval_json`.

**Fix**: Make the preprocessing output key names match the names consumed by the data loaders/cache checker, or update all consumers to a single shared schema.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-029. Same preprocess output key mismatch.
- *chief_b*: Duplicate of BUG-029. Same return-key naming mismatch.

---

### BUG-060: `EvaluateTools/evaluate.py` L83

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | evaluation |
| Assignment | Stage II - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: `evaluate()` cannot faithfully reconstruct checkpoints trained with non-default model options and may instantiate the wrong model or fail to load weights.

**Root Cause**: The evaluator rebuilds `args` from a partial hard-coded list instead of the saved checkpoint config, omitting training-time model-defining fields such as normalization-related options.

**Fix**: Load the checkpoint config first and rebuild the model from `ckpt['config']` (or expose/pass through every model-defining argument).

**Chief Reasoning**:
- *chief_a*: Subsumed by BUG-040. BUG-039 focuses on model options; BUG-040 is the same root cause stated more broadly.
- *chief_b*: evaluate.py rebuilds the model from function defaults instead of loading ckpt['config']. If training used non-default options (e.g., different activation, normalization, or init), the eval model architecture silently differs from the trained one, producing wrong results despite weights loading without error (shapes may coincidentally match). Fix: load and use ckpt['config'].

---

### BUG-061: `EvaluateTools/evaluate.py` L118

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | evaluation |
| Assignment | Stage II - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Shape-compatible non-default checkpoints can load and run under different math than the model that was trained (for example different normalization/activation or different embedding/vocab metadata with the same tensor shapes), producing silently wrong evaluation metrics.

**Root Cause**: `train()` persists the resolved config, but `evaluate()` never reloads it; the checkpoint is read only after embeddings/model have already been built from the function's own defaults.

**Fix**: Load `ckpt['config']` (or `run_config.json`) first, merge only evaluation-specific overrides, then rebuild `args`, embeddings, and the model from that resolved config before `load_state_dict`.

**Chief Reasoning**:
- *chief_a*: evaluate.py never loads ckpt['config']. If training used non-default norm_name, activation, init_name etc., the evaluator silently constructs the wrong model architecture. With differing layer types (e.g., group_norm vs layer_norm), load_state_dict would fail with key/shape mismatches. With shape-compatible but different configs, evaluation produces silently wrong results. save_checkpoint stores 'config' in the checkpoint, but evaluate() ignores it.
- *chief_b*: Same concern as BUG-039: evaluate() doesn't reload the training config from the checkpoint. Different angle but identical root cause and fix.

---

### BUG-062: `Models/Initializations/xavier.py` L30

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | initialization |
| Assignment | Stage II - Task 5: Initialization |
| Confidence | ? |
| Discovered by | claude-opus-4-6[think:adaptive] (chief) |

**Symptom**: Xavier uniform initialization has incorrect variance, producing values that are orders of magnitude too small for typical layer sizes.

**Root Cause**: `xavier_uniform_` uses `fan_in * fan_out` instead of `fan_in + fan_out` in the denominator — the same bug as `xavier_normal_` (BUG-043). For a layer with fan_in=96, fan_out=96: code gives sqrt(2/(96*96))=0.0147 vs correct sqrt(2/(96+96))=0.102, a 7x difference.

**Fix**: Change `math.sqrt(2.0 / (fan_in * fan_out))` to `math.sqrt(2.0 / (fan_in + fan_out))` in `xavier_uniform_`.

**Why Missed by Teams**: BUG-043 only explicitly references xavier_normal_ (line 19). Teams examining the file likely spotted the bug in one function and assumed (or didn't verify) the other function was correct, or simply didn't report both locations separately.

---

### BUG-063: `Models/encoder.py` L67

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | attention |
| Assignment | Stage II - Task 3: Attention Mechanism |
| Confidence | ? |
| Discovered by | claude-opus-4-6[think:adaptive] (chief) |

**Symptom**: After fixing BUG-049's input permute to (0,2,1,3) making data B-major, the attention mask remains H-major due to `repeat(self.num_heads, 1, 1)`. This causes wrong mask-to-data alignment: data element b*H+h (batch b, head h) gets masked by batch (b*H+h)%B instead of batch b.

**Root Cause**: `mask.unsqueeze(1).expand(-1, length, -1).repeat(self.num_heads, 1, 1)` produces an H-major [H*B, L, L] tensor (B elements repeated H times). With B-major data [B*H, L, d_k], element i corresponds to batch i//H but the mask corresponds to batch i%B. These only align when B==1 or H divides B evenly.

**Fix**: Replace the mask expansion with a B-major scheme: `mask.unsqueeze(1).expand(-1,L,-1).unsqueeze(1).expand(-1,H,-1,-1).reshape(B*H,L,L)` to match B-major data ordering.

**Why Missed by Teams**: With the current buggy permute(2,0,1,3) creating H-major data, the H-major mask is actually *consistent* — so the attention computation works correctly with the current code. The mask bug is latent and only manifests when BUG-049's input permute is fixed. Teams analyzing the attention only saw the permute issue, not the coupled mask ordering dependency.

---

### BUG-064: `Models/qanet.py` L46

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | embedding |
| Assignment | Stage II - Task 7: Embedding |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: GloVe word embeddings are fine-tuned during training instead of being frozen, degrading generalization.

**Root Cause**: nn.Embedding.from_pretrained for word_emb uses freeze=False instead of freeze=True.

**Fix**: Change freeze=False to freeze=True for word_emb.

**Chief Reasoning**:
- *chief_a*: Models/qanet.py line 46: `nn.Embedding.from_pretrained(word_mat, freeze=False)`. The original QANet paper freezes GloVe word embeddings to prevent catastrophic forgetting of pretrained representations. freeze=False allows fine-tuning, degrading generalization especially with small datasets.
- *chief_b*: Models/qanet.py line 46: `nn.Embedding.from_pretrained(..., freeze=False)` for word_emb. Standard practice freezes pretrained GloVe embeddings to prevent overfitting on the relatively small SQuAD dataset. freeze=False allows fine-tuning, which degrades generalization. Fix: freeze=True.

---

### BUG-065: `Tools/preproc.py` LNone

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: For multi-answer questions (notably in dev), an example can be filtered based on one gold span but saved with a different gold span, making validation supervision inconsistent and able to bypass the intended span-length filter.

**Root Cause**: `filter_func()` uses `ex['y1s'][0]` / `ex['y2s'][0]`, but the labels written to the NPZ use `example['y1s'][-1]` / `example['y2s'][-1]`.

**Fix**: Use the same answer index consistently for both filtering and stored labels (typically the first answer), or store/handle multiple valid spans explicitly.

**Chief Reasoning**:
- *chief_a*: Tools/preproc.py: filter_func uses ex['y1s'][0] and ex['y2s'][0] (first answer), but the saved labels use example['y1s'][-1] and example['y2s'][-1] (last answer). For multi-answer dev questions, the filter may pass based on answer[0] but store a different (potentially longer) answer[-1]. Inconsistent filtering and labeling.
- *chief_b*: Tools/preproc.py build_features: filter_func uses `ex['y1s'][0]`/`ex['y2s'][0]` but saved labels use `example['y1s'][-1]`/`example['y2s'][-1]`. For multi-answer dev questions, a different answer span is filtered vs saved. The saved span could exceed ans_limit if only the first answer was short enough.

---

### BUG-066: `Tools/preproc.py` L173

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | gpt-5.4-pro[reason:xhigh] |

**Symptom**: For dev-set questions that have multiple annotated answers, the last answer span is selected as the ground-truth label instead of the first, producing different y1/y2 targets than the canonical convention and potentially degrading validation loss signal.

**Root Cause**: In `build_features`, `example["y1s"][-1]` and `example["y2s"][-1]` select the last answer span instead of the first (`[0]`). SQuAD v1.1 convention is to use the first annotated answer for supervision.

**Fix**: Change `example["y1s"][-1]` to `example["y1s"][0]` and `example["y2s"][-1]` to `example["y2s"][0]`.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-060. Same y1s[-1]/y2s[-1] vs y1s[0]/y2s[0] inconsistency.
- *chief_b*: Duplicate of BUG-060. Same observation: y1s[-1]/y2s[-1] used for saving instead of [0].

---

### BUG-067: `Tools/preproc.py` L195

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] |

**Symptom**: The filter_func checks answer span length using the first answer (index [0]), but the features that are actually saved use the last answer (index [-1]). For dev-set questions with multiple answers of differing lengths, this means (a) the saved span may exceed ans_limit because only the first answer was checked, and (b) a different (last) answer is selected as ground truth instead of the canonical first answer, leading to inconsistent validation targets.

**Root Cause**: build_features appends example['y1s'][-1] and example['y2s'][-1] instead of example['y1s'][0] and example['y2s'][0], while filter_func uses index [0].

**Fix**: Change `example["y1s"][-1]` to `example["y1s"][0]` and `example["y2s"][-1]` to `example["y2s"][0]` so the saved ground truth matches the index used in the filter.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-060.
- *chief_b*: Duplicate of BUG-060. Combines the filter/save inconsistency observation.

---

### BUG-068: `Tools/preproc.py` L296

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: If `fasttext=True` without `fasttext_file` or `pretrained_char=True` without `glove_char_file`, preprocessing silently emits random embeddings instead of the requested pretrained matrices.

**Root Cause**: The selected embedding path becomes `None`, and `get_embedding(..., emb_file=None)` interprets `None` as 'randomly initialize' rather than as a configuration error.

**Fix**: Validate that the corresponding embedding file path is provided (and exists) whenever a pretrained-embedding flag is enabled, and raise an error otherwise.

**Chief Reasoning**:
- *chief_a*: Tools/preproc.py: If fasttext=True but fasttext_file=None, word_emb_source becomes None. get_embedding(emb_file=None) falls through to random initialization silently. Same for pretrained_char=True without glove_char_file. Should raise a configuration error.
- *chief_b*: Tools/preproc.py: when fasttext=True without fasttext_file, word_emb_source=None. get_embedding(emb_file=None) falls into the random-init branch, silently producing random word embeddings. Same for pretrained_char=True without glove_char_file. Should validate and raise an error.

---

### BUG-069: `Tools/preproc.py` L353

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Enabling `fasttext=True` without `fasttext_file`, or `pretrained_char=True` without `glove_char_file`, silently produces random embeddings instead of the requested pretrained ones, so training runs with the wrong input representations.

**Root Cause**: Those flag-controlled branches pass `None` into `get_embedding()`, and `get_embedding()` interprets `emb_file=None` as the random-initialization path rather than raising a configuration error.

**Fix**: Validate that `fasttext_file` / `glove_char_file` are provided whenever the corresponding flags are enabled, and raise a clear error otherwise.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-065.
- *chief_b*: Duplicate of BUG-065. Same silent-random-embedding issue.

---

### BUG-070: `TrainTools/train.py` L148

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | training_loop |
| Assignment | Stage II - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: Early stopping almost never triggers: patience only increments when *both* F1 and EM are strictly worse than their respective bests, so any minor fluctuation in one metric resets patience indefinitely.

**Root Cause**: The condition `if dev_f1 < best_f1 and dev_em < best_em` uses `and` (both must degrade) with strict `<` (ties don't count). The correct early-stop condition should be that *neither* metric improved, i.e. `dev_f1 <= best_f1 and dev_em <= best_em`, or equivalently the improvement branch should check `dev_f1 > best_f1 or dev_em > best_em`.

**Fix**: Change the condition to check for lack of any improvement, e.g. swap `and` for `or`: `if dev_f1 < best_f1 or dev_em < best_em:` (or restructure to check for improvement first).

**Chief Reasoning**:
- *chief_a*: TrainTools/train.py line 148: `if dev_f1 < best_f1 and dev_em < best_em` requires BOTH metrics to strictly decline to increment patience. If either metric stays flat or improves even slightly, patience resets. This makes early stopping nearly impossible to trigger. Fix: use `or` or restructure to check for any improvement.
- *chief_b*: TrainTools/train.py line ~148: `if dev_f1 < best_f1 and dev_em < best_em: patience += 1`. With 'and', patience only increments when BOTH metrics worsen. If either metric improves (or even ties at best), patience resets. Early stopping is effectively disabled since it's rare for both F1 and EM to simultaneously drop below their independent bests.

---

### BUG-071: `Tools/preproc.py` L184

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | minor |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | high |
| Discovered by | claude-opus-4-6[think:adaptive] |

**Symptom**: For dev-set examples (which have multiple annotated answers), the stored y1/y2 ground-truth span comes from the last annotator instead of the first, making dev loss inconsistent with the filter criterion (which uses [0]) and with the standard QANet/R-Net convention.

**Root Cause**: In `build_features`, `y1s.append(example["y1s"][-1])` and `y2s.append(example["y2s"][-1])` use index `[-1]` (last answer) instead of `[0]` (first answer).

**Fix**: Change `example["y1s"][-1]` to `example["y1s"][0]` and `example["y2s"][-1]` to `example["y2s"][0]` to use the first annotated answer, matching both the filter function and the standard convention.

**Chief Reasoning**:
- *chief_a*: Duplicate of BUG-060. Same answer index inconsistency ([-1] vs [0]).
- *chief_b*: Duplicate of BUG-060. Same [-1] vs [0] answer index issue.

---

## Nice to Fix (3)

### BUG-072: `EvaluateTools/evaluate.py` L38

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | evaluation |
| Assignment | Stage II - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: `evaluate()` can silently report dev loss under the wrong criterion whenever training used a non-default `loss_name`, so standalone eval loss no longer matches train history/checkpoint selection.

**Root Cause**: The evaluation entry point ignores the saved loss-registry choice from training and falls back to its own default `loss_name='qa_nll'` unless the caller manually repeats the training setting.

**Fix**: Default `loss_name` from the saved checkpoint/config and treat the function argument as an explicit override only when provided.

**Chief Reasoning**:
- *chief_a*: Subsumed by BUG-040. BUG-038 focuses on loss_name not loaded from checkpoint; BUG-040 covers the broader issue of the entire config not being loaded, which includes loss_name.
- *chief_b*: evaluate.py defaults loss_name='qa_nll' regardless of what was used during training. If training used 'qa_ce', the eval loss is computed differently. However, F1/EM metrics are unaffected (they don't depend on the loss). Only the reported loss value would be inconsistent. Low practical impact.

---

### BUG-073: `Tools/preproc.py` L131

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | medium |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: When `word_count_limit` or `char_count_limit` is set, tokens/chars appearing exactly at the documented cutoff are incorrectly dropped from the vocabulary, increasing OOVs and changing the dataset statistics.

**Root Cause**: `get_embedding()` uses a strict `> limit` test (`v > limit` / `counter[word] > limit`) even though the API documents the limit as the minimum frequency to keep.

**Fix**: Use an inclusive cutoff (`>= limit`) consistently when filtering counter entries for the embedding matrix.

**Chief Reasoning**:
- *chief_a*: Tools/preproc.py line 131: `v > limit` uses strict inequality. Tokens at exactly the limit frequency are dropped. With default limit=-1 this is harmless (all counts > -1). Only matters when word_count_limit or char_count_limit is explicitly set to a positive value.
- *chief_b*: Tools/preproc.py get_embedding: `v > limit` uses strict inequality. With default limit=-1, all tokens pass (count > -1 is always true), so no practical impact with defaults. With explicit limits, tokens at exactly the cutoff are dropped. Minor vocabulary difference.

---

### BUG-074: `Tools/preproc.py` L155

| Field | Value |
|-------|-------|
| Stage | stage2 |
| Severity | major |
| Category | preprocessing |
| Assignment | Stage II - Task 5: Preprocessing |
| Confidence | medium |
| Discovered by | gpt-5.4-pro[reason:xhigh] |

**Symptom**: Examples whose gold answer is `ans_limit + 1` tokens long are incorrectly kept, so the saved dataset can violate the configured maximum answer-span length.

**Root Cause**: `build_features.filter_func()` measures span length as `y2 - y1` instead of the inclusive token count `y2 - y1 + 1`.

**Fix**: Compute answer length inclusively (`y2 - y1 + 1`) when applying `ans_limit`, or redefine/document `ans_limit` to match the implemented formula.

**Chief Reasoning**:
- *chief_a*: Tools/preproc.py line 155: `(y2-y1) > ans_limit` measures span as y2-y1 tokens but inclusive count is y2-y1+1. A span of exactly ans_limit+1 tokens (y2-y1=ans_limit) passes the filter. Off-by-one that rarely matters in practice.
- *chief_b*: Tools/preproc.py filter_func: `(ex['y2s'][0] - ex['y1s'][0]) > ans_limit` computes exclusive length. A span from token 0 to token 30 has 31 tokens but y2-y1=30, passing the filter when ans_limit=30. Off-by-one allows one-token-too-long answers. Minor practical impact.

---

## Ignorable (1)

### BUG-033: `EvaluateTools/evaluate.py` L80

| Field | Value |
|-------|-------|
| Stage | stage1 |
| Severity | critical |
| Category | evaluation |
| Assignment | Stage I - Task 2: Train/Eval Loop |
| Confidence | high |
| Discovered by | gemini-3.1-pro-preview[think:high] | claude-opus-4-6[think:adaptive] | gpt-5.4-pro[reason:xhigh] | claude-opus-4-6[think:adaptive,budget:16000] |

**Symptom**: Evaluation crashes with `AttributeError: 'Namespace' object has no attribute 'norm_name'` (or similar) during `QANet` initialization.

**Root Cause**: The manually constructed `args` namespace in `evaluate.py` is missing several architecture configuration keys (`norm_name`, `norm_groups`, `use_batch_norm`, `activation`, `init_name`) required by `QANet`.

**Fix**: Load the configuration from the saved checkpoint (e.g., `ckpt['config']`) or add the missing arguments to the `evaluate` function and `args` namespace.

**Chief Reasoning**:
- *chief_a*: Same false positive as BUG-002. All optional model architecture attributes in QANet.__init__ use getattr with fallback defaults. No AttributeError will occur. The real issue is Stage 2 silent misconfiguration (BUG-040).
- *chief_b*: Duplicate of BUG-002. Same claim that missing args attributes cause AttributeError, but all are accessed via getattr with defaults in QANet.__init__. No crash occurs.

---

## Rejected Bugs (1)

These were flagged by audit teams but both chief engineers agreed they are false positives.

### ~~BUG-002~~: `EvaluateTools/evaluate.py` L65

**Original claim**: Evaluation crashes with an AttributeError during QANet initialization due to missing configuration attributes.

**Chief A rejection**: False positive. QANet.__init__ accesses norm_name, activation, init_name, norm_groups via getattr with defaults: e.g. `str(getattr(args, 'norm_name', 'layer_norm'))`. The missing attributes in evaluate.py's Namespace will NOT cause an AttributeError — they silently fall back to defaults. The real issue (wrong defaults vs. training config) is a Stage 2 bug, correctly captured by BUG-040.
**Chief B rejection**: QANet.__init__ accesses norm_name, activation, init_name, norm_groups all via getattr with default values (e.g., `getattr(args, 'norm_name', 'layer_norm')`). The missing attributes will not raise AttributeError; they silently fall back to defaults. The real eval issue is BUG-004 (ckpt key mismatch) and BUG-039 (config not reloaded), not missing attributes.

---

## Bug Distribution by Category

| Category | Count |
|----------|-------|
| optimizer | 8 |
| preprocessing | 8 |
| normalization | 6 |
| attention | 6 |
| other | 6 |
| embedding | 6 |
| data_pipeline | 6 |
| shape_mismatch | 5 |
| evaluation | 5 |
| lr_scheduler | 4 |
| training_loop | 4 |
| initialization | 4 |
| checkpoint | 2 |
| activation | 2 |
| loss_function | 1 |
| regularization | 1 |

## Bug Distribution by Assignment Task

| Assignment Task | Count |
|-----------------|-------|
| Stage I - Task 3: Model Architecture | 18 |
| Stage II - Task 3: Attention Mechanism | 9 |
| Stage I - Task 2: Train/Eval Loop | 8 |
| Stage II - Task 5: Preprocessing | 8 |
| Stage I - Task 1: Data Pipeline | 6 |
| Stage II - Task 1: Optimizer | 5 |
| Stage II - Task 2: Train/Eval Loop | 5 |
| Stage II - Task 5: Initialization | 4 |
| Stage II - Task 4: Regularization | 3 |
| Stage II - Task 2: LR Scheduler | 3 |
| Stage II - Task 6: Activation | 2 |
| Stage I - Task 7: Embedding | 1 |
| Stage II - Task 9: Optimizer | 1 |
| Stage II - Task 7: Embedding | 1 |
