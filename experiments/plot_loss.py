import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

log_path = r"d:\Assignment\COMP4329\ass1\Assignment1_2026\experiments\final-result1_adam+lambda.md"
out_dir = r"d:\Assignment\COMP4329\ass1\Assignment1_2026\experiments\prism-uploads"

with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

steps = []
step_loss, valid_loss, test_loss = [], [], []
valid_f1, test_f1 = [], []
valid_em, test_em = [], []

for m in re.finditer(
    r"STEP\s+(\d+)\s+loss\s+([\d.]+).*?"
    r"VALID\(train\)\s+loss\s+([\d.]+)\s+F1\s+([\d.]+)\s+EM\s+([\d.]+).*?"
    r"TEST\s+loss\s+([\d.]+)\s+F1\s+([\d.]+)\s+EM\s+([\d.]+)",
    text, re.DOTALL
):
    steps.append(int(m.group(1)))
    step_loss.append(float(m.group(2)))
    valid_loss.append(float(m.group(3)))
    valid_f1.append(float(m.group(4)))
    valid_em.append(float(m.group(5)))
    test_loss.append(float(m.group(6)))
    test_f1.append(float(m.group(7)))
    test_em.append(float(m.group(8)))

print(f"Parsed {len(steps)} checkpoints, steps {steps[0]}..{steps[-1]}")

CKPT_STEP = 40800
COLORS = {"valid": "#38a169", "test": "#e53e3e", "step": "#2b6cb0", "ckpt": "#805ad5"}

def make_plot(ylabel, title, series, filename, annotate=None):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for label, data, color, lw, alpha in series:
        ax.plot(steps, data, label=label, color=color, linewidth=lw, alpha=alpha)
    ax.axvline(x=CKPT_STEP, color=COLORS["ckpt"], linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"Selected ckpt (step {CKPT_STEP})")
    if annotate:
        for ann_label, ann_step, ann_val, ann_color, offset in annotate:
            ax.plot(ann_step, ann_val, 'o', color=ann_color, markersize=7, zorder=5)
            ax.annotate(ann_label, xy=(ann_step, ann_val), xytext=offset,
                        textcoords='offset points', fontsize=8, fontweight='bold',
                        color=ann_color, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=ann_color, alpha=0.85),
                        arrowprops=dict(arrowstyle='->', color=ann_color, lw=1.2))
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="best")
    ax.set_xlim(0, steps[-1] + 500)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{out_dir}/{filename}"
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close()

ckpt_idx = steps.index(CKPT_STEP)
sel_test_f1 = test_f1[ckpt_idx]
sel_test_em = test_em[ckpt_idx]
sel_test_loss_val = test_loss[ckpt_idx]

make_plot("F1", "Adam + Lambda: F1 Trajectories", [
    ("VALID(train) F1", valid_f1, COLORS["valid"], 1.4, 1.0),
    ("TEST (dev) F1",   test_f1,  COLORS["test"],  1.4, 1.0),
], "adam_lambda_valid_test_f1_shared_step_40800_exact_style.png",
    annotate=[
        (f"Selected TEST F1 = {sel_test_f1:.2f}\nStep {CKPT_STEP}",
         CKPT_STEP, sel_test_f1, COLORS["test"], (-80, 20)),
    ])

make_plot("EM", "Adam + Lambda: EM Trajectories", [
    ("VALID(train) EM", valid_em, COLORS["valid"], 1.4, 1.0),
    ("TEST (dev) EM",   test_em,  COLORS["test"],  1.4, 1.0),
], "adam_lambda_valid_test_em_shared_step_40800_exact_style.png",
    annotate=[
        (f"Selected TEST EM = {sel_test_em:.2f}\nStep {CKPT_STEP}",
         CKPT_STEP, sel_test_em, COLORS["test"], (-80, 20)),
    ])

make_plot("Loss", "Adam + Lambda: Loss Trajectories", [
    ("VALID(train) loss",       valid_loss, COLORS["valid"], 1.4, 1.0),
    ("TEST (dev) loss",         test_loss,  COLORS["test"],  1.4, 1.0),
], "adam_lambda_loss_shared_step_40800_exact_style.png",
    annotate=[
        (f"Selected TEST loss = {sel_test_loss_val:.2f}\nStep {CKPT_STEP}",
         CKPT_STEP, sel_test_loss_val, COLORS["test"], (-80, 20)),
    ])
