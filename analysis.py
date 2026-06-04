"""
analysis.py
-----------
1. Re-generates mlp_comparison.png with a clean, readable style.
2. Investigates and visualises the "director is" spurious association
   in the trained MLP + Restricted Self-Attention model.
"""

import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import loader as ld

# ── Model class definitions needed for torch.load ────────────────────────────

atten_size = 5  # must match training configuration

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.matrix = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
            requires_grad=True)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)
        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ReLU = nn.ReLU()
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self): return "MLP"

    def forward(self, x):
        x = self.ReLU(self.layer1(x))
        x = self.ReLU(self.layer2(x))
        return self.layer3(x)


class ExMLPWithAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        from torch.nn.functional import pad as F_pad
        self.hidden_size = hidden_size
        self.atten_size = atten_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(2)
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        n_pos = 2 * atten_size + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, n_pos, hidden_size) * 0.01)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self): return "MLP_with_atten"

    def forward(self, x):
        from torch.nn.functional import pad as F_pad
        as_ = self.atten_size
        x = self.ReLU(self.layer1(x))
        x = self.ReLU(self.layer2(x))
        padded = F_pad(x, (0, 0, as_, as_, 0, 0))
        x_nei = torch.stack(
            [torch.roll(padded, k, 1) for k in range(-as_, as_ + 1)], dim=2
        )[:, as_:-as_, :]
        x_nei = x_nei + self.pos_encoding
        query = self.W_q(x)
        keys  = self.W_k(x_nei)
        vals  = self.W_v(x_nei)
        scores = (query.unsqueeze(2) * keys).sum(-1) / self.sqrt_hidden_size
        atten_weights = self.softmax(scores)
        context = (atten_weights.unsqueeze(-1) * vals).sum(2)
        return self.layer3(context), atten_weights

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Parse training log → reconstruct MLP histories
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE = "training_output.log"

def parse_mlp_histories(log_path):
    """Return histories for MLP and MLP+atten from the training log."""
    histories = {}
    current_key = None
    pat = re.compile(
        r"Epoch \[(\d+)/\d+\] Step\s+(\d+)"
        r" \| Train Loss (\S+) Acc (\S+)"
        r" \| Test\s+Loss (\S+) Acc (\S+)"
    )
    with open(log_path) as f:
        for line in f:
            line = line.rstrip()
            if "=== Training MLP_h64 ===" in line:
                current_key = "MLP"
                histories[current_key] = {k: [] for k in ["steps","train_loss","test_loss","train_acc","test_acc"]}
            elif "=== Training MLP_with_atten_h64 ===" in line:
                current_key = "MLP+atten"
                histories[current_key] = {k: [] for k in ["steps","train_loss","test_loss","train_acc","test_acc"]}
            elif current_key and line.startswith("Epoch"):
                m = pat.match(line)
                if m:
                    step = int(m.group(2))
                    # Only record MLP / MLP+atten sections
                    histories[current_key]["steps"].append(step)
                    histories[current_key]["train_loss"].append(float(m.group(3)))
                    histories[current_key]["train_acc"].append(float(m.group(4)))
                    histories[current_key]["test_loss"].append(float(m.group(5)))
                    histories[current_key]["test_acc"].append(float(m.group(6)))
    return histories

def smooth(values, w=15):
    """Uniform moving average of width w."""
    kernel = np.ones(w) / w
    padded = np.pad(values, (w//2, w//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]

histories = parse_mlp_histories(LOG_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Regenerate mlp_comparison.png — clean style
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"MLP": "#1f77b4", "MLP+atten": "#d62728"}  # blue / red

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Task 2 & 4 — MLP vs MLP + Restricted Self-Attention", fontsize=12, y=1.01)

for key, color in COLORS.items():
    hist = histories[key]
    steps = np.array(hist["steps"])
    tr_loss = np.array(hist["train_loss"])
    te_loss = np.array(hist["test_loss"])
    tr_acc  = np.array(hist["train_acc"])
    te_acc  = np.array(hist["test_acc"])

    ls = "-" if key == "MLP" else "--"

    # Raw noisy curves (very transparent)
    axes[0].plot(steps, tr_loss, color=color, lw=0.6, alpha=0.12, ls=ls)
    axes[0].plot(steps, te_loss, color=color, lw=0.6, alpha=0.12, ls=ls)
    # Smoothed curves (train lighter, test full)
    axes[0].plot(steps, smooth(tr_loss), color=color, lw=1.2, alpha=0.45, ls=ls)
    axes[0].plot(steps, smooth(te_loss), color=color, lw=2.2, alpha=1.00, ls=ls,
                 label=f"{key} (test)")

    axes[1].plot(steps, tr_acc,  color=color, lw=0.6, alpha=0.12, ls=ls)
    axes[1].plot(steps, te_acc,  color=color, lw=0.6, alpha=0.12, ls=ls)
    axes[1].plot(steps, smooth(tr_acc),  color=color, lw=1.2, alpha=0.45, ls=ls)
    axes[1].plot(steps, smooth(te_acc),  color=color, lw=2.2, alpha=1.00, ls=ls,
                 label=f"{key} (test)")

axes[0].set_title("Cross-Entropy Loss")
axes[0].set_xlabel("Training Step")
axes[0].set_ylabel("Loss")
axes[0].legend(loc="upper right")
axes[0].set_ylim(bottom=0)
# Annotate final test values
for key, color in COLORS.items():
    final_loss = histories[key]["test_loss"][-1]
    final_step = histories[key]["steps"][-1]
    axes[0].annotate(f"{final_loss:.3f}", xy=(final_step, final_loss),
                     xytext=(-40, 8), textcoords="offset points",
                     color=color, fontsize=8, arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Training Step")
axes[1].set_ylabel("Accuracy")
axes[1].legend(loc="lower right")
axes[1].set_ylim(0.4, 1.0)
for key, color in COLORS.items():
    final_acc = histories[key]["test_acc"][-1]
    final_step = histories[key]["steps"][-1]
    axes[1].annotate(f"{final_acc:.3f}", xy=(final_step, final_acc),
                     xytext=(-40, -14), textcoords="offset points",
                     color=color, fontsize=8, arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

# Legend for train vs test line weight
from matplotlib.lines import Line2D
legend_extras = [
    Line2D([0], [0], color='gray', lw=2.2, alpha=1.0,  label='test (bold)'),
    Line2D([0], [0], color='gray', lw=1.2, alpha=0.45, label='train (faint)'),
]
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_extras, fontsize=8)

plt.tight_layout()
plt.savefig("mlp_comparison.png", dpi=150, bbox_inches='tight')
print("Saved mlp_comparison.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  "director is" investigation
# ─────────────────────────────────────────────────────────────────────────────

atten_model = torch.load("MLP_with_atten_h64.pth", weights_only=False)
atten_model.eval()

def score_review(model, text):
    """Return per-word logits [T, 2] and word list for a review string."""
    words  = ld.tokinize(text)
    enc    = ld.preprocess_review(text)          # [1, 100, 100]
    with torch.no_grad():
        result   = model(enc)
        sub_score = result[0] if isinstance(result, tuple) else result  # [1, T, 2]
    return words, sub_score[0].numpy()           # [T, 2]

# ── 3a. Probe sentences: same target word, varied context ────────────────────

probes = [
    # (description, text, target_word, expected_sentiment)
    ("director is talented",
     "The director is talented and brought a fresh vision to the film.",
     "director", "positive"),
    ("director is brilliant",
     "The director is brilliant, crafting a masterpiece of modern cinema.",
     "director", "positive"),
    ("director is awful",
     "The director is awful and ruined every scene with poor choices.",
     "director", "negative"),
    ("director is terrible",
     "The director is terrible, delivering one of the worst films of the year.",
     "director", "negative"),
    ("director is nothing short of brilliant",
     "The director's vision is nothing short of brilliant and the film excels.",
     "director", "positive"),
    # Isolated baselines — word alone without "director"
    ("brilliant alone",
     "brilliant stunning wonderful magnificent amazing perfect excellent superb flawless gorgeous",
     "brilliant", "positive"),
    ("awful alone",
     "awful terrible horrible disgusting dreadful appalling atrocious abysmal dull boring",
     "awful", "negative"),
    # Control: "director" without "is"
    ("director without is",
     "The director crafted a masterpiece of stunning beauty and technical precision.",
     "director", "positive"),
    ("director in negative review",
     "The director failed completely with a confusing mess of a film.",
     "director", "negative"),
]

def find_word_score(words, scores, target):
    """Return pos_logit, neg_logit for first occurrence of target word."""
    for i, w in enumerate(words):
        if w == target:
            return float(scores[i, 0]), float(scores[i, 1])
    return None, None

results = []
for desc, text, target, expected in probes:
    words, scores = score_review(atten_model, text)
    pos, neg = find_word_score(words, scores, target)
    pred = "positive" if (pos is not None and pos > neg) else "negative"
    results.append((desc, target, expected, pos, neg, pred))
    print(f"{desc:<45} | target='{target}' | pos={pos:.2f} neg={neg:.2f} | pred={pred} expected={expected}")

# ── 3b. Visualise: bar chart of pos-logit for "director" across contexts ─────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MLP+Attention: Spurious 'director is' Association", fontsize=13)

# Left: bar chart — pos logit for target word across probe sentences
labels_bar = [r[0] for r in results]
pos_logits  = [r[3] if r[3] is not None else 0 for r in results]
neg_logits  = [r[4] if r[4] is not None else 0 for r in results]
net         = [p - n for p, n in zip(pos_logits, neg_logits)]  # positive = model says positive

colors_bar = ['#2ca02c' if v >= 0 else '#d62728' for v in net]
y_pos = np.arange(len(labels_bar))

axes[0].barh(y_pos, net, color=colors_bar, edgecolor='black', linewidth=0.5)
axes[0].axvline(0, color='black', lw=1.0)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(labels_bar, fontsize=8)
axes[0].set_xlabel("pos_logit − neg_logit  (green=positive prediction, red=negative)")
axes[0].set_title("Target word net sentiment score\nacross probe sentences")
for i, v in enumerate(net):
    axes[0].text(v + (0.3 if v >= 0 else -0.3), i,
                 f"{v:+.2f}", va='center', ha='left' if v >= 0 else 'right',
                 fontsize=7, color='black')

# Right: per-word logit heatmap for the "nothing short of brilliant" review ───
text_cs = "The director's vision is nothing short of brilliant, and the performances are nothing less than extraordinary."
words_cs, scores_cs = score_review(atten_model, text_cs)
n = min(len(words_cs), 16)
words_cs  = words_cs[:n]
net_cs    = scores_cs[:n, 0] - scores_cs[:n, 1]  # pos − neg per word

bar_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in net_cs]
x_pos = np.arange(n)
axes[1].bar(x_pos, net_cs, color=bar_colors, edgecolor='black', linewidth=0.4)
axes[1].axhline(0, color='black', lw=1.0)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(words_cs, rotation=35, ha='right', fontsize=8)
axes[1].set_ylabel("pos_logit − neg_logit")
axes[1].set_title("Per-word scores: \"The director's vision is nothing short of brilliant…\"\n(MLP+Attention — model predicts NEGATIVE despite positive content)")

# Highlight the "director" and "is" bars
for i, w in enumerate(words_cs):
    if w in ("director", "is"):
        axes[1].bar(x_pos[i], net_cs[i], color='#9467bd', edgecolor='black',
                    linewidth=1.5, label=f"'{w}' (spurious negative)")

# Add value labels
for i, v in enumerate(net_cs):
    axes[1].text(i, v + (0.4 if v >= 0 else -0.8), f"{v:.1f}",
                 ha='center', va='bottom' if v >= 0 else 'top',
                 fontsize=6.5, color='black')

# Remove duplicate legend entries
handles, labels = axes[1].get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
axes[1].legend(seen.values(), seen.keys(), fontsize=8)

plt.tight_layout()
plt.savefig("director_is_analysis.png", dpi=150, bbox_inches='tight')
print("Saved director_is_analysis.png")
plt.close()

# ── 3c. Additional: show how "director" score changes with/without "is" context

print("\n=== 'director' logit across contexts ===")
print(f"{'Context':<50} {'pos':>7} {'neg':>7} {'net':>7}")
print("-" * 70)
for desc, target, expected, pos, neg, pred in results:
    net_val = (pos - neg) if pos is not None else float('nan')
    print(f"{desc:<50} {pos:>7.2f} {neg:>7.2f} {net_val:>7.2f}   pred={pred} expected={expected}")
