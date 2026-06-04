"""
plot_word_scores.py
-------------------
Generates per-word net-sentiment bar charts (pos_logit - neg_logit) for every
word/logit table in the README:
  - error_analysis_mlp.png        : TP / TN / FP / FN (plain MLP)
  - error_analysis_atten.png      : TP / TN / FP / FN (MLP + Attention)
  - error_analysis_fp_fn_compare.png : FP and FN side-by-side, MLP vs MLP+Atten
  - context_review_1.png          : "Far from masterpiece" — both models
  - context_review_2.png          : "not all terrible" — both models
  - context_review_3.png          : "nothing short of brilliant" — both models
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import loader as ld

# ── Model definitions (needed for torch.load) ─────────────────────────────────

atten_size = 5

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.matrix = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(in_channels, out_channels)))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, out_channels))
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

# ── Load models ───────────────────────────────────────────────────────────────

mlp_model   = torch.load("MLP_h64.pth",            weights_only=False)
atten_model = torch.load("MLP_with_atten_h64.pth", weights_only=False)
mlp_model.eval()
atten_model.eval()

# ── Helper: get per-word net scores ──────────────────────────────────────────

def get_word_scores(model, text, max_words=25):
    """Return (words, net) where net[i] = pos_logit[i] - neg_logit[i]."""
    words = ld.tokinize(text)[:max_words]
    enc   = ld.preprocess_review(text)          # [1, 100, 100]
    with torch.no_grad():
        result    = model(enc)
        sub_score = result[0] if isinstance(result, tuple) else result  # [1, T, 2]
    scores = sub_score[0].numpy()               # [T, 2]
    net    = scores[:len(words), 0] - scores[:len(words), 1]
    return words, net

def bar_chart(ax, words, net, title, true_label, predicted, highlight=None):
    """Draw a horizontal bar chart of per-word net sentiment scores."""
    n = len(words)
    x = np.arange(n)
    bar_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in net]

    # Optionally highlight specific words in a different colour
    if highlight:
        for i, w in enumerate(words):
            if w in highlight:
                bar_colors[i] = '#9467bd'

    bars = ax.bar(x, net, color=bar_colors, edgecolor='white', linewidth=0.4, zorder=3)
    ax.axhline(0, color='black', lw=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=40, ha='right', fontsize=7.5)
    ax.set_ylabel("pos − neg logit", fontsize=8)
    ax.grid(axis='y', lw=0.4, alpha=0.4, zorder=0)

    # Value labels on the tallest ±10 bars (to avoid clutter)
    threshold = np.sort(np.abs(net))[-min(10, n)]
    for i, v in enumerate(net):
        if abs(v) >= threshold:
            ax.text(i, v + (0.5 if v >= 0 else -0.5),
                    f"{v:.1f}", ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=5.5, color='#333333')

    # Build title with prediction badge
    correct = predicted == true_label
    badge   = "✓ correct" if correct else "✗ wrong"
    badge_color = '#2ca02c' if correct else '#d62728'
    ax.set_title(title, fontsize=9, pad=4)
    ax.text(0.99, 0.97, f"pred: {predicted}  {badge}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color=badge_color,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=badge_color, lw=0.8))


def predict_from_net(net, words_n):
    return "positive" if net.mean() > 0 else "negative"


# ── Review data ───────────────────────────────────────────────────────────────

error_texts  = ld.error_analysis_texts
error_labels = ld.error_analysis_labels
error_tags   = ["TP", "TN", "FP", "FN"]
error_descs  = [
    "TP — clear positive words",
    "TN — clear negative words",
    "FP — positive surface, negative intent",
    "FN — negation chains, positive intent",
]

ctx_texts  = ld.context_sensitive_texts
ctx_labels = ld.context_sensitive_labels
ctx_descs  = [
    '"Far from the masterpiece…" (true: negative)',
    '"…not all follow-ups are terrible…" (true: positive)',
    '"…nothing short of brilliant…" (true: positive)',
]

# ── Plot 1: Error analysis — plain MLP ───────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Error Analysis — Plain MLP\nPer-word net sentiment (pos − neg logit)",
             fontsize=12, y=1.01)

for ax, tag, desc, text, true_lbl in zip(
        axes.flat, error_tags, error_descs, error_texts, error_labels):
    words, net = get_word_scores(mlp_model, text)
    pred       = predict_from_net(net, len(words))
    bar_chart(ax, words, net, f"[{tag}] {desc}", true_lbl, pred)

plt.tight_layout()
plt.savefig("error_analysis_mlp.png", dpi=150, bbox_inches='tight')
print("Saved error_analysis_mlp.png")
plt.close()

# ── Plot 2: Error analysis — MLP + Attention ─────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Error Analysis — MLP + Restricted Self-Attention\nPer-word net sentiment (pos − neg logit)",
             fontsize=12, y=1.01)

for ax, tag, desc, text, true_lbl in zip(
        axes.flat, error_tags, error_descs, error_texts, error_labels):
    words, net = get_word_scores(atten_model, text)
    pred       = predict_from_net(net, len(words))
    bar_chart(ax, words, net, f"[{tag}] {desc}", true_lbl, pred)

plt.tight_layout()
plt.savefig("error_analysis_atten.png", dpi=150, bbox_inches='tight')
print("Saved error_analysis_atten.png")
plt.close()

# ── Plot 3: FP and FN side-by-side, MLP vs MLP+Attention ─────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("FP and FN Cases: Plain MLP vs MLP + Restricted Self-Attention",
             fontsize=12, y=1.01)

for row, (tag, text, true_lbl) in enumerate(
        zip(["FP", "FN"], error_texts[2:], error_labels[2:])):
    # MLP
    words_m, net_m = get_word_scores(mlp_model, text)
    pred_m = predict_from_net(net_m, len(words_m))
    bar_chart(axes[row][0], words_m, net_m,
              f"[{tag}] Plain MLP", true_lbl, pred_m)

    # MLP + Attention
    words_a, net_a = get_word_scores(atten_model, text)
    pred_a = predict_from_net(net_a, len(words_a))
    bar_chart(axes[row][1], words_a, net_a,
              f"[{tag}] MLP + Attention", true_lbl, pred_a)

plt.tight_layout()
plt.savefig("error_analysis_fp_fn_compare.png", dpi=150, bbox_inches='tight')
print("Saved error_analysis_fp_fn_compare.png")
plt.close()

# ── Plots 4-6: Context-sensitive reviews, each on its own figure ──────────────

highlight_map = [
    {"masterpiece", "disappointment", "failed"},
    {"terrible", "brilliantly", "satisfying"},
    {"brilliant", "extraordinary", "nothing", "director", "is"},
]

for i, (text, true_lbl, desc, hl) in enumerate(
        zip(ctx_texts, ctx_labels, ctx_descs, highlight_map), start=1):

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    fig.suptitle(f"Context-Sensitive Review {i}: {desc}", fontsize=11, y=1.02)

    words_m, net_m = get_word_scores(mlp_model,   text)
    words_a, net_a = get_word_scores(atten_model, text)
    pred_m = predict_from_net(net_m, len(words_m))
    pred_a = predict_from_net(net_a, len(words_a))

    bar_chart(axes[0], words_m, net_m, "Plain MLP",            true_lbl, pred_m, highlight=hl)
    bar_chart(axes[1], words_a, net_a, "MLP + Restricted Self-Attention", true_lbl, pred_a, highlight=hl)

    # Shared y-axis range so bars are directly comparable
    all_net = np.concatenate([net_m, net_a])
    ymax = max(abs(all_net.min()), abs(all_net.max())) * 1.15
    for ax in axes:
        ax.set_ylim(-ymax, ymax)

    plt.tight_layout()
    fname = f"context_review_{i}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close()

print("\nAll plots generated.")
