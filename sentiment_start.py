########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld


batch_size = 32
output_size = 2
hidden_size = 64        # to experiment with

run_recurrent = True    # else run Token-wise MLP
use_RNN = True          # otherwise GRU
atten_size = 5          # atten > 0 means using restricted self atten
run_recurrent_comparison = False  # set True to train all 4 RNN/GRU configs
run_mlp = True          # set True to train MLP experiment
use_mlp_atten = True    # True = MLP + restricted self-attention; False = plain MLP only
run_attention = False   # set True to train standalone attention model (Task 3)

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x+ self.bias 
        return x
        
# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # MLP for final sentiment prediction from hidden state
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)
        hidden = torch.tanh(self.in2hidden(combined))
        output = self.hidden2output(hidden)
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)

# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        # GRU Cell weights
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # MLP for final sentiment prediction from hidden state
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        combined = torch.cat((x, hidden_state), dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        n = torch.tanh(self.new_gate(torch.cat((x, r * hidden_state), dim=1)))
        hidden = (1 - z) * hidden_state + z * n
        output = self.hidden2output(hidden)
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.hidden_size = hidden_size
        self.ReLU = torch.nn.ReLU()

        # Token-wise MLP: input → hidden → hidden → output_size
        # MatMul operates on the last two axes, so it applies identically to
        # every word position: shape [batch, num_words, dim] is preserved.
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP"

    def forward(self, x):
        # x: [batch, num_words, input_size]
        # Each word is projected independently → [batch, num_words, output_size]
        x = self.ReLU(self.layer1(x))
        x = self.ReLU(self.layer2(x))
        x = self.layer3(x)
        return x


class ExMLPWithAtten(nn.Module):
    """Task 2 MLP (input→hidden→hidden) with restricted self-attention inserted
    before the final output projection, instead of going straight to logits."""

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.atten_size = atten_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(2)   # over neighbor axis

        # Same two hidden layers as ExMLP
        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, hidden_size)

        # Restricted self-attention (single head, learnable Q/K/V)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        n_pos = 2 * atten_size + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, n_pos, hidden_size) * 0.01)

        # Final output projection (same role as ExMLP layer3)
        self.layer3 = MatMul(hidden_size, output_size)

    def name(self):
        return "MLP_with_atten"

    def forward(self, x):
        as_ = self.atten_size

        # ── MLP body (Task 2 layers 1 & 2) ───────────────────────────────────
        x = self.ReLU(self.layer1(x))    # [B, T, H]
        x = self.ReLU(self.layer2(x))    # [B, T, H]

        # ── Restricted self-attention ─────────────────────────────────────────
        padded = pad(x, (0, 0, as_, as_, 0, 0))   # [B, T+2*as, H]
        x_nei = torch.stack(
            [torch.roll(padded, k, 1) for k in range(-as_, as_ + 1)], dim=2
        )[:, as_:-as_, :]                          # [B, T, 2*as+1, H]
        x_nei = x_nei + self.pos_encoding          # add relative positional bias

        query = self.W_q(x)                        # [B, T, H]
        keys  = self.W_k(x_nei)                    # [B, T, 2*as+1, H]
        vals  = self.W_v(x_nei)                    # [B, T, 2*as+1, H]

        scores       = (query.unsqueeze(2) * keys).sum(-1) / self.sqrt_hidden_size
        atten_weights = self.softmax(scores)        # [B, T, 2*as+1]
        context      = (atten_weights.unsqueeze(-1) * vals).sum(2)  # [B, T, H]

        # ── Output logits ─────────────────────────────────────────────────────
        return self.layer3(context), atten_weights  # [B, T, output_size]


class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.atten_size = atten_size          # freeze window size at construction
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)    # softmax over the neighbor axis

        # Token-wise projection (shared across all word positions)
        self.layer1 = MatMul(input_size, hidden_size)

        # Single-head attention: learnable Q / K / V projections (no bias, standard practice)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        # Output projection: context vector → per-word logits
        self.layer2 = MatMul(hidden_size, output_size)

        # Learnable relative positional encoding.
        # Shape [1, 1, 2*atten_size+1, hidden_size] broadcasts over (batch, time).
        # Initialized near zero so attention starts approximately uniform.
        n_pos = 2 * atten_size + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, n_pos, hidden_size) * 0.01)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # x: [B, T, input_size]
        as_ = self.atten_size

        # 1. Token-wise MLP projection
        x = self.ReLU(self.layer1(x))         # [B, T, H]

        # 2. Build neighbor window using roll + padding.
        #    Pad T axis by atten_size on each side so rolled rows wrap into zeros.
        padded = pad(x, (0, 0, as_, as_, 0, 0))   # [B, T+2*as, H]

        x_nei = []
        for k in range(-as_, as_ + 1):
            x_nei.append(torch.roll(padded, k, 1))

        # Stack along a new neighbor axis, then strip the padding rows
        x_nei = torch.stack(x_nei, 2)             # [B, T+2*as, 2*as+1, H]
        x_nei = x_nei[:, as_:-as_, :]             # [B, T,      2*as+1, H]

        # 3. Add learnable relative positional encoding to neighbor representations
        x_nei = x_nei + self.pos_encoding          # broadcast → [B, T, 2*as+1, H]

        # 4. Compute Q / K / V
        #    W_q, W_k, W_v are MatMul layers; MatMul uses torch.matmul which broadcasts
        #    over all leading dimensions, so it works on both [B,T,H] and [B,T,N,H].
        query = self.W_q(x)                        # [B, T, H]
        keys  = self.W_k(x_nei)                    # [B, T, 2*as+1, H]
        vals  = self.W_v(x_nei)                    # [B, T, 2*as+1, H]

        # 5. Scaled dot-product attention scores
        #    Expand query to [B, T, 1, H] and element-wise multiply with keys, then sum H
        scores = (query.unsqueeze(2) * keys).sum(dim=-1) / self.sqrt_hidden_size
        #    scores: [B, T, 2*as+1]

        # 6. Softmax over the neighbor dimension → attention weights
        atten_weights = self.softmax(scores)       # [B, T, 2*as+1]

        # 7. Weighted sum of values → context vector
        context = (atten_weights.unsqueeze(-1) * vals).sum(dim=2)  # [B, T, H]

        # 8. Per-word output logits
        x_out = self.layer2(context)               # [B, T, output_size]

        return x_out, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    """Print per-word sub-scores, the averaged logit, softmax, and the true label.

    Softmax is applied AFTER averaging the logits (not before), because:
      softmax(mean(logits)) != mean(softmax(logits))
    Averaging raw logits preserves the linear scale of confidence so that a
    highly confident word dominates proportionally; applying softmax first
    compresses all scores to [0,1] and distorts that balance (a word with
    logit=10 and one with logit=0 would get 1.0 and 0.5 respectively, yet
    their average probability 0.75 no longer reflects the true logit gap).
    CrossEntropyLoss also expects raw logits, so we stay consistent.
    """
    # Show first 20-30 real (non-padding) words
    words = list(rev_text)[:30]
    n = len(words)

    print(f"\n{'Word':<22} {'Pos logit':>10} {'Neg logit':>10}")
    print("-" * 44)
    for word, s1, s2 in zip(words, sbs1[:n], sbs2[:n]):
        print(f"{word:<22} {s1:>10.4f} {s2:>10.4f}")
    print("-" * 44)

    # Average logits over real words, then softmax
    avg_pos = float(np.mean(sbs1[:n]))
    avg_neg = float(np.mean(sbs2[:n]))
    exp_p, exp_n = np.exp(avg_pos), np.exp(avg_neg)
    prob_pos = exp_p / (exp_p + exp_n)
    prob_neg = exp_n / (exp_p + exp_n)

    pred  = "positive" if avg_pos > avg_neg else "negative"
    true  = "positive" if lbl1 > lbl2 else "negative"
    print(f"Avg logits : pos={avg_pos:.4f}  neg={avg_neg:.4f}")
    print(f"Softmax    : pos={prob_pos:.4f}  neg={prob_neg:.4f}")
    print(f"Predicted  : {pred:<10}  True: {true}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()


def train_model(model):
    label = f"{model.name()}_h{model.hidden_size}"
    print(f"\n=== Training {label} ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = 1.0
    test_loss  = 1.0
    train_acc  = 0.5
    test_acc   = 0.5

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "steps": []}
    step = 0
    test_iterator = iter(test_dataset)

    for epoch in range(num_epochs):
        for labels, reviews, reviews_text in train_dataset:
            step += 1

            if step % test_interval == 0:
                test_iter = True
                try:
                    labels, reviews, reviews_text = next(test_iterator)
                except StopIteration:
                    test_iterator = iter(test_dataset)
                    labels, reviews, reviews_text = next(test_iterator)
            else:
                test_iter = False

            target_labels = torch.argmax(labels, dim=1) if labels.dim() == 2 else labels

            hidden_state = model.init_hidden(int(labels.shape[0]))
            for i in range(num_words):
                output, hidden_state = model(reviews[:, i, :], hidden_state)

            loss = criterion(output, target_labels)
            acc  = (output.argmax(dim=1) == target_labels).float().mean().item()

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
                train_acc  = 0.9 * acc + 0.1 * train_acc
            else:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                test_acc  = 0.8 * acc + 0.2 * test_acc
                history["train_loss"].append(train_loss)
                history["test_loss"].append(test_loss)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["steps"].append(step)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Step {step:5d} | "
                    f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
                    f"Test  Loss {test_loss:.4f} Acc {test_acc:.3f}"
                )

    torch.save(model, label + ".pth")
    return history


def train_mlp(model):
    label = f"{model.name()}_h{model.hidden_size}"
    print(f"\n=== Training {label} ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = 1.0
    test_loss  = 1.0
    train_acc  = 0.5
    test_acc   = 0.5

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "steps": []}
    step = 0
    test_iterator = iter(test_dataset)

    for epoch in range(num_epochs):
        for labels, reviews, reviews_text in train_dataset:
            step += 1

            if step % test_interval == 0:
                test_iter = True
                try:
                    labels, reviews, reviews_text = next(test_iterator)
                except StopIteration:
                    test_iterator = iter(test_dataset)
                    labels, reviews, reviews_text = next(test_iterator)
            else:
                test_iter = False

            target_labels = torch.argmax(labels, dim=1) if labels.dim() == 2 else labels

            result = model(reviews)
            sub_score = result[0] if isinstance(result, tuple) else result  # [B, T, 2]
            mask = (reviews.abs().sum(dim=-1) > 0).float()         # [B, T]
            masked = sub_score * mask.unsqueeze(-1)
            output = masked.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)  # [B, 2]

            loss = criterion(output, target_labels)
            acc  = (output.argmax(dim=1) == target_labels).float().mean().item()

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
                train_acc  = 0.9 * acc + 0.1 * train_acc
            else:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                test_acc  = 0.8 * acc + 0.2 * test_acc
                history["train_loss"].append(train_loss)
                history["test_loss"].append(test_loss)
                history["train_acc"].append(train_acc)
                history["test_acc"].append(test_acc)
                history["steps"].append(step)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] Step {step:5d} | "
                    f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
                    f"Test  Loss {test_loss:.4f} Acc {test_acc:.3f}"
                )
                nump_subs = sub_score.detach().numpy()
                lbl_np    = labels.detach().numpy()
                print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1],
                             lbl_np[0, 0], lbl_np[0, 1])

    torch.save(model, label + ".pth")
    return history


if run_recurrent_comparison:
    # ── Run all four configurations ───────────────────────────────────────────

    configs = [
        ("RNN", 64),
        ("RNN", 128),
        ("GRU", 64),
        ("GRU", 128),
    ]

    all_histories = {}
    for arch, hs in configs:
        if arch == "RNN":
            model = ExRNN(input_size, output_size, hs)
        else:
            model = ExGRU(input_size, output_size, hs)
        key = f"{arch}_h{hs}"
        all_histories[key] = train_model(model)

    # ── Visualise results ─────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    styles = {
        "RNN_h64":  ("blue",   "-"),
        "RNN_h128": ("blue",   "--"),
        "GRU_h64":  ("orange", "-"),
        "GRU_h128": ("orange", "--"),
    }

    for key, hist in all_histories.items():
        color, ls = styles[key]
        steps = hist["steps"]
        axes[0].plot(steps, hist["train_loss"], color=color, linestyle=ls, alpha=0.4)
        axes[0].plot(steps, hist["test_loss"],  color=color, linestyle=ls, label=key)
        axes[1].plot(steps, hist["train_acc"],  color=color, linestyle=ls, alpha=0.4)
        axes[1].plot(steps, hist["test_acc"],   color=color, linestyle=ls, label=key)

    axes[0].set_title("Loss (solid=test, faded=train)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].set_title("Accuracy (solid=test, faded=train)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("rnn_gru_comparison.png", dpi=150)
    print("\nPlot saved to rnn_gru_comparison.png")

    # ── Evaluate on custom test reviews ──────────────────────────────────────

    DESCRIPTIONS = [
        "short positive (baseline)",
        "short negative (baseline)",
        "negation at START → positive",
        "negation at START → negative",
        "negation at END   → negative",
        "late flip         → positive",
        "sentiment at START, neutral filler",
        "sentiment in MIDDLE, neutral around",
        "sentiment at END only",
        "long positive",
        "long negative",
        "very short positive",
        "very short negative",
        "sarcasm (surface=positive, true=negative)",
    ]

    labels_str = [
        "positive", "negative",
        "positive", "negative",
        "negative", "positive",
        "positive", "negative",
        "positive", "positive",
        "negative", "positive",
        "negative", "negative",
    ]

    print("\n" + "="*90)
    print(f"{'Review description':<44} {'True':^8}", end="")
    for key in all_histories:
        print(f" {key:^12}", end="")
    print()
    print("="*90)

    trained_models = {}
    for arch, hs in configs:
        key = f"{arch}_h{hs}"
        if arch == "RNN":
            m = ExRNN(input_size, output_size, hs)
        else:
            m = ExGRU(input_size, output_size, hs)
        m = torch.load(key + ".pth", weights_only=False)
        m.eval()
        trained_models[key] = m

    for i, (text, true_label, desc) in enumerate(zip(ld.my_test_texts, labels_str, DESCRIPTIONS)):
        encoded = ld.preprocess_review(text)
        print(f"{desc:<44} {true_label:^8}", end="")
        for key, m in trained_models.items():
            with torch.no_grad():
                h = m.init_hidden(1)
                for t in range(encoded.shape[1]):
                    out, h = m(encoded[:, t, :], h)
                pred = "positive" if out.argmax(dim=1).item() == 0 else "negative"
                correct = "✓" if pred == true_label else "✗"
                print(f" {pred[:3]}{correct:>9}", end="")
        print()

    print("="*90)


def run_error_and_context_analysis(model, model_label):
    """Run TP/TN/FP/FN error analysis and context-sensitive review for a model."""

    def eval_review(text, true_label):
        encoded = ld.preprocess_review(text)
        words   = ld.tokinize(text)
        with torch.no_grad():
            result    = model(encoded)
            sub_score = result[0] if isinstance(result, tuple) else result
            mask      = (encoded.abs().sum(dim=-1) > 0).float()
            masked    = sub_score * mask.unsqueeze(-1)
            output    = masked.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
            pred      = "positive" if output.argmax(dim=1).item() == 0 else "negative"
        lbl1 = 1.0 if true_label == "positive" else 0.0
        return pred, sub_score[0].detach().numpy(), words, lbl1

    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS — {model_label}")
    print(f"{'='*60}")
    for tag, text, true_label in zip(
        ["TP", "TN", "FP", "FN"], ld.error_analysis_texts, ld.error_analysis_labels
    ):
        pred, sbs, words, lbl1 = eval_review(text, true_label)
        outcome = "correct" if pred == true_label else "WRONG"
        print(f"\n[{tag}] true={true_label}, predicted={pred}  ({outcome})")
        print_review(words, sbs[:, 0], sbs[:, 1], lbl1, 1.0 - lbl1)

    print(f"\n{'='*60}")
    print(f"CONTEXT-SENSITIVE REVIEWS — {model_label}")
    print(f"{'='*60}")
    for text, true_label in zip(ld.context_sensitive_texts, ld.context_sensitive_labels):
        pred, sbs, words, lbl1 = eval_review(text, true_label)
        outcome = "correct" if pred == true_label else "WRONG"
        print(f"\ntrue={true_label}, predicted={pred}  ({outcome})")
        print_review(words, sbs[:, 0], sbs[:, 1], lbl1, 1.0 - lbl1)


if run_mlp:
    # ── Train plain MLP (Task 2) ──────────────────────────────────────────────
    mlp_plain   = ExMLP(input_size, output_size, hidden_size)
    hist_plain  = train_mlp(mlp_plain)
    mlp_plain.eval()

    if use_mlp_atten:
        # ── Train MLP + restricted self-attention ─────────────────────────────
        mlp_atten  = ExMLPWithAtten(input_size, output_size, hidden_size)
        hist_atten = train_mlp(mlp_atten)
        mlp_atten.eval()

    # ── Loss / accuracy plot ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for hist, lbl, ls in [(hist_plain, "MLP", "-")] + (
        [(hist_atten, "MLP+atten", "--")] if use_mlp_atten else []
    ):
        steps = hist["steps"]
        axes2[0].plot(steps, hist["train_loss"], ls=ls, alpha=0.4)
        axes2[0].plot(steps, hist["test_loss"],  ls=ls, label=lbl)
        axes2[1].plot(steps, hist["train_acc"],  ls=ls, alpha=0.4)
        axes2[1].plot(steps, hist["test_acc"],   ls=ls, label=lbl)
    axes2[0].set_title("Loss (solid=test, faded=train)")
    axes2[0].set_xlabel("Step"); axes2[0].set_ylabel("Cross-entropy loss"); axes2[0].legend()
    axes2[1].set_title("Accuracy (solid=test, faded=train)")
    axes2[1].set_xlabel("Step"); axes2[1].set_ylabel("Accuracy"); axes2[1].legend()
    plt.tight_layout()
    plt.savefig("mlp_comparison.png", dpi=150)
    print("Plot saved to mlp_comparison.png")

    # ── Error analysis + context reviews ─────────────────────────────────────
    run_error_and_context_analysis(mlp_plain, "Plain MLP")
    if use_mlp_atten:
        run_error_and_context_analysis(mlp_atten, "MLP + Restricted Self-Attention")


if run_attention:
    # ── Restricted self-attention experiment ──────────────────────────────────

    atten_model = ExLRestSelfAtten(input_size, output_size, hidden_size)
    atten_history = train_mlp(atten_model)   # same token-wise training loop
    atten_model.eval()

    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    steps = atten_history["steps"]
    axes3[0].plot(steps, atten_history["train_loss"], alpha=0.4, label="train")
    axes3[0].plot(steps, atten_history["test_loss"],            label="test")
    axes3[0].set_title("MLP + Restricted Self-Attention  Loss")
    axes3[0].set_xlabel("Step"); axes3[0].set_ylabel("Cross-entropy loss")
    axes3[0].legend()
    axes3[1].plot(steps, atten_history["train_acc"], alpha=0.4, label="train")
    axes3[1].plot(steps, atten_history["test_acc"],             label="test")
    axes3[1].set_title("MLP + Restricted Self-Attention  Accuracy")
    axes3[1].set_xlabel("Step"); axes3[1].set_ylabel("Accuracy")
    axes3[1].legend()
    plt.tight_layout()
    plt.savefig("atten_training.png", dpi=150)
    print("Attention plot saved to atten_training.png")