import torch
# from torchtext.legacy.data import Field
import torchtext as tx
from torchtext.vocab import GloVe
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
import re
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100
embedding_size = 100
Train_size=30000



def review_clean(text):
    text = re.sub(r'[^A-Za-z]+', ' ', text)  # remove non alphabetic character
    text = re.sub(r'https?:/\/\S+', ' ', text)  # remove links
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # remove singale char
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokinize(s):
    s = review_clean(s).lower()
    splited = s.split()
    return splited[:MAX_LENGTH]


def load_data_set(load_my_reviews=False):
    data=pd.read_csv("IMDB Dataset.csv")
    train_data=data[:Train_size]
    train_iter=ReviewDataset(train_data["review"],train_data["sentiment"])
    test_data=data[Train_size:]
    if load_my_reviews:
        my_data = pd.DataFrame({"review": my_test_texts, "sentiment": my_test_labels})
        test_data=test_data.append(my_data)
    test_data=test_data.reset_index(drop=True)
    test_iter=ReviewDataset(test_data["review"],test_data["sentiment"])
    return train_iter, test_iter


embadding = GloVe(name='6B', dim=embedding_size)
tokenizer = get_tokenizer(tokenizer=tokinize)


def preprocess_review(s):
    cleaned = tokinize(s)
    embadded = embadding.get_vecs_by_tokens(cleaned)
    if embadded.shape[0] != 100 or embadded.shape[1] != 100:
        embadded = torch.nn.functional.pad(embadded, (0, 0, 0, MAX_LENGTH - embadded.shape[0]))
    return torch.unsqueeze(embadded, 0)


def preprocess_label(label):
    return [0.0, 1.0] if label == "negative" else [1.0, 0.0]


def collact_batch(batch):
    label_list = []
    review_list = []
    embadding_list=[]
    for  review,label in batch:
        label_list.append(preprocess_label(label))### label
        review_list.append(tokinize(review))### the  actuall review
        processed_review = preprocess_review(review).detach()
        embadding_list.append(processed_review) ### the embedding vectors
    label_list = torch.tensor(label_list, dtype=torch.float32).reshape((-1, 2))
    embadding_tensor= torch.cat(embadding_list)
    return label_list.to(device), embadding_tensor.to(device) ,review_list


##########################
# ADD YOUR OWN TEST TEXT #
##########################

# Labels: "negative" → negative class; anything else (e.g. "positive") → positive class.
# Each review is designed to probe a specific model capability.

my_test_texts = []
my_test_labels = []

# --- Baseline: short, unambiguous ---

my_test_texts.append(
    "This film is an absolute masterpiece. Brilliant acting, stunning visuals, and a deeply moving story."
)
my_test_labels.append("positive")   # expected: positive

my_test_texts.append(
    "Terrible in every way. Dull plot, wooden acting, and a waste of two hours."
)
my_test_labels.append("negative")   # expected: negative

# --- Negation at the START ---
# The negation word appears first; the model must carry it through the rest of the sentence.
# RNN often loses it; GRU retains it via the update gate.

my_test_texts.append(
    "Not once did this film disappoint me. Every scene was gripping, every performance felt real, "
    "and the ending left me breathless with admiration."
)
my_test_labels.append("positive")   # double-negative → positive; tricky for RNN

my_test_texts.append(
    "Not good, not entertaining, not even remotely watchable. "
    "The screenplay was a mess, the direction was listless, and the cast seemed utterly disengaged."
)
my_test_labels.append("negative")   # negation chain at start

# --- Negation at the END ---
# Positive-sounding content fills most of the review; the flip comes only at the very end.
# RNN hidden state is dominated by positive tokens; GRU can still update late.

my_test_texts.append(
    "Breathtaking cinematography. A powerful ensemble cast. A score that moved me to tears. "
    "Every frame was composed with care and the screenplay crackled with intelligence. "
    "I absolutely did not enjoy a single moment of this pretentious, hollow spectacle."
)
my_test_labels.append("negative")   # late negation flip; hard for RNN

my_test_texts.append(
    "The first act dragged, the second act confused me, and the characters felt like cardboard cutouts. "
    "I nearly walked out halfway through. But the final thirty minutes completely redeemed it — "
    "a tour-de-force conclusion that made the whole painful journey worthwhile. I loved it."
)
my_test_labels.append("positive")   # late positive flip

# --- Key information at the START, neutral filler afterwards ---
# The decisive sentiment word is in the first few tokens; the rest is neutral plot summary.
# Both models should succeed, but RNN may drift on long padding.

my_test_texts.append(
    "Magnificent. The story follows a young soldier who returns home after the war and struggles "
    "to reconnect with his family. He visits old friends, walks through familiar streets, "
    "and slowly rebuilds his life over the course of several months."
)
my_test_labels.append("positive")   # sentiment at position 0

# --- Key information in the MIDDLE ---
# Neutral opening, decisive word in the middle, neutral closing.

my_test_texts.append(
    "The film opens with a long tracking shot through a busy marketplace. "
    "We follow the protagonist as she navigates the crowd, buys flowers, and heads home. "
    "This movie is absolutely dreadful and one of the worst I have seen this decade. "
    "The second half continues with a series of domestic scenes and quiet conversations "
    "that round out the narrative structure before the credits roll."
)
my_test_labels.append("negative")   # sentiment buried in the middle

# --- Key information at the END only ---
# Deliberately neutral language throughout; the only sentiment word is the very last token.
# This is the hardest test for RNN (vanishing gradient across long neutral sequence).

my_test_texts.append(
    "The director establishes the setting through a series of wide establishing shots. "
    "Characters are introduced one by one, each given a brief backstory via dialogue. "
    "The cinematographer makes use of natural light throughout, giving the film a documentary feel. "
    "Sound design is minimalist, and the editing follows a steady rhythm from scene to scene. "
    "The screenplay adheres closely to the three-act structure with few surprises along the way. "
    "All things considered, an absolutely unforgettable and deeply wonderful film."
)
my_test_labels.append("positive")   # sentiment only at the very end; GRU advantage

# --- Long review (tests memory over many tokens near MAX_LENGTH) ---

my_test_texts.append(
    "I have been watching films for over forty years and I can say with complete confidence that "
    "this ranks among the greatest achievements in cinema history. The director weaves together "
    "multiple storylines with astonishing grace, never losing the emotional thread that binds them. "
    "The lead actress delivers a career-defining performance, subtle yet devastating, and her "
    "chemistry with the supporting cast is electric. The score is hauntingly beautiful, the "
    "production design is immaculate, and every line of dialogue rings with truth. "
    "I left the theatre feeling deeply moved and immediately wanted to see it again. "
    "A rare film that rewards repeated viewing and stays with you long after the credits roll. "
    "Absolutely essential cinema."
)
my_test_labels.append("positive")   # long positive; tests memory over full sequence

my_test_texts.append(
    "I sat through this for two and a half hours hoping it would improve, and it never did. "
    "The script is a patchwork of clichés and borrowed ideas, none of them executed with "
    "any conviction. The pacing is glacial, the dialogue is laughably bad, and the visual "
    "effects look cheaper than a student production. Every twist is telegraphed well in advance, "
    "and the finale is a baffling, incoherent mess that answers none of the questions raised. "
    "The cast clearly has talent — you can see them trying — but they are let down at every "
    "turn by the material. A catastrophic waste of resources and the audience's time. "
    "Avoid at all costs."
)
my_test_labels.append("negative")   # long negative

# --- Short review (minimal context for the model to work with) ---

my_test_texts.append("Loved it.")
my_test_labels.append("positive")

my_test_texts.append("Absolute garbage.")
my_test_labels.append("negative")

# --- Sarcasm / irony (hard for all models) ---
# Surface tokens are positive; actual sentiment is negative.

my_test_texts.append(
    "Oh yes, this film was just spectacular. I definitely did not fall asleep twice "
    "and I absolutely cannot wait to never watch it again. Pure genius."
)
my_test_labels.append("negative")   # sarcastic; surface tokens mislead

##########################
##########################

# ── MLP Error-Analysis Reviews ────────────────────────────────────────────────
# These four reviews are designed to expose the MLP's key weakness:
# it classifies each word independently and has no notion of context or negation.
#
# TP – clearly positive surface words → MLP correctly predicts positive
# TN – clearly negative surface words → MLP correctly predicts negative
# FP – NEGATIVE review packed with positive surface words (awards, love, praise)
#       → MLP is tricked into predicting positive
# FN – POSITIVE review packed with negative surface words (not, fail, disappoint)
#       → MLP is tricked into predicting negative

error_analysis_texts = []
error_analysis_labels = []

# TP: true positive, predicted positive
error_analysis_texts.append(
    "Wonderful, brilliant, and deeply moving. The performances were extraordinary "
    "and the story was beautiful. I loved every single moment of this masterpiece."
)
error_analysis_labels.append("positive")

# TN: true negative, predicted negative
error_analysis_texts.append(
    "Awful, boring, and painfully dull. The worst film I have seen this year. "
    "Terrible acting, a dismal script, and a complete waste of time."
)
error_analysis_labels.append("negative")

# FP: true NEGATIVE but surface is flooded with positive words → MLP predicts positive
# The decisive sentiment is at the end but is overwhelmed by positive tokens earlier.
error_analysis_texts.append(
    "This celebrated award-winning masterpiece received tremendous praise from critics "
    "worldwide. Audiences loved the brilliant performances and the beautiful cinematography. "
    "Acclaimed as a triumph of modern cinema. I personally found it utterly hollow, "
    "pretentious, and profoundly disappointing."
)
error_analysis_labels.append("negative")

# FN: true POSITIVE but surface is flooded with negative words → MLP predicts negative
# Negation chains confuse the MLP because it cannot combine 'not' with the word it negates.
error_analysis_texts.append(
    "I cannot say this film disappointed me. Not once did it fail to impress. "
    "I was not bored, not frustrated, not unmoved. Far from terrible or forgettable. "
    "Nothing about it was bad. I would not hesitate to recommend it."
)
error_analysis_labels.append("positive")

# ── Context-Sensitive Reviews ─────────────────────────────────────────────────
# These reviews contain words whose sentiment FLIPS depending on neighboring words.
# A plain MLP (per-word, no context) is expected to misclassify them.
# A model with restricted self-attention can read the surrounding window and correct the signal.

context_sensitive_texts = []
context_sensitive_labels = []

# (1) "far from masterpiece" — "masterpiece" is strongly positive alone,
#     but "far from X" inverts the sentiment of X.
#     MLP sees "masterpiece" → predicts positive.  Attention sees "far from masterpiece" → negative.
context_sensitive_texts.append(
    "Far from the masterpiece critics called it, this film was a profound disappointment "
    "that failed to deliver on every promise made by its trailer."
)
context_sensitive_labels.append("negative")

# (2) "not all sequels are terrible" — "terrible" is strongly negative alone,
#     but "not all ... are terrible" is actually a positive claim about this film.
#     MLP sees "terrible" → predicts negative.  Attention sees "not ... terrible" → positive.
context_sensitive_texts.append(
    "This sequel proves that not all follow-ups are terrible; "
    "it is genuinely moving, brilliantly acted, and wholly satisfying."
)
context_sensitive_labels.append("positive")

# (3) Comparative negation: "nothing short of brilliant" — "nothing" and "short" are
#     neutral-to-negative tokens, but together with "of brilliant" they form superlative praise.
context_sensitive_texts.append(
    "The director's vision is nothing short of brilliant, "
    "and the performances are nothing less than extraordinary."
)
context_sensitive_labels.append("positive")


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, review_list, labels):
        'Initialization'
        self.labels = labels
        self.reviews = review_list

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        X = self.reviews[index]
        y = self.labels[index]
        return X, y



def get_data_set(batch_size, toy=False):
        train_data, test_data = load_data_set(load_my_reviews=toy)
        train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                      shuffle=True, collate_fn=collact_batch)
        test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                     shuffle=True, collate_fn=collact_batch)
        return train_dataloader, test_dataloader, MAX_LENGTH, embedding_size


