# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv (3.15.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # IMDB Sentiment Analysis

# %%
# %pip -q install datasets

# %% [markdown]
# ## 1. Load a balanced IMDB subset
#
# We load the IMDB movie review dataset and select equal numbers of positive and negative reviews for training and testing. This balanced sample keeps the classes evenly represented, making the model comparisons easier to interpret.

# %%
from datasets import load_dataset
from collections import Counter, defaultdict
from html import unescape
import re
import random

SEED = 89

raw = load_dataset("stanfordnlp/imdb")

def balanced_split(split, n_per_class, seed=SEED):
    rows = list(split)
    random.Random(seed).shuffle(rows)
    negative_reviews = []
    positive_reviews = []

    for row in rows:
        if row["label"] == 0 and len(negative_reviews) < n_per_class:
            negative_reviews.append(row)
        elif row["label"] == 1 and len(positive_reviews) < n_per_class:
            positive_reviews.append(row)

        if len(negative_reviews) == n_per_class and len(positive_reviews) == n_per_class:
            break

    if len(negative_reviews) < n_per_class or len(positive_reviews) < n_per_class:
        raise ValueError("The dataset split does not contain enough examples for each label.")

    selected = negative_reviews + positive_reviews
    random.Random(seed).shuffle(selected)
    return selected

train_data = balanced_split(raw["train"], 2500)
test_data = balanced_split(raw["test"], 500)

print(f"Training reviews: {len(train_data)} | labels: {dict(Counter(x['label'] for x in train_data))}")
print(f"Test reviews:     {len(test_data)} | labels: {dict(Counter(x['label'] for x in test_data))}")
print(f"Raw review example, Label = {train_data[0]['label']}:\n", train_data[0]["text"][:700])


# %% [markdown]
# ## 2. Pre-processing and tokenization
#
# We clean each review by decoding HTML entities, removing markup, converting text to lowercase, and standardizing whitespace and punctuation. Then we split the cleaned text into tokens and add start and end markers so the language model can learn sentence boundaries.

# %%
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

def preprocess_text(text):
    text = unescape(text)   
    text = re.sub(r"<[^>]+>", " ", text)        
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    tokens = preprocess_text(text).split()
    return [START_TOKEN] + tokens + [END_TOKEN]

example_raw = train_data[0]["text"]
print("Cleaned text example:\n", preprocess_text(example_raw)[:500])
print("Tokens example:\n", tokenize(example_raw)[:35])


# %% [markdown]
# ## 3. Build positive and negative bigram language models
#
# We count adjacent token pairs separately in positive and negative training reviews. These bigram models capture how likely one word is to follow another in each sentiment class, while a shared vocabulary handles words consistently.

# %%
train_tokens = [tokenize(row["text"]) for row in train_data]
word_vocabulary = {token for sequence in train_tokens for token in sequence}
word_vocabulary.add(UNK_TOKEN)
VOCAB_SIZE = len(word_vocabulary)

def map_unk(tokens, vocabulary=word_vocabulary):
    return [token if token in vocabulary else UNK_TOKEN for token in tokens]

def count_bigrams(token_sequences):
    bigram_counts = Counter()
    context_counts = Counter()
    for tokens in token_sequences:
        for first, second in zip(tokens, tokens[1:]):
            bigram_counts[(first, second)] += 1
            context_counts[first] += 1
    return bigram_counts, context_counts

def fit_bigram_lm(token_sequences):
    bigrams, contexts = count_bigrams(token_sequences)
    return {"bigram_counts": bigrams, "context_counts": contexts, "vocab_size": VOCAB_SIZE}

positive_sequences = [map_unk(tokens) for row, tokens in zip(train_data, train_tokens) if row["label"] == 1]
negative_sequences = [map_unk(tokens) for row, tokens in zip(train_data, train_tokens) if row["label"] == 0]

positive_lm = fit_bigram_lm(positive_sequences)
negative_lm = fit_bigram_lm(negative_sequences)

print(f"Vocabulary size: {VOCAB_SIZE:,}")
print(f"Positive bigram count: {sum(positive_lm['bigram_counts'].values()):,}")
print(f"Negative bigram count: {sum(negative_lm['bigram_counts'].values()):,}")

# %% [markdown]
# ## 4. Log-likelihood and perplexity
#
# We score a review under each sentiment model using add one smoothing, which gives unseen bigrams a nonzero probability. The review is assigned to the model with lower perplexity, meaning that model predicts its token sequence more consistently.

# %%
import math

def log_probability(lm, first, second):
    num = lm["bigram_counts"][(first, second)] + 1
    denum = lm["context_counts"][first] + lm["vocab_size"]
    return math.log(num / denum)

def score_document(text, lm):
    tokens = map_unk(tokenize(text))
    log_likelihood = sum(
        log_probability(lm, first, second)
        for first, second in zip(tokens, tokens[1:])
    )
    number_of_bigrams = len(tokens) - 1
    perplexity = math.exp(-log_likelihood / number_of_bigrams)
    return log_likelihood, perplexity

def compare_lms(text):
    pos_ll, pos_ppl = score_document(text, positive_lm)
    neg_ll, neg_ppl = score_document(text, negative_lm)
    return {
        "positive_log_likelihood": pos_ll,
        "positive_perplexity": pos_ppl,
        "negative_log_likelihood": neg_ll,
        "negative_perplexity": neg_ppl,
    }

example_result = compare_lms(test_data[0]["text"])
print("True label:", "positive" if test_data[0]["label"] == 1 else "negative")
for metric, value in example_result.items():
    print(f"{metric}: {value:.4f}")


# %% [markdown]
# ## 5. LM Prediction

# %%

predicted_labels = []
true_labels = []

for row in test_data:
    result = compare_lms(row["text"])

    if result["positive_perplexity"] <= result["negative_perplexity"]:
        prediction = 1
    else:
        prediction = 0

    predicted_labels.append(prediction)
    true_labels.append(row["label"])

correct = sum(
    p == t for p, t in zip(predicted_labels, true_labels)
)
lm_acc = correct / len(true_labels)

TP = sum(p == 1 and t == 1 for p, t in zip(predicted_labels, true_labels))
FP = sum(p == 1 and t == 0 for p, t in zip(predicted_labels, true_labels))
FN = sum(p == 0 and t == 1 for p, t in zip(predicted_labels, true_labels))
TN = sum(p == 0 and t == 0 for p, t in zip(predicted_labels, true_labels))

precision_fun = TP / (TP + FP) if TP + FP > 0 else 0
recall_fun = TP / (TP + FN) if TP + FN > 0 else 0

lm_f1= (
    2 * precision_fun * recall_fun / (precision_fun + recall_fun)
    if precision_fun + recall_fun > 0 else 0
)

print("LM Classification Results")
print("-" * 30)
print(f"Accuracy:  {lm_acc:.4f}")
print(f"Precision: {precision_fun:.4f}")
print(f"Recall:    {recall_fun:.4f}")
print(f"F1-Score:  {lm_f1:.4f}")
print(f"True Positives:  {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Negatives:  {TN}")


# %% [markdown]
# ## 6. LM Inference

# %%


def generate_text(sentence, lm, max_words=20):
    # Clean and tokenize the input sentence
    prefix_words = preprocess_text(sentence).split()
    context_tokens = map_unk(prefix_words)

    if not context_tokens:
        context_tokens = [START_TOKEN]

    generated_words = []

    for _ in range(max_words):
        current_word = context_tokens[-1]

        candidates = word_vocabulary

        next_word = max(
            candidates,
            key=lambda word: log_probability(lm, current_word, word)
        )

        if next_word == END_TOKEN:
            break

        generated_words.append(next_word)
        context_tokens.append(next_word)

    return " ".join(prefix_words + generated_words)

sentences = [
    "the acting was",
    "i really enjoyed",
    "the story is",
    "this film had"
]

for sentence in sentences:
    print("\nInput:", sentence)

    print("Positive LM:")
    print(generate_text(sentence, positive_lm, max_words=10))

    print("Negative LM:")
    print(generate_text(sentence, negative_lm, max_words=10))


# %% [markdown]
# ## Output Analysis
# The positive language model frequently generates words such as “great,” while the negative language model tends to generate more generic phrases such as “a movie.” This demonstrates that the two models learn different word-transition patterns from positive and negative reviews.
#
# However, the generated sentences are sometimes repetitive and lack grammatical coherence. This is a limitation of the bigram model because it considers only the previous word when predicting the next word, rather than the full sentence context.

# %% [markdown]
# # Part 2: Naive Bayes
#

# %% [markdown]
# ### 1. Build the Vocabulary 

# %%
from collections import Counter
def extract_features(tokens):
    # vocab of unigram and bigrams
    bigrams=[]
    for first,second in zip(tokens,tokens[1:]):
        bigram=" ".join((first,second))
        bigrams.append(bigram)
    return tokens+bigrams

def build_vocab(train_tokens,vocab_size=2000):
    feature_counts=Counter()
    feature_index_dict={}
    for review_tokens in train_tokens:
        d=Counter(extract_features(review_tokens))
        feature_counts.update(d)
    top_features=feature_counts.most_common(vocab_size)
    
    for i,(feature,_) in enumerate(top_features):
        feature_index_dict[feature]=i
    return feature_index_dict
feature_to_index=build_vocab(train_tokens)


# %% [markdown]
# ### 2. Class-condtional Probabilities

# %% [markdown]
# Count of Occurences for each feature in both positive and negative reviews

# %%
import numpy as np

def count_features(token_sequences,feature_index_dict):
    counts=np.zeros(len(feature_index_dict))
    for token in token_sequences:
        for feature in extract_features(token):
            if feature in feature_index_dict:
                counts[feature_index_dict[feature]]+=1
    return counts
    
positive_tokens = [tokens for row, tokens in zip(train_data, train_tokens) if row["label"] == 1]
negative_tokens = [tokens for row, tokens in zip(train_data, train_tokens) if row["label"] == 0]

positive_counts = count_features(positive_tokens, feature_to_index)
negative_counts = count_features(negative_tokens, feature_to_index)

print(f"Positive total feature occurrences: {int(positive_counts.sum()):,}")
print(f"Negative total feature occurrences: {int(negative_counts.sum()):,}")
print()

# Top 5 features per class
index_to_feature = {index: feature for feature, index in feature_to_index.items()}
print(f"{'Feature':<20}{'Positive':>10}{'Negative':>10}")
print("-" * 40)
for index in range(5):
    feature = index_to_feature[index]
    print(f"{feature:<20}{int(positive_counts[index]):>10,}{int(negative_counts[index]):>10,}")


# %% [markdown]
# Laplace Smoothed probabilities

# %%
def smoothed_probs(counts):
    v=len(counts)
    n=counts.sum()
    probs=np.zeros(v)
    for i in range(v):
        probs[i]=(counts[i]+1)/(n+ v)
    return probs

positive_probs = smoothed_probs(positive_counts)
negative_probs = smoothed_probs(negative_counts)

print(np.isclose(positive_probs.sum(),1.0))
print(np.isclose(negative_probs.sum(),1.0))


# %% [markdown]
# ### 3. Log-ratios(r)

# %%
def log_ratios(p1,p2):
    return np.log(p1) - np.log(p2)

def log_prior_ratio(p1,p2):
    n_p1=len(p1)
    n_p2=len(p2)
    return math.log(n_p1/n_p2)
    
feature_names = [None] * len(feature_to_index)
for feature, index in feature_to_index.items():
    feature_names[index] = feature

r=log_ratios(positive_probs,negative_probs)

sorted_indices = np.argsort(r)
top_10_negative_indices = sorted_indices[:10]
top_10_positive_indices = sorted_indices[-10:]
print("10 Most Negative Features:")
for i in top_10_negative_indices:
    print(f"{feature_names[i]}: {r[i]:.4f}")

print("\n10 Most Positive Features:")
for i in reversed(top_10_positive_indices):
    print(f"{feature_names[i]}: {r[i]:.4f}")

# %% [markdown]
# ### 4. Naive Bayes Prediction

# %%
def predict_naive_bayes(text,feature_to_index,r,prior):
    features=extract_features(tokenize(text))
    score=prior
    for feature in features:
        if feature in feature_to_index:
            score+=r[feature_to_index[feature]]
    label=1 if score>0 else 0
    return label,score

prior = log_prior_ratio(positive_tokens, negative_tokens) 

label, score = predict_naive_bayes(test_data[0]["text"], feature_to_index, r, prior)

print(f"Test Example:{test_data[0]['text']}")
print(f"True Label: {test_data[0]['label']}")
print(f"Predicted: {label}  (score {score:.3f})  ")

# %% [markdown]
# Evaluation Metrics

# %%
def accuracy(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

def precision_fun(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_fun(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def confusion_counts(predictions, true_labels):
    tp = fp = fn = tn = 0
    for predicted, actual in zip(predictions, true_labels):
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn

# %% [markdown]
# ### 5. Evaluation over the test data

# %%
predictions=[]
scores=[]
true_labels=[row['label'] for row in test_data]

for row in test_data:
    label,score= predict_naive_bayes(row['text'],feature_to_index,r,prior)
    predictions.append(label)
    scores.append(score)
    
tp, fp, fn, tn = confusion_counts(predictions, true_labels)
acc=accuracy(tp,fp,fn,tn)
prec=precision_fun(tp,fp)
rec=recall_fun(tp,fn)
f1=f1_score(prec,rec)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")



# %% [markdown]
# # Part 3: NB-Weighted Logistic Regression

# %% [markdown]
# ### 1. Feature Matrix Construction

# %%
import matplotlib.pyplot as plt

# Keep the Naive Bayes results
nb_acc, nb_prec, nb_rec, nb_f1 = acc, prec, rec, f1

def build_count_matrix(token_sequences, feature_index_dict):
    X = np.zeros((len(token_sequences), len(feature_index_dict)))
    for i, tokens in enumerate(token_sequences):
        for feature in extract_features(tokens):
            j = feature_index_dict.get(feature)
            if j is not None:
                X[i, j] += 1
    return X

test_tokens = [tokenize(row["text"]) for row in test_data]

X_train_counts = build_count_matrix(train_tokens, feature_to_index)
X_test_counts = build_count_matrix(test_tokens, feature_to_index)

# Weight every count by its Naive Bayes log-ratio 
X_train_nb = X_train_counts * r
X_test_nb = X_test_counts * r

y_train = np.array([row["label"] for row in train_data])
y_test = np.array([row["label"] for row in test_data])

print("Train matrix:", X_train_nb.shape, "| Test matrix:", X_test_nb.shape)

# %% [markdown]
# ### 2. Standardization + Bias column

# %%
mu = X_train_nb.mean(axis=0)
sigma = X_train_nb.std(axis=0)
sigma[sigma == 0] = 1.0   # guard against constant columns

def standardize_with_bias(X):
    X_std = (X - mu) / sigma
    return np.hstack([X_std, np.ones((X.shape[0], 1))])  

X_train = standardize_with_bias(X_train_nb)
X_test = standardize_with_bias(X_test_nb)

print("Final train matrix:", X_train.shape, "| Final test matrix:", X_test.shape)
print(f"Standardized train features: mean ~ {X_train[:, :-1].mean():.4f}, std ~ {X_train[:, :-1].std():.4f}")

# %% [markdown]
# ### 3. Logistic Regression with Gradient Descent (Binary Cross-Entropy)

# %%
class LogisticRegression:
    def __init__(self, lr=0.05, epochs=300, l2=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2                  
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, X, y):
        p = self.sigmoid(X @ self.w)
        eps = 1e-12
        bce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return bce + 0.5 * self.l2 * np.sum(self.w[:-1] ** 2)

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.loss_history = []
        for epoch in range(1, self.epochs + 1):
            p = self.sigmoid(X @ self.w)
            grad = X.T @ (p - y) / n
            grad[:-1] += self.l2 * self.w[:-1]      # L2 gradient, bias excluded
            self.w -= self.lr * grad

            current_loss = self.loss(X, y)
            self.loss_history.append(current_loss)

            if epoch == 1 or epoch % 50 == 0:
                train_acc = np.mean(self.predict(X) == y)
                print(f"Epoch {epoch:4d}/{self.epochs} | loss: {current_loss:.4f} | train acc: {train_acc:.4f}")
        return self

# %%
lr_model = LogisticRegression(lr=0.05, epochs=1000, l2=0.01)
lr_model.fit(X_train, y_train)

# %%
plt.figure(figsize=(6, 3.5))
plt.plot(lr_model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training loss (BCE + L2)")
plt.title("Logistic Regression: Gradient Descent")
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ### 4. Evaluation

# %%
lr_predictions = lr_model.predict(X_test)
tp, fp, fn, tn = confusion_counts(lr_predictions, y_test)

lr_acc = accuracy(tp, fp, fn, tn)
lr_prec = precision_fun(tp, fp)
lr_rec = recall_fun(tp, fn)
lr_f1 = f1_score(lr_prec, lr_rec)

print(f"Accuracy:  {lr_acc:.4f}")
print(f"Precision: {lr_prec:.4f}")
print(f"Recall:    {lr_rec:.4f}")
print(f"F1-score:  {lr_f1:.4f}")

# %% [markdown]
# # Part 4: Evaluation and Analysis

# %%
print(f"{'Model':<28}{'Accuracy':>10}{'F1':>10}")
print("-" * 48)
print(f"{'Bigram LM ':<28}{lm_acc:>10.4f}{lm_f1:>10.4f}")
print(f"{'Naive Bayes':<28}{nb_acc:>10.4f}{nb_f1:>10.4f}")
print(f"{'NB-weighted Logistic Reg.':<28}{lr_acc:>10.4f}{lr_f1:>10.4f}")

# %% [markdown]
# ## Analysis
#
# | Model | Input representation | Accuracy | F1 |
# |---|---|---|---|
# | Bigram LM | Ordered token sequence, full 42k vocabulary, previous-word context | 0.855 | 0.855 |
# | Naive Bayes | Bag of 2,000 unigram+bigram counts, summed log-ratios | 0.815 | 0.814 |
# | Logistic Regression | Same 2,000 features, count × log-ratio, standardized | 0.843 | 0.828 |
#
# The Bigram LM performed best. It sees the full vocabulary and word order, while the other two only see 2,000 features and discard order.
#
# Naive Bayes was lowest because unigrams and bigrams overlap, which violates its conditional-independence assumption: the same words are counted several times.
#
# Logistic Regression improved on Naive Bayes by about 2 points because it learns feature weights jointly instead of independently. Its training accuracy (97.5%) is far above its test accuracy (83.5%), so it overfits. We added a small L2 penalty to limit this, but it only reduces the gap.

# %% [markdown]
# # Team 
# Amr Attia 8907
#
#
# Sama Mohamed 8803
#
#
# Habiba Rezq 8633
#
#
# Yousef Nabil 8749
