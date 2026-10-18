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
