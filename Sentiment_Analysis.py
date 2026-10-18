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
    
r=log_ratios(positive_probs,negative_probs)

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

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(tp, fn):
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
prec=precision(tp,fp)
rec=recall(tp,fn)
f1=f1_score(prec,rec)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")



# %%
