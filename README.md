# Movie Review Sentiment Analysis

Sentiment analysis on IMDB reviews using bigram language models, Naive Bayes, and Logistic Regression, all implemented from scratch with Python and NumPy.

**Dataset:** balanced IMDB subset, 5,000 train reviews (2,500 positive, 2,500 negative) and 1,000 test reviews (500 positive, 500 negative), with a fixed seed (`89`).

| Part | What it does |
|---|---|
| 1 | Cleaning, tokenization, class-conditional bigram LMs (Add-1), perplexity-based classification, next-word prediction |
| 2 | 2,000-feature unigram and bigram vocabulary, Naive Bayes with log-ratios |
| 3 | NB-weighted feature matrix, standardization, Logistic Regression with gradient descent |
| 4 | Comparison of the three models and discussion |
