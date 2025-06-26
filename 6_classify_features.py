import scipy.stats
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


feature_df = pd.read_csv("data/feature_activations.csv")
feature_data = feature_df.to_dict(orient="records")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Step 1: Automatically set dead feature threshold
mean_activations = [f["mean_activation"] for f in feature_data]
activation_threshold = 0.005 

print(f"Using activation threshold: {activation_threshold}")

# Step 2: Junk vs Rich classification (rule-based)
def classify_junk_vs_rich(top_tokens, token_entropy):
    total = len(top_tokens)
    if total == 0:
        return "junk"

    unique = len(set(top_tokens))
    lexical_diversity = unique / total
    stopword_ratio = sum(1 for t in top_tokens if t.lower() in stop_words) / total
    avg_token_len = np.mean([len(t) for t in top_tokens])

    vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='char')
    ngram_freq = vectorizer.fit_transform([' '.join(top_tokens)])
    ngram_entropy = scipy.stats.entropy(np.asarray(ngram_freq.sum(axis=0)).flatten())

    if token_entropy < 1.5 and lexical_diversity < 0.5:
        return "junk"
    if stopword_ratio > 0.5 or avg_token_len < 3:
        return "junk"
    if ngram_entropy < 3.0:
        return "junk"
    return "rich"

# Step 3: Split into 3 categories
dead_features = []
junk_features = []
rich_features = []

for f in feature_data:
    if f["mean_activation"] < activation_threshold:
        dead_features.append(f)
    else:
        label = classify_junk_vs_rich(f["top_tokens"], f["token_entropy"])
        if label == "junk":
            junk_features.append(f)
        else:
            rich_features.append(f)

# Convert to separate DataFrames
dead_features_df = pd.DataFrame(dead_features)
junk_features_df = pd.DataFrame(junk_features)
rich_features_df = pd.DataFrame(rich_features)

# Optional: Save to disk
dead_features_df.to_csv("data/dead_features.csv", index=False)
junk_features_df.to_csv("data/junk_features.csv", index=False)
rich_features_df.to_csv("data/rich_features.csv", index=False)

print(f"\n✅ Split complete!")
print(f"☠️  Dead features: {len(dead_features_df)}")
print(f"🟡 Junk features: {len(junk_features_df)}")
print(f"🟢 Rich features: {len(rich_features_df)}")