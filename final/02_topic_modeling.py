# ============================================================
# BERTopic Pipeline for PA Government Data Center Statements
# ============================================================

import pandas as pd
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# ── 1. Excel ──────────────────────────────────────────
df = pd.read_excel("pa_gov_statements.xlsx")
print(f"load {len(df)} ")
print(f"column: {df.columns.tolist()}")

# ── Preprocessin ──────────────────────────────
def preprocess(text):
    if not isinstance(text, str):
        return ""
    # blank space
    text = re.sub(r'\s+', ' ', text)
    # delete punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    # remove http
    text = re.sub(r'http\S+', '', text)

df["clean_text"] = df["full_text"].apply(preprocess)

print(f" {len(df)} corpus")

# ── 3. prepare list ───────────────────────────────────────────
docs = df["clean_text"].tolist()

# ── 4. BERT Embedding ─────────────────────────────────────
embedding_model = SentenceTransformer("all-mpnet-base-v2")
print("Embedding finished")

# ── 5. reduce dimensionality（UMAP） ───────────────────────────────────────

umap_model = UMAP(
    n_neighbors=10,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42  #reproducibility
)

# ── 6. cluster（HDBSCAN） ────────────────────────────────────
# min_cluster_size: 163
hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    min_samples=3,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# ── CountVectorizer ─────────────────────────

vectorizer_model = CountVectorizer(
    stop_words="english",
    min_df=2,          # in at least 2 documents
    ngram_range=(1, 2) 
)

# ── 8. BERTopic ───────────────────────────────────────
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_n_words=10,        # 10 keywords
    verbose=True
)

# ── 9. model ───────────────────────────────────────────────
print("\n train BERTopic..")
topics, probs = topic_model.fit_transform(docs)

# ── 10. results ──────────────────────────────────────────────
print("\n=== Topic  ===")
topic_info = topic_model.get_topic_info()
print(topic_info)

print("\n=== keywords in teach Topic ===")
for topic_id in topic_info["Topic"].tolist():
    if topic_id == -1:
        continue  
    words = topic_model.get_topic(topic_id)
    keywords = [w[0] for w in words[:8]]
    count = topic_info[topic_info["Topic"] == topic_id]["Count"].values[0]
    print(f"Topic {topic_id} ({count}): {', '.join(keywords)}")

# ── merge back to DataFrame ───────────────────────────────────
df["topic"] = topics
df["topic_prob"] = probs

# keyword label
topic_labels = {}
for topic_id in set(topics):
    if topic_id == -1:
        topic_labels[-1] = "Noise"
        continue
    words = topic_model.get_topic(topic_id)
    topic_labels[topic_id] = "_".join([w[0] for w in words[:3]])

df["topic_label"] = df["topic"].map(topic_labels)

# ── 12. save result ──────────────────────────────────────────────
df.to_excel("pa_gov_topic_results.xlsx", index=False)
print(f"\n finish  save to pa_gov_topic_results.xlsx")
print(f"all {len(topic_info[topic_info['Topic'] != -1])}  topics")
print(f"noise: {(df['topic'] == -1).sum()} ")

