# train_baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("gh_data/csv/repos_dataset.csv")
# Keep only repos with README and a non-"other" label (optionnel)
df = df[df["readme_len"] > 50]
# Use auto_label but ideally use human labels
X = df["readme_text"].fillna("")  # you can concat description + topics
y = df["auto_label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1,2))
Xtr = tfidf.fit_transform(X_train)
Xv = tfidf.transform(X_val)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(Xtr, y_train)

pred = clf.predict(Xv)
print(classification_report(y_val, pred))
joblib.dump((tfidf, clf), "models/tfidf_logreg.joblib")
