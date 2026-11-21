"""INST414 Module 5: Can we predict which movies audiences will rate highly?

Supervised classification on the course IMDB dataset (films 2000-2022).
Label: well_rated = 1 if the film's average IMDB user rating is >= 7.0.

Run:  python predict_ratings.py
The script downloads the data from the course GitHub archive if it isn't in data/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_URL = ("https://raw.githubusercontent.com/cbuntain/umd.inst414/main/data/"
            "imdb_movies_2000to2022.prolific.json")
DATA_PATH = Path("data/imdb_movies_2000to2022.prolific.json")
CLUSTER_PATH = Path("data/movie_to_cluster.csv")   # Module 5 k-means clusters (course ELMS file)
FIG = Path("figures")
OUT = Path("outputs")
THRESHOLD = 7.0      # rating that counts as "well rated"
MIN_VOTES = 100      # below this, the average rating is too noisy to use as a label
SEED = 414

FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. Load
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_bytes(requests.get(DATA_URL, timeout=60).content)

records = [json.loads(line) for line in DATA_PATH.open()]
raw = pd.DataFrame(records)
print(f"Loaded {len(raw):,} films")

# ---------------------------------------------------------------- 2. Clean
df = raw.copy()
df["avg_rating"] = df["rating"].apply(lambda r: r.get("avg") if isinstance(r, dict) else np.nan)
df["votes"] = df["rating"].apply(lambda r: r.get("votes") if isinstance(r, dict) else np.nan)
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")      # "\N" -> NaN
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["genres"] = df["genres"].apply(lambda gs: [g for g in gs if g])  # drop empty-string genre
clusters = pd.read_csv(CLUSTER_PATH).rename(columns={"movie_id": "imdb_id"})
df = df.merge(clusters, on="imdb_id", how="left")

cleaning = {"loaded": len(df)}
df = df[df["avg_rating"].notna()]
cleaning["has_rating"] = len(df)
df = df[df["votes"] >= MIN_VOTES]
cleaning[f"votes_ge_{MIN_VOTES}"] = len(df)
df = df.drop_duplicates("imdb_id").reset_index(drop=True)
cleaning["deduplicated"] = len(df)
print("Cleaning steps:", cleaning)

df["well_rated"] = (df["avg_rating"] >= THRESHOLD).astype(int)
print(f"Well-rated share: {df['well_rated'].mean():.1%}")

# ---------------------------------------------------------------- 3. Split first (no leakage)
train, test = train_test_split(df, test_size=0.2, stratify=df["well_rated"], random_state=SEED)
train, test = train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------- 4. Features
GENRES = sorted({g for gs in train["genres"] for g in gs})
CLUSTERS = sorted(train["cluster"].dropna().astype(int).unique())


def actor_table(films):
    """One row per (film, actor) with the film's rating."""
    rows = [(f.imdb_id, a[0], f.avg_rating) for f in films.itertuples() for a in f.actors]
    return pd.DataFrame(rows, columns=["imdb_id", "actor_id", "avg_rating"])


train_links = actor_table(train)
actor_stats = train_links.groupby("actor_id")["avg_rating"].agg(["sum", "count"])
global_mean = train["avg_rating"].mean()


def cast_features(films, is_train):
    """Cast track record = average rating of each actor's OTHER training films.
    For training films the film itself is left out (leave-one-out), so a film's own
    rating never leaks into its features. Test films only see training films."""
    links = actor_table(films).join(actor_stats, on="actor_id")
    links[["sum", "count"]] = links[["sum", "count"]].fillna(0)
    if is_train:
        links["sum"] -= links["avg_rating"]
        links["count"] -= 1
    links["other_films"] = links["count"]
    links["actor_mean"] = np.where(links["count"] > 0, links["sum"] / links["count"].clip(lower=1), np.nan)
    agg = links.groupby("imdb_id").agg(
        cast_mean_rating=("actor_mean", "mean"),
        cast_best_rating=("actor_mean", "max"),
        cast_known_actors=("actor_mean", lambda s: s.notna().sum()),
        cast_total_films=("other_films", "sum"),
    )
    return films[["imdb_id"]].join(agg, on="imdb_id").drop(columns="imdb_id")


def build_features(films, is_train):
    X = pd.DataFrame(index=films.index)
    for g in GENRES:
        X[f"genre_{g}"] = films["genres"].apply(lambda gs, g=g: int(g in gs))
    X["n_genres"] = films["genres"].apply(len)
    for c in CLUSTERS:
        X[f"cluster_{c}"] = (films["cluster"] == c).astype(int)
    X["runtime_missing"] = films["runtime"].isna().astype(int)
    X["runtime"] = films["runtime"].fillna(train["runtime"].median())
    X["year"] = films["year"]
    X["log_votes"] = np.log10(films["votes"])
    X["n_actors_listed"] = films["actors"].apply(len)
    cast = cast_features(films, is_train)
    X["cast_has_history"] = cast["cast_known_actors"].gt(0).astype(int).values
    X["cast_mean_rating"] = cast["cast_mean_rating"].fillna(global_mean).values
    X["cast_best_rating"] = cast["cast_best_rating"].fillna(global_mean).values
    X["cast_total_films"] = cast["cast_total_films"].values
    return X


X_train, X_test = build_features(train, True), build_features(test, False)
y_train, y_test = train["well_rated"].values, test["well_rated"].values
print(f"Train {len(X_train):,} / test {len(X_test):,} films, {X_train.shape[1]} features")

# ---------------------------------------------------------------- 5. Pick a model with 5-fold CV on train only
candidates = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Naive Bayes (Gaussian)": make_pipeline(StandardScaler(), GaussianNB()),
    "kNN (k=25)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    "Logistic regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                            class_weight="balanced", n_jobs=-1, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_rows = []
for name, model in candidates.items():
    proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    cv_rows.append(dict(model=name, accuracy=accuracy_score(y_train, pred),
                        precision=precision_score(y_train, pred, zero_division=0),
                        recall=recall_score(y_train, pred), f1=f1_score(y_train, pred),
                        roc_auc=roc_auc_score(y_train, proba) if proba.std() > 0 else 0.5))
cv_table = pd.DataFrame(cv_rows)
cv_table.to_csv(OUT / "cv_model_comparison.csv", index=False)
print("\n5-fold CV on training set:\n", cv_table.round(3).to_string(index=False))

# ---------------------------------------------------------------- 6. Final model on held-out test set
final = candidates["Random forest"].fit(X_train, y_train)
test_proba = final.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)
baseline_pred = np.zeros_like(y_test)

test_metrics = pd.DataFrame([
    dict(model="Majority baseline", accuracy=accuracy_score(y_test, baseline_pred),
         precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5),
    dict(model="Random forest", accuracy=accuracy_score(y_test, test_pred),
         precision=precision_score(y_test, test_pred), recall=recall_score(y_test, test_pred),
         f1=f1_score(y_test, test_pred), roc_auc=roc_auc_score(y_test, test_proba)),
])
test_metrics.to_csv(OUT / "test_metrics.csv", index=False)
print("\nHeld-out test set:\n", test_metrics.round(3).to_string(index=False))

cm = confusion_matrix(y_test, test_pred)
print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
pd.DataFrame(cm, index=["actual <7", "actual >=7"], columns=["pred <7", "pred >=7"]).to_csv(OUT / "confusion_matrix.csv")

# Shortlist view: how many well-rated films land in each score decile
results = test[["imdb_id", "title", "year", "genres", "avg_rating", "votes"]].copy()
results["score"] = test_proba
results["predicted"] = test_pred
results["actual"] = y_test
results["decile"] = pd.qcut(results["score"].rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = results.groupby("decile", observed=True).agg(films=("actual", "size"), well_rated=("actual", "sum"),
                                                    hit_rate=("actual", "mean")).reset_index()
lift.to_csv(OUT / "hit_rate_by_score_decile.csv", index=False)
print("\nHit rate by model-score decile (1 = highest scores):\n", lift.round(3).to_string(index=False))

# Feature importance
imp = pd.Series(final.feature_importances_, index=X_train.columns).sort_values(ascending=False)
imp.to_csv(OUT / "feature_importance.csv", header=["importance"])
print("\nTop features:\n", imp.head(12).round(3).to_string())

# Sanity check: genre hit rates in the training data
genre_rates = pd.DataFrame([(g, train.loc[train.genres.apply(lambda gs: g in gs), "well_rated"].mean(),
                             train.genres.apply(lambda gs: g in gs).sum()) for g in GENRES],
                           columns=["genre", "well_rated_share", "films"]).sort_values("well_rated_share", ascending=False)
genre_rates.to_csv(OUT / "genre_hit_rates_train.csv", index=False)

# Errors with the most confidence (for the 5-sample discussion)
X_test_out = X_test.reset_index(drop=True)
results = pd.concat([results, X_test_out[["runtime", "log_votes", "cast_mean_rating", "cast_has_history"]]], axis=1)
fp = results[(results.predicted == 1) & (results.actual == 0)].sort_values("score", ascending=False)
fn = results[(results.predicted == 0) & (results.actual == 1)].sort_values("score")
fp.head(15).to_csv(OUT / "top_false_positives.csv", index=False)
fn.head(15).to_csv(OUT / "top_false_negatives.csv", index=False)
results.to_csv(OUT / "test_predictions.csv", index=False)
print("\nMost confident false positives:\n", fp.head(8)[["title", "year", "genres", "avg_rating", "votes", "score", "cast_mean_rating"]].to_string(index=False))
print("\nMost confident false negatives:\n", fn.head(8)[["title", "year", "genres", "avg_rating", "votes", "score", "cast_mean_rating"]].to_string(index=False))

# ---------------------------------------------------------------- 7. Figures
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GREY, BLUE = "#9aa0a6", "#1a73e8"

fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.arange(1, 10.01, 0.2)
ax.hist(df.loc[df.avg_rating < THRESHOLD, "avg_rating"], bins=bins, color=GREY)
ax.hist(df.loc[df.avg_rating >= THRESHOLD, "avg_rating"], bins=bins, color=BLUE)
ax.axvline(THRESHOLD, color="black", lw=1, ls="--")
ax.set_title(f"Only {df.well_rated.mean():.0%} of films reach a {THRESHOLD} rating", loc="left")
ax.set_xlabel("Average IMDB user rating")
ax.set_ylabel("Films")
fig.tight_layout(); fig.savefig(FIG / "fig1_rating_distribution.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = [BLUE if d <= 1 else GREY for d in lift.decile.astype(int)]
ax.bar(lift.decile.astype(int), lift.hit_rate * 100, color=colors)
ax.axhline(y_test.mean() * 100, color="black", lw=1, ls="--")
ax.text(10.4, y_test.mean() * 100 + 1, f"base rate {y_test.mean():.0%}", ha="right")
for d, r in zip(lift.decile.astype(int), lift.hit_rate):
    ax.text(d, r * 100 + 1, f"{r:.0%}", ha="center", fontsize=10)
ax.set_xticks(range(1, 11))
ax.set_xlabel("Model score decile (1 = films the model likes most)")
ax.set_ylabel("Share actually rated 7.0+ (%)")
ax.set_title(f"The model's top 10% of films are {lift.hit_rate.iloc[0]:.0%} well-rated", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig2_hit_rate_by_decile.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.6))
top = imp.head(12)[::-1]
ax.barh([n.replace("genre_", "genre: ").replace("_", " ") for n in top.index], top.values,
        color=[BLUE if i >= len(top) - 3 else GREY for i in range(len(top))])
ax.set_xlabel("Random forest importance (mean decrease in impurity)")
ax.set_title("Popularity, runtime and cast track record drive the predictions", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig3_feature_importance.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
ax.set_xticks([0, 1], ["Predicted <7", "Predicted 7+"])
ax.set_yticks([0, 1], ["Actual <7", "Actual 7+"])
ax.set_title("Confusion matrix, held-out test set", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig4_confusion_matrix.png", dpi=200); plt.close(fig)

pd.Series(cleaning).to_csv(OUT / "cleaning_steps.csv", header=["films_remaining"])
print("\nDone. Figures in figures/, tables in outputs/.")
