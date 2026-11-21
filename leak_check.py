"""Leakage check: what happens if a film's own rating is NOT left out of its cast average?"""
import runpy, io, contextlib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path("predict_ratings.py")
train, X_train, y_train = g["train"], g["X_train"], g["y_train"]
links = g["actor_table"](train).join(g["actor_stats"], on="actor_id")
links["actor_mean"] = links["sum"] / links["count"]          # own rating included = leak
leaky = links.groupby("imdb_id")["actor_mean"].agg(["mean", "max"])
X_leak = X_train.copy()
X_leak["cast_mean_rating"] = train["imdb_id"].map(leaky["mean"]).values
X_leak["cast_best_rating"] = train["imdb_id"].map(leaky["max"]).values
cv = StratifiedKFold(5, shuffle=True, random_state=414)
for name, X in [("leave-one-out (used)", X_train), ("own rating included (leaky)", X_leak)]:
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=414)
    p = cross_val_predict(m, X, y_train, cv=cv, method="predict_proba")[:, 1]
    print(f"{name}: CV F1 {f1_score(y_train, p >= .5):.3f}, ROC AUC {roc_auc_score(y_train, p):.3f}")
