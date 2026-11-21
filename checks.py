"""Validation checks used in the post: ablation without votes, genre sanity check, error-film casts."""
import runpy, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path("predict_ratings.py")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
X_train, X_test, y_train, y_test = g["X_train"], g["X_test"], g["y_train"], g["y_test"]
rows = []
for name, drop in [("All features", []), ("No vote count", ["log_votes"]),
                   ("No cast features", [c for c in X_train if c.startswith("cast_")]),
                   ("Genre + runtime + year only", [c for c in X_train if c.startswith("cast_")] + ["log_votes", "n_actors_listed"])]:
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=414)
    m.fit(X_train.drop(columns=drop), y_train)
    p = m.predict_proba(X_test.drop(columns=drop))[:, 1]; pr = (p >= .5).astype(int)
    rows.append(dict(features=name, precision=precision_score(y_test, pr), recall=recall_score(y_test, pr),
                     f1=f1_score(y_test, pr), roc_auc=roc_auc_score(y_test, p)))
abl = pd.DataFrame(rows); abl.to_csv("outputs/feature_ablation.csv", index=False)
print(abl.round(3).to_string(index=False))
print(pd.read_csv("outputs/genre_hit_rates_train.csv").round(3).to_string(index=False))
test = g["test"]
for t in ["The Amazing Spider-Man", "Wall Street: Money Never Sleeps", "Dreams Rewired", "Black Dynamite",
          "Tenets of the Maladjusted", "The Overnight"]:
    r = test[test.title == t].iloc[0]
    print(t, r.year, r.runtime, r.genres, r.avg_rating, r.votes, [a[1] for a in r.actors])
print("films with no genre after cleaning:", (g["df"].genres.apply(len) == 0).sum())
print("threshold sensitivity: share >=6.5", (g["df"].avg_rating >= 6.5).mean().round(3))
