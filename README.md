# Can We Predict a Hit Before We License It? (INST414 Module 5)

Supervised classification on 15,201 IMDB films (2000–2022, 100+ votes). The model predicts whether a film's
average IMDB user rating is **7.0 or higher**, using only information a streaming acquisitions team would
know before licensing it: genre, Module 5 k-means cluster, runtime, year, vote count, and the cast's track record.

## Results (held-out test set, 3,041 films)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Always "not well-rated" | 0.862 | 0.000 | 0.000 | 0.000 | 0.500 |
| Random forest | 0.850 | 0.474 | 0.764 | 0.585 | 0.895 |

The model's top 10% of films by score are 65% well-rated, versus a 14% base rate.

## Run it

```
pip install -r requirements.txt
python predict_ratings.py   # main analysis: downloads the IMDB data, trains, writes figures/ and outputs/
python checks.py            # validation: feature ablation, genre sanity check, error-film details
python leak_check.py        # shows what happens if cast features leak each film's own rating
```

`predict_ratings.py` downloads the IMDB data from the course archive on first run.

## Data

- IMDB films: [imdb_movies_2000to2022.prolific.json](https://raw.githubusercontent.com/cbuntain/umd.inst414/main/data/imdb_movies_2000to2022.prolific.json)
  from the [course repo](https://github.com/cbuntain/umd.inst414). Not committed here; the script downloads it.
- `data/movie_to_cluster.csv`: Module 5 k-means cluster for each film (from the course ELMS page).

## Outputs

- `figures/`: the four charts used in the post
- `outputs/cv_model_comparison.csv`: 5-fold CV results for Naive Bayes, kNN, logistic regression, random forest
- `outputs/test_metrics.csv`, `outputs/confusion_matrix.csv`: held-out test results
- `outputs/hit_rate_by_score_decile.csv`: share of well-rated films in each score decile
- `outputs/feature_importance.csv`, `outputs/feature_ablation.csv`: what drives the model
- `outputs/top_false_positives.csv`, `outputs/top_false_negatives.csv`: the most confident mistakes
- `outputs/test_predictions.csv`: every test film with its score and prediction
