"""
Assignment 2 language Analytics
By: Réka Forgó
Date: 18. 03. 2026.

#Methods: 
# Featuriser: TF-IDF
# Model: SVM
# Hyperparameter tuning using GridSearch CV, across 5 folds



"""


import os
import time
import random
import argparse
import numpy as np
import pandas as pd

 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

from utils import (
    load_data,
    plot_cv_scores,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    save_metrics_log,
)

#Set Random seed!
random.seed(42)
np.random.seed(42)

# paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "out")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

for d in [OUTPUT_DIR, FIG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# timestamp for filenames
ts = time.strftime("%Y%m%d_%H%M")


def build_pipeline():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(random_state=42, max_iter=2000)),  
    ])
    return pipeline
 
def tune_pipeline(pipeline, X_train, y_train):
    # hyperparameter grid for grid search
    param_grid = {
        "tfidf__max_features": [5000, 20000, None],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.1, 1.0, 10.0]
    }
 
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
 
    print("Running grid search...")
    grid.fit(X_train, y_train)
    print("Best CV accuracy:", grid.best_score_)
    print("Best params:", grid.best_params_)
    return grid
 
 
def evaluate_model_on_test(model, X_test, y_test):
    # required function - takes test data and returns accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)
    print(classification_report(y_test, y_pred))
    return acc, y_pred
 
 

def main():
    # parse arguments
    # run without test data:  python main.py
    # run with test data:     python main.py --test_data /path/to/test.csv
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="path to test dataset. if not provided, only cross-validation on training data is run",
    )
    args = parser.parse_args()
 
    # load training data
    train_file = os.path.join(DATA_DIR, "fake_real_news_train_data.csv")
    df_train = load_data(train_file)
 
    X_train = df_train["text"]
    y_train = df_train["label"]
 
    print("Train size:", len(X_train))
 
    # tune model using cross-validation on training data only
    pipeline = build_pipeline()
    grid = tune_pipeline(pipeline, X_train, y_train)
    best_model = grid.best_estimator_
 
    # cross-validate the best model on training data to get per-fold scores
    print("Running cross-validation on training data with best model...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print("CV scores per fold:", cv_scores)
    print("Mean CV accuracy:", round(cv_scores.mean(), 4))
    print("Std CV accuracy:", round(cv_scores.std(), 4))
    
    # plot cv scores histogram
    plot_cv_scores(cv_scores, FIG_DIR, ts)
 
    # save metrics so far
    metrics_dict = {
        "timestamp": ts,
        "best_params": grid.best_params_,
        "best_cv_accuracy": round(grid.best_score_, 4),
        "mean_cv_accuracy": round(cv_scores.mean(), 4),
        "std_cv_accuracy": round(cv_scores.std(), 4),
        "cv_scores_per_fold": cv_scores.tolist(),
    }
 
    # only run test evaluation if a test set was provided
    if args.test_data is not None:
        print("Loading test data from:", args.test_data)
        df_test = load_data(args.test_data)
        X_test = df_test["text"]
        y_test = df_test["label"]
 
        test_acc, y_pred = evaluate_model_on_test(best_model, X_test, y_test)  # ← unpack both
        labels = sorted(y_train.unique().tolist())

        pos_label = "FAKE"
 
        plot_confusion_matrix(y_test, y_pred, labels, FIG_DIR, ts)
        roc_auc = plot_roc_curve(best_model, X_test, y_test, pos_label, FIG_DIR, ts)
        avg_prec = plot_pr_curve(best_model, X_test, y_test, pos_label, FIG_DIR, ts)
 
        metrics_dict["test_data"] = args.test_data
        metrics_dict["test_accuracy"] = round(test_acc, 4)
        metrics_dict["roc_auc"] = round(roc_auc, 4)
        metrics_dict["avg_precision"] = round(avg_prec, 4)
    else:
        print("No test data provided - skipping test evaluation")
 
    save_metrics_log(metrics_dict, OUTPUT_DIR, ts)
 
    # save model
    model_path = os.path.join(MODEL_DIR, f"{ts}_svm_pipeline.joblib")
    dump(best_model, model_path)
    print("Model saved to:", model_path)
 
    return best_model
 
 
if __name__ == "__main__":
    main()