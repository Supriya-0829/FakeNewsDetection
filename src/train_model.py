"""
Model Training Module for Fake News Detection.

Loads preprocessed dataset, splits into stratified train/test sets (80/20),
trains primary Logistic Regression TF-IDF pipeline and optional comparison models
(Naive Bayes, Passive Aggressive, Linear SVM), evaluates performance,
and saves the trained pipeline with Joblib.
"""

import os
import sys
from pathlib import Path

# Add project root to path for robust imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_and_prepare_dataset
from src.evaluate_model import evaluate_pipeline, compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_pipeline(classifier_type: str = "logistic_regression", random_state: int = 42) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with TF-IDF vectorization and classifier.

    Args:
        classifier_type: One of 'logistic_regression', 'naive_bayes',
                         'passive_aggressive', 'linear_svm'.
        random_state: Reproducibility seed.

    Returns:
        Scikit-learn Pipeline.
    """
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )

    if classifier_type == "logistic_regression":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )
    elif classifier_type == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif classifier_type == "passive_aggressive":
        clf = PassiveAggressiveClassifier(
            max_iter=1000,
            random_state=random_state,
            early_stopping=True,
        )
    elif classifier_type == "linear_svm":
        base_svc = LinearSVC(random_state=random_state, max_iter=2000)
        clf = CalibratedClassifierCV(estimator=base_svc)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return Pipeline([("tfidf", tfidf), ("classifier", clf)])


def train_and_evaluate(
    fake_path: str = "dataset/Fake.csv",
    true_path: str = "dataset/True.csv",
    models_dir: str = "models",
    test_size: float = 0.20,
    random_state: int = 42,
    train_comparison_models: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Executes the end-to-end training and evaluation workflow according to
    FR-01, FR-03, FR-04, FR-05, FR-06, and Section 8.4.

    Args:
        fake_path: Path to Fake.csv
        true_path: Path to True.csv
        models_dir: Destination folder for serialized artifacts.
        test_size: Proportion of dataset to hold out for testing (default: 0.20).
        random_state: Seed for reproducible stratified split.
        train_comparison_models: Whether to also benchmark comparison algorithms.

    Returns:
        Tuple of (trained_primary_pipeline, evaluation_metrics_dict)
    """
    os.makedirs(models_dir, exist_ok=True)

    # Step 1: Load and prepare dataset (FR-01, FR-02)
    logger.info("Loading and preparing datasets...")
    df, meta = load_and_prepare_dataset(
        fake_path=fake_path,
        true_path=true_path,
        random_state=random_state,
    )

    X = df["cleaned_text"]
    y = df["label"].values

    # Step 2: Stratified Train/Test Split (Section 8.4)
    logger.info(f"Splitting dataset: {1 - test_size:.0%} train, {test_size:.0%} test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(f"Training set: {len(X_train):,} samples; Test set: {len(X_test):,} samples.")

    # Step 3: Train Primary Model (Logistic Regression Pipeline) (FR-03, FR-04)
    logger.info("Building and training primary Logistic Regression Pipeline...")
    primary_pipeline = build_pipeline("logistic_regression", random_state=random_state)
    primary_pipeline.fit(X_train, y_train)
    logger.info("Primary pipeline training complete.")

    # Step 4: Evaluate Primary Model on held-out test data (FR-05)
    logger.info("Evaluating primary model on held-out test data...")
    metrics = evaluate_pipeline(
        pipeline=primary_pipeline,
        X_test=X_test,
        y_test=y_test,
        save_dir=models_dir,
    )
    metrics["dataset_metadata"] = meta
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)

    # Step 5: Save Primary Pipeline with Joblib (FR-06)
    pipeline_path = os.path.join(models_dir, "fake_news_pipeline.joblib")
    joblib.dump(primary_pipeline, pipeline_path)
    logger.info(f"Primary pipeline saved successfully to: {pipeline_path}")

    # Step 6: Train & Benchmark Comparison Algorithms (Section 8.3)
    comparison_results = {}
    comparison_results["Logistic Regression (Primary)"] = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
    }

    if train_comparison_models:
        benchmarks = [
            ("Multinomial Naive Bayes", "naive_bayes"),
            ("Passive Aggressive Classifier", "passive_aggressive"),
            ("Linear Support Vector Machine", "linear_svm"),
        ]
        for name, clf_type in benchmarks:
            try:
                logger.info(f"Training comparison algorithm: {name}...")
                pipe = build_pipeline(clf_type, random_state=random_state)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                bm_metrics = compute_classification_metrics(y_test, y_pred)
                comparison_results[name] = {
                    "accuracy": bm_metrics["accuracy"],
                    "precision": bm_metrics["precision"],
                    "recall": bm_metrics["recall"],
                    "f1_score": bm_metrics["f1_score"],
                }
                logger.info(f"{name} Results -> Acc: {bm_metrics['accuracy']:.4f}, "
                            f"F1: {bm_metrics['f1_score']:.4f}")
            except Exception as e:
                logger.warning(f"Could not benchmark {name}: {e}")

        # Save comparisons JSON
        comp_file = os.path.join(models_dir, "model_comparisons.json")
        with open(comp_file, "w") as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"Model comparisons saved to: {comp_file}")
        metrics["model_comparisons"] = comparison_results

    # Update evaluation_metrics.json with full context
    metrics_file = os.path.join(models_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training completed successfully! Final Test Accuracy: {metrics['accuracy']:.4%}")
    return primary_pipeline, metrics


if __name__ == "__main__":
    train_and_evaluate()
