"""
Prediction and Inference Module for Fake News Detection.

Loads the saved scikit-learn pipeline, validates user inputs, applies
text preprocessing, performs classification, calculates confidence scores,
and returns structured predictions according to FR-07, FR-08, NFR-03, and NFR-06.
"""

import os
import sys
from pathlib import Path

# Add project root to path for robust imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import joblib

from src.data_preprocessing import clean_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "fake_news_pipeline.joblib",
)


@lru_cache(maxsize=4)
def load_cached_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads and caches the trained scikit-learn pipeline in memory.

    Args:
        model_path: Path to the serialized pipeline .joblib file.

    Returns:
        Loaded Pipeline object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    logger.info(f"Loading pipeline from: {model_path}")
    return joblib.load(model_path)


def predict_news(
    text: Optional[str],
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Classifies an input news article or headline as REAL or FAKE.

    Implements:
    - FR-07: News Prediction
    - FR-08: Prediction Display & Confidence Estimate
    - NFR-03: Graceful error and empty input handling
    - NFR-06: Text-only parsing without executing user code

    Args:
        text: Raw user-provided news text or headline.
        model_path: Filepath of the trained joblib model pipeline.

    Returns:
        Dictionary containing prediction results, probabilities, and diagnostics.
    """
    # NFR-03: Validate empty or whitespace-only inputs
    if text is None or not isinstance(text, str) or not text.strip():
        return {
            "status": "error",
            "message": "Input text is empty. Please enter news headline or article content.",
            "label": None,
            "confidence": 0.0,
        }

    raw_text = text.strip()
    raw_word_count = len(raw_text.split())

    # Preprocess text (FR-02)
    cleaned = clean_text(raw_text)
    cleaned_words = cleaned.split()
    cleaned_word_count = len(cleaned_words)

    if cleaned_word_count == 0:
        return {
            "status": "error",
            "message": "Input contains only punctuation, special symbols, or stopwords. Please provide descriptive text.",
            "label": None,
            "confidence": 0.0,
            "cleaned_text": "",
        }

    # Load model pipeline safely
    try:
        pipeline = load_cached_pipeline(model_path)
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"Model pipeline artifact was not found at '{model_path}'. Please train the model first.",
            "label": None,
            "confidence": 0.0,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load model pipeline: {str(e)}",
            "label": None,
            "confidence": 0.0,
        }

    # Perform inference (FR-07)
    prediction = pipeline.predict([cleaned])[0]
    class_id = int(prediction)
    label = "REAL" if class_id == 1 else "FAKE"

    # Calculate probabilities and confidence estimate (FR-08)
    probabilities = {"FAKE": 0.5, "REAL": 0.5}
    confidence = 0.5

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba([cleaned])[0]
            # proba layout: index 0 is FAKE (class 0), index 1 is REAL (class 1)
            prob_fake = float(proba[0])
            prob_real = float(proba[1])
            probabilities = {
                "FAKE": round(prob_fake, 4),
                "REAL": round(prob_real, 4),
            }
            confidence = prob_real if class_id == 1 else prob_fake
        except Exception as e:
            logger.warning(f"Could not compute predict_proba: {e}")

    # Interpret confidence level
    if confidence >= 0.85:
        confidence_level = "High"
    elif confidence >= 0.65:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low / Uncertain"

    return {
        "status": "success",
        "label": label,
        "class_id": class_id,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{confidence * 100:.1f}%",
        "confidence_level": confidence_level,
        "probabilities": probabilities,
        "cleaned_text": cleaned,
        "raw_word_count": raw_word_count,
        "cleaned_word_count": cleaned_word_count,
        "confidence_disclaimer": (
            "Confidence estimates represent linguistic similarity to the training dataset, "
            "not verification of factual ground truth."
        ),
    }


if __name__ == "__main__":
    test_articles = [
        "NASA announces successful launch of new deep space exploration satellite telescope.",
        "SHOCKING: Secret alien base discovered under government building! Click to see shocking footage!",
    ]
    print("\n--- INFERENCE TEST ---")
    for article in test_articles:
        res = predict_news(article)
        print(f"Article: {article[:50]}...")
        print(f"Result: {res['label']} ({res.get('confidence_percent')}) - Level: {res.get('confidence_level')}\n")
