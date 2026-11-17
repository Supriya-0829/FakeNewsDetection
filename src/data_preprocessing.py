"""
Data Preprocessing Module for Fake News Detection.

Performs data loading, label assignment, text cleaning, deduplication,
and data preparation for training and inference.
"""

import os
import re
import string
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Common English stop words fallback list in case NLTK data path is restricted
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

# Try importing NLTK stopwords safely
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import nltk
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = FALLBACK_STOPWORDS


def clean_text(text: Optional[str], strip_publisher_tags: bool = True) -> str:
    """
    Cleans raw textual input according to FR-02 and Section 8.1:
    - Handles non-string or None values safely.
    - Converts text to lowercase.
    - Removes HTML tags and markup.
    - Removes web URLs.
    - Strips publisher dateline artifacts (e.g. 'WASHINGTON (Reuters) - ').
    - Removes bracketed text and citations.
    - Removes punctuation and digits.
    - Normalizes multiple spaces and tabs to a single space.

    Args:
        text: Input string of news text.
        strip_publisher_tags: Whether to strip news agency datelines.

    Returns:
        Cleaned text string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>+", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove news agency datelines that cause shortcut learning (e.g., 'WASHINGTON (Reuters) - ')
    if strip_publisher_tags:
        text = re.sub(r"^[a-z\s]+ \((?:reuters)\)\s*-\s*", " ", text)
        text = re.sub(r"\((?:reuters)\)\s*-\s*", " ", text)
        text = re.sub(r"\breuters\b", " ", text)

    # Remove bracketed content (e.g., [VIDEO], [Image])
    text = re.sub(r"\[.*?\]", " ", text)

    # Remove punctuation
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)

    # Remove digits and alphanumeric numbers
    text = re.sub(r"\w*\d\w*", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_prepare_dataset(
    fake_path: str = "dataset/Fake.csv",
    true_path: str = "dataset/True.csv",
    include_title: bool = True,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads, cleans, labels, and merges the fake and real news datasets according
    to Section 4.4 and FR-01/FR-02.

    Labels:
        0 = FAKE
        1 = REAL

    Args:
        fake_path: Filepath to Fake.csv
        true_path: Filepath to True.csv
        include_title: Whether to combine headline title with body text.
        sample_size: Optional integer to take a stratified sample (useful for quick testing).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (prepared_dataframe, summary_metadata_dict)
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake news dataset not found at: {fake_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True news dataset not found at: {true_path}")

    logger.info(f"Loading Fake news from: {fake_path}")
    fake_df = pd.read_csv(fake_path)
    logger.info(f"Loaded {len(fake_df):,} fake news records.")

    logger.info(f"Loading True news from: {true_path}")
    true_df = pd.read_csv(true_path)
    logger.info(f"Loaded {len(true_df):,} true news records.")

    # Assign binary labels: 0 for FAKE, 1 for REAL
    fake_df["label"] = 0
    fake_df["label_name"] = "FAKE"
    true_df["label"] = 1
    true_df["label_name"] = "REAL"

    # Merge datasets
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)
    raw_count = len(combined_df)
    logger.info(f"Merged dataset total rows: {raw_count:,}")

    # Handle missing values
    combined_df["title"] = combined_df["title"].fillna("")
    combined_df["text"] = combined_df["text"].fillna("")

    # Construct content text
    if include_title:
        # Combine title and text to capture headline signals + article content
        combined_df["raw_content"] = combined_df["title"] + " " + combined_df["text"]
    else:
        combined_df["raw_content"] = combined_df["text"]

    # Apply text cleaning
    logger.info("Applying text cleaning pipeline...")
    combined_df["cleaned_text"] = combined_df["raw_content"].apply(clean_text)

    # Remove articles with virtually empty text after cleaning (fewer than 5 characters)
    initial_cleaned_count = len(combined_df)
    combined_df = combined_df[combined_df["cleaned_text"].str.len() >= 5].copy()
    empty_removed = initial_cleaned_count - len(combined_df)
    logger.info(f"Removed {empty_removed:,} records with empty/insufficient text.")

    # Remove duplicates on cleaned text
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["cleaned_text"], keep="first").copy()
    duplicates_removed = before_dedup - len(combined_df)
    logger.info(f"Removed {duplicates_removed:,} duplicate articles.")

    # Shuffle combined dataset
    combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Optional subsampling if requested
    if sample_size and sample_size < len(combined_df):
        logger.info(f"Subsampling to {sample_size:,} records...")
        combined_df = combined_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 2), random_state=random_state)
        ).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    fake_count = int((combined_df["label"] == 0).sum())
    real_count = int((combined_df["label"] == 1).sum())
    total_count = len(combined_df)

    metadata = {
        "raw_count": raw_count,
        "empty_removed": empty_removed,
        "duplicates_removed": duplicates_removed,
        "final_count": total_count,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_ratio": round(fake_count / total_count, 4) if total_count > 0 else 0,
        "real_ratio": round(real_count / total_count, 4) if total_count > 0 else 0,
        "random_state": random_state,
    }

    logger.info(f"Dataset preparation complete: {fake_count:,} FAKE ({metadata['fake_ratio']:.1%}), "
                f"{real_count:,} REAL ({metadata['real_ratio']:.1%}), Total: {total_count:,}")

    return combined_df, metadata


if __name__ == "__main__":
    df, meta = load_and_prepare_dataset()
    print("\n--- DATASET SUMMARY ---")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("\nSample Cleaned Text:")
    print(df[["label_name", "cleaned_text"]].head(3))
