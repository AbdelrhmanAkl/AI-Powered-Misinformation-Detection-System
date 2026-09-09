from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "misinformation_model_package.joblib"
)

TRUE_DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "True.csv"
)

FAKE_DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "Fake.csv"
)

# Number of samples from each class.
# Set to None to evaluate the complete datasets.
SAMPLES_PER_CLASS = 500

RANDOM_STATE = 42


# ============================================================
# Validation
# ============================================================

def validate_paths() -> None:
    """Validate required project files."""

    required_paths = {
        "Model package": MODEL_PATH,
        "True dataset": TRUE_DATA_PATH,
        "Fake dataset": FAKE_DATA_PATH,
    }

    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )


# ============================================================
# Load Production Package
# ============================================================

def load_model_package():
    """Load the frozen production model package."""

    package = joblib.load(MODEL_PATH)

    if not isinstance(package, dict):
        raise ValueError(
            "Invalid model package: expected a dictionary."
        )

    required_keys = {
        "model",
        "vectorizer",
        "threshold",
        "label_mapping",
        "feature_policy",
        "metadata",
    }

    missing_keys = required_keys - set(package.keys())

    if missing_keys:
        raise ValueError(
            "Model package is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return package


# ============================================================
# Dataset Loading
# ============================================================

def load_datasets() -> pd.DataFrame:
    """
    Load True.csv and Fake.csv and create ground-truth labels.

    Label mapping:
        0 = Real News
        1 = Fake News
    """

    true_df = pd.read_csv(TRUE_DATA_PATH)
    fake_df = pd.read_csv(FAKE_DATA_PATH)

    required_columns = {"title", "text"}

    for name, df in [
        ("True.csv", true_df),
        ("Fake.csv", fake_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing)}"
            )

    # Keep only production features.
    true_df = true_df[["title", "text"]].copy()
    fake_df = fake_df[["title", "text"]].copy()

    # Ground truth.
    true_df["label"] = 0
    fake_df["label"] = 1

    # Combine datasets.
    df = pd.concat(
        [true_df, fake_df],
        ignore_index=True,
    )

    # Clean missing values.
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Remove completely empty articles.
    df = df[
        (df["title"].str.strip() != "")
        | (df["text"].str.strip() != "")
    ].copy()

    return df


# ============================================================
# Sampling
# ============================================================

def sample_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a balanced evaluation sample.

    Sampling is performed separately for each class.
    """

    if SAMPLES_PER_CLASS is None:
        return df.reset_index(drop=True)

    sampled_parts = []

    for label in [0, 1]:
        class_df = df[df["label"] == label]

        sample_size = min(
            SAMPLES_PER_CLASS,
            len(class_df),
        )

        sampled = class_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )

        sampled_parts.append(sampled)

    result = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    return result.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


# ============================================================
# Feature Preparation
# ============================================================

def prepare_text(df: pd.DataFrame) -> pd.Series:
    """
    Build inference text according to the production feature policy.
    """

    combined = (
        df["title"].str.strip()
        + " "
        + df["text"].str.strip()
    ).str.strip()

    if combined.eq("").any():
        raise ValueError(
            "One or more samples contain no usable text."
        )

    return combined


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    df: pd.DataFrame,
    package: dict,
):
    """Run production inference and calculate metrics."""

    model = package["model"]
    vectorizer = package["vectorizer"]

    threshold = float(package["threshold"])

    label_mapping = package["label_mapping"]

    texts = prepare_text(df)

    # IMPORTANT:
    # transform() only.
    # No fit() or fit_transform().
    features = vectorizer.transform(texts)

    # Production probability estimates.
    probabilities = model.predict_proba(features)

    # Class 1 = Fake News.
    probability_fake = probabilities[:, 1]

    # Frozen production threshold.
    predictions = (
        probability_fake >= threshold
    ).astype(int)

    y_true = df["label"].to_numpy()

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=[
            label_mapping.get(0, "Real News"),
            label_mapping.get(1, "Fake News"),
        ],
        zero_division=0,
    )

    return (
        metrics,
        matrix,
        report,
        probability_fake,
        predictions,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Production Model Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    validate_paths()

    # --------------------------------------------------------
    # Load production package
    # --------------------------------------------------------

    package = load_model_package()

    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = float(package["threshold"])
    feature_policy = package["feature_policy"]
    label_mapping = package["label_mapping"]

    print("\nProduction Configuration")
    print("-" * 70)
    print(f"Model           : {type(model).__name__}")
    print(f"Vectorizer      : {type(vectorizer).__name__}")
    print(f"Threshold       : {threshold}")
    print(f"Feature Policy  : {feature_policy}")
    print(f"Label Mapping   : {label_mapping}")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    df = load_datasets()

    total_real = int((df["label"] == 0).sum())
    total_fake = int((df["label"] == 1).sum())

    print("\nAvailable Dataset")
    print("-" * 70)
    print(f"Real articles   : {total_real}")
    print(f"Fake articles   : {total_fake}")
    print(f"Total articles  : {len(df)}")

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    evaluation_df = sample_dataset(df)

    real_count = int(
        (evaluation_df["label"] == 0).sum()
    )

    fake_count = int(
        (evaluation_df["label"] == 1).sum()
    )

    print("\nEvaluation Dataset")
    print("-" * 70)
    print(f"Real samples    : {real_count}")
    print(f"Fake samples    : {fake_count}")
    print(f"Total samples   : {len(evaluation_df)}")

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    (
        metrics,
        matrix,
        report,
        probability_fake,
        predictions,
    ) = evaluate(
        evaluation_df,
        package,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\nPerformance Metrics")
    print("-" * 70)
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"F1 Score        : {metrics['f1']:.4f}")

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix")
    print("-" * 70)
    print("Rows    = Actual")
    print("Columns = Predicted")
    print()
    print("              Predicted")
    print("              Real   Fake")
    print(
        f"Actual Real   {matrix[0, 0]:5d}  {matrix[0, 1]:5d}"
    )
    print(
        f"Actual Fake   {matrix[1, 0]:5d}  {matrix[1, 1]:5d}"
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report")
    print("-" * 70)
    print(report)

    # --------------------------------------------------------
    # Prediction Distribution
    # --------------------------------------------------------

    real_predictions = int(
        (predictions == 0).sum()
    )

    fake_predictions = int(
        (predictions == 1).sum()
    )

    print("Prediction Distribution")
    print("-" * 70)
    print(f"Predicted Real : {real_predictions}")
    print(f"Predicted Fake : {fake_predictions}")

    # --------------------------------------------------------
    # Probability Summary
    # --------------------------------------------------------

    print("\nFake-News Probability")
    print("-" * 70)
    print(
        f"Minimum        : {probability_fake.min():.4f}"
    )
    print(
        f"Maximum        : {probability_fake.max():.4f}"
    )
    print(
        f"Mean           : {probability_fake.mean():.4f}"
    )

    # --------------------------------------------------------
    # Important Note
    # --------------------------------------------------------

    print("\nEvaluation Note")
    print("-" * 70)
    print(
        "These results are valid as a production inference "
        "diagnostic only if these samples were not used during "
        "model training."
    )

    print(
        "If True.csv and Fake.csv contain training data used to "
        "fit the model, the reported metrics may be optimistic "
        "and should not be presented as held-out test performance."
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()