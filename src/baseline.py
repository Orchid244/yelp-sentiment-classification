import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import (
    BASELINE_DIR,
    ID2LABEL,
    RATING_COL,
    RANDOM_SEED,
    TEXT_COL,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from evaluate import evaluate_predictions, plot_confusion
from utils import save_json, save_text



def run_baseline(train_df, val_df, test_df):
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_df["text_baseline"])
    X_val = vectorizer.transform(val_df["text_baseline"])
    X_test = vectorizer.transform(test_df["text_baseline"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        #multi_class="ovr",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_metrics = evaluate_predictions(y_val, val_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)

    save_json(val_metrics, os.path.join(BASELINE_DIR, "val_metrics.json"))
    save_json(test_metrics, os.path.join(BASELINE_DIR, "test_metrics.json"))
    save_text(val_metrics["classification_report_text"], os.path.join(BASELINE_DIR, "val_report.txt"))
    save_text(test_metrics["classification_report_text"], os.path.join(BASELINE_DIR, "test_report.txt"))

    labels = [ID2LABEL[i] for i in range(len(ID2LABEL))]
    plot_confusion(
        val_metrics["confusion_matrix"],
        labels,
        "Baseline Validation Confusion Matrix",
        os.path.join(BASELINE_DIR, "val_confusion_matrix.png"),
    )
    plot_confusion(
        test_metrics["confusion_matrix"],
        labels,
        "Baseline Test Confusion Matrix",
        os.path.join(BASELINE_DIR, "test_confusion_matrix.png"),
    )

    pred_df = test_df[[TEXT_COL, RATING_COL, "label_name"]].copy()
    pred_df["pred_label_name"] = [ID2LABEL[i] for i in test_pred]
    pred_df.to_csv(os.path.join(BASELINE_DIR, "test_predictions.csv"), index=False)

    return {
        "model": "TF-IDF + Logistic Regression",
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }