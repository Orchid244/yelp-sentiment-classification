import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    OUTPUT_DIR,
    RATING_COL,
    RANDOM_SEED,
    SPLIT_DIR,
    TEST_SIZE,
    TEXT_COL,
    VAL_SIZE,
)
from preprocessing import (
    clean_text_for_baseline,
    clean_text_for_transformer,
    map_label_to_id,
    map_rating_to_label,
)
from utils import save_json



def load_and_prepare_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [TEXT_COL, RATING_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[[TEXT_COL, RATING_COL]].copy()
    df = df.dropna(subset=[TEXT_COL, RATING_COL]).reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=[RATING_COL]).reset_index(drop=True)
    df[RATING_COL] = df[RATING_COL].astype(int)
    df = df[df[RATING_COL].between(1, 5)].reset_index(drop=True)

    df["label_name"] = df[RATING_COL].apply(map_rating_to_label)
    df["label"] = df["label_name"].apply(map_label_to_id)
    df["text_baseline"] = df[TEXT_COL].apply(clean_text_for_baseline)
    df["text_transformer"] = df[TEXT_COL].apply(clean_text_for_transformer)

    df = df[df["text_baseline"].str.len() > 0].reset_index(drop=True)
    return df



def split_data(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    relative_val_size = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_size,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )



def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)



def save_data_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    summary = {
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "train_label_distribution": train_df["label_name"].value_counts().to_dict(),
        "val_label_distribution": val_df["label_name"].value_counts().to_dict(),
        "test_label_distribution": test_df["label_name"].value_counts().to_dict(),
    }
    save_json(summary, os.path.join(OUTPUT_DIR, "data_summary.json"))