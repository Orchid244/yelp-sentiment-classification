import os

import pandas as pd

from baseline import run_baseline
from config import (
    RANDOM_SEED,
    RESULT_DIR,
    RUN_FROZEN_ENCODER,
    RUN_FULL_FINETUNE,
    TRANSFORMER_MODEL_DIR,
    TRANSFORMER_RESULT_DIR,
)
from data_loader import load_and_prepare_data, save_data_summary, save_splits, split_data
from train_transformer import TransformerConfig, train_transformer
from utils import ensure_dirs, set_seed



def main():
    set_seed(RANDOM_SEED)
    ensure_dirs()

    print("Loading data...")
    df = load_and_prepare_data()
    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df)
    save_data_summary(train_df, val_df, test_df)

    print("Running baseline model...")
    baseline_result = run_baseline(train_df, val_df, test_df)
    results = [baseline_result]

    if RUN_FROZEN_ENCODER:
        print("Running DistilBERT feature extraction (frozen encoder)...")
        frozen_cfg = TransformerConfig(
            run_name="DistilBERT-Frozen",
            freeze_encoder=True,
            model_dir=os.path.join(TRANSFORMER_MODEL_DIR, "distilbert_frozen"),
            result_dir=os.path.join(TRANSFORMER_RESULT_DIR, "distilbert_frozen"),
        )
        results.append(train_transformer(train_df, val_df, test_df, frozen_cfg))

    if RUN_FULL_FINETUNE:
        print("Running DistilBERT full fine-tuning...")
        finetune_cfg = TransformerConfig(
            run_name="DistilBERT-Finetuned",
            freeze_encoder=False,
            model_dir=os.path.join(TRANSFORMER_MODEL_DIR, "distilbert_finetuned"),
            result_dir=os.path.join(TRANSFORMER_RESULT_DIR, "distilbert_finetuned"),
        )
        results.append(train_transformer(train_df, val_df, test_df, finetune_cfg))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)

    print("\nFinal comparison:")
    print(results_df)
    print("\nDone. Outputs saved to ./outputs")


if __name__ == "__main__":
    main()