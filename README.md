# Yelp Restaurant Review Sentiment Classification

The goal of the project is to classify restaurant reviews into three sentiment categories: negative, neutral, and positive.

The project compares a traditional machine learning baseline with transformer-based models:

- TF-IDF + Logistic Regression baseline
- DistilBERT with frozen encoder
- DistilBERT with full fine-tuning

## Project Overview

Restaurant review sentiment classification is a common natural language processing task. In this project, review ratings are converted into sentiment labels, and different models are trained and evaluated on the same train, validation, and test splits.

The main research question is whether fine-tuning a transformer model improves performance compared with a traditional TF-IDF baseline.

## Dataset

The dataset used in this project is the **Yelp Restaurant Reviews** dataset from Kaggle:

https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews?resource=download

Due to dataset licensing and redistribution considerations, the raw dataset is **not included** in this repository. To reproduce this project, please download the dataset directly from Kaggle and place the CSV file in the `data/` directory.

Expected input columns:

- `text`: review text
- `rating`: star rating from 1 to 5

The ratings are mapped into three sentiment classes:

| Rating | Sentiment Label |
|---|---|
| 1-2 | Negative |
| 3 | Neutral |
| 4-5 | Positive |

## Repository Structure

```text
.
├── baseline.py
├── config.py
├── data_loader.py
├── evaluate.py
├── main.py
├── preprocessing.py
├── train_transformer.py
├── utils.py
├── README.md
├── requirements.txt
├── .gitignore
└── outputs/
    └── results/
        ├── baseline/
        └── transformer/
            ├── distilbert_finetuned/
            └── distilbert_frozen/
```

The following folders are intentionally excluded from the repository:

- `data/`: raw Kaggle dataset files
- `outputs/splits/`: train, validation, and test split CSV files
- `outputs/models/`: trained model checkpoints and saved tokenizer/model files
- `test_predictions.csv`: prediction files that may contain original review text

## Methods

### 1. TF-IDF + Logistic Regression Baseline

The baseline model uses TF-IDF vectorization to convert review text into numerical features. A Logistic Regression classifier is then trained for three-class sentiment classification.

This model provides a strong traditional machine learning benchmark.

### 2. DistilBERT with Frozen Encoder

The frozen DistilBERT model uses a pre-trained DistilBERT encoder, while only the classification head is trained. This setup tests whether pre-trained language representations alone are sufficient for the task.

### 3. DistilBERT with Full Fine-Tuning

The fine-tuned DistilBERT model updates the full transformer model during training. This allows the model to better adapt to the Yelp review sentiment classification task.

## Results

The following table summarizes the test set performance:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8558 | 0.7249 | 0.8656 |
| DistilBERT Frozen | 0.7744 | 0.6187 | 0.7959 |
| DistilBERT Fine-tuned | 0.8648 | 0.7327 | 0.8722 |

The fine-tuned DistilBERT model achieved the best overall performance. However, the improvement over the TF-IDF + Logistic Regression baseline was relatively small. The frozen DistilBERT model performed worse than both, suggesting that only training the classification head was not sufficient for this dataset.

Across all models, the positive class was generally the easiest to classify, while the neutral class was more difficult.

## Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- torch
- transformers

## How to Run

1. Download the dataset from Kaggle:

   https://www.kaggle.com/datasets/farukalam/yelp-restaurant-reviews?resource=download

2. Place the dataset CSV file inside the `data/` directory.

3. Check the dataset path in `config.py` and update it if necessary.

4. Run the full experiment pipeline:

```bash
python main.py
```

The script will:

- Load and clean the dataset
- Convert ratings into sentiment labels
- Split the data into train, validation, and test sets
- Train the TF-IDF + Logistic Regression baseline
- Train the DistilBERT frozen model
- Train the DistilBERT fine-tuned model
- Save evaluation results to `outputs/results/`

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Classification report
- Confusion matrix

Macro F1 is especially important because the dataset is class-imbalanced and it gives equal weight to each class.

## Notes on Reproducibility

The project sets a random seed for reproducibility. However, transformer training results may still vary slightly depending on hardware, CUDA behavior, and library versions.

The raw dataset, split files, and trained model checkpoints are not included in this repository. They can be regenerated by running the pipeline after downloading the dataset.

## License

This repository is for academic coursework purposes. The dataset is provided by Kaggle and should be used according to the dataset's original terms and conditions.
