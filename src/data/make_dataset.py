# -*- coding: utf-8 -*-
"""
Script used to read external files in order to generate training, development and test sets.
python -m src.data.make_dataset
"""

import logging
import pandas as pd
from pathlib import Path
from src.data.text_preprocessing import TextPreprocessing
from src.data.read_dataset import split_data

project_dir = Path(__file__).resolve().parents[2]


def create_train_test_sets(dataset_file: str, train_frac: float) -> None:
    """
    Create training and test sets with the same distribution for axis (class), in order to train and evaluate trained models
    param dataset_file: Data to be split
    param train_frac: Ratio of train set to whole dataset
    Randomly split dataset, based on these ratios:
        'train': train_frac
        'test': 1-train_frac
    Eg: passing train_frac=0.8 gives a 80% / 20% split
    """
    path = project_dir / 'data' / 'blenderbot' / 'processed'
    df = pd.read_csv(dataset_file, delimiter=",",
                        header=0, encoding='utf-8', engine='python')

    # Renaming columns
    #df.rename(columns={'axis': 'label', 'eval_labels': 'text'}, inplace=True)
    df.rename(columns={'noun_gender': 'label', 'eval_labels': 'text'}, inplace=True)

    # Filter
    #df = df[(df['label'] == 'body_type') | (df['label'] == 'characteristics') | (df['label'] == 'ability')]

    # Clean
    df['text'] = df.apply(lambda row: TextPreprocessing(str(row.text)).remove_html(), axis=1)

    # Get labels
    labels = df['label']

    # Split data
    train, test = split_data(df, labels, train_frac)

    # Resume
    logging.info('\ntrain-------------------------------------------------------------')
    logging.info(train.shape)
    logging.info('label     %')
    logging.info(f" {round(train.groupby('label')['text'].count() * 100 / train.shape[0], 2)}")

    logging.info('\ndev-------------------------------------------------------------')
    logging.info(test.shape)
    logging.info('label     %')
    logging.info(f" {round(test.groupby('label')['text'].count() * 100 / test.shape[0], 2)}")

    # Save files
    train.to_csv(path / 'train.csv', index=False)
    test.to_csv(path / 'test.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    file_path = project_dir / 'data' / 'blenderbot' / 'all_perplexities.csv'
    create_train_test_sets(file_path, 0.7)