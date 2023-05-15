# -*- coding: utf-8 -*-
"""
Script used to read datasets files.
"""
import pandas as pd
import logging
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parents[2]


def split_data(data, labels, train_frac, random_state=None) -> Tuple[str, str]:
    """
    param data: Data to be split
    param labels: labels to be used on stratify
    param train_frac: Ratio of train set to whole dataset
    Randomly split dataset, based on these ratios:
        'train': train_frac
        'test': 1-train_frac
    Eg: passing train_frac=0.8 gives a 80% / 20% split
    """

    assert 0 <= train_frac <= 1, "Invalid training set fraction"

    train, test = train_test_split(data, train_size=train_frac, random_state=random_state, stratify=labels)
    return train, test


def get_data(data_source) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads datasets."""
    path = project_dir / 'data' / data_source / 'processed'
    if path.exists():
        try:
            df_train = pd.read_csv(path / 'train.csv', delimiter=",",
                                   usecols=["axis", "descriptor", "text", "ppl", "label"], #"ppl",
                                   header=0, encoding='utf-8', engine='python')

            #df_dev = pd.read_csv(path / 'dev.csv', delimiter=",",
            #                     header=0, encoding='utf-8', engine='python')

            df_test = pd.read_csv(path / 'test.csv', delimiter=",",
                                  usecols=["axis", "descriptor", "text", "ppl", "label"],
                                  header=0, encoding='utf-8', engine='python')
        except pd.errors.EmptyDataError:
            logging.error(f'file is empty and has been skipped.')
        #return df_train, df_dev, df_test
        return df_train, df_test


class DatasetReader:
    """Handles dataset reading"""

    def __init__(self):
        pass