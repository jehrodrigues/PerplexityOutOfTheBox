import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.read_dataset import get_data
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import ColumnTransformer
from src.data.text_preprocessing import convert_labels

project_dir = Path(__file__).resolve().parents[2]


class BaselinePredict:
    """
    Provides a classic baseline for comparison
    """

    def __init__(self, model_name, data_source):
        self._data = data_source
        self._model = self.train(model_name)

    def my_predict(self):
        """Predict the binary class of a sentence using a Logistic Regression
        Args:
            Data (pd.DataFrame): DataFrame
        Returns:
            binary class (str): Apartment (class 0) | House (class 1)
        """
        # predict
        return self._model

    def train(self, model) -> str:
        """Train a logistic regression method"""
        #try:

        # Get data
        df_train, _ = get_data(self._data)

        # Removing label
        train = df_train.drop(["label"], axis=1)
        #train = train.drop(["ppl"], axis=1)

        # Numerical transformations
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ])
        # Categorical transformations
        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        # Text transformations
        text_transformer = Pipeline(steps=[
            ('vect', CountVectorizer())
            #('tfidf', TfidfTransformer())
        ])

        # Numerical features
        # numeric_features = train.select_dtypes(include=['int64', 'float64']).columns

        # Categorical features
        categorical_features = train.select_dtypes(include=['object']).columns

        # Convert in a column
        preprocessor = ColumnTransformer(
            transformers=[
                #('num', numeric_transformer, 'ppl'),
                ('tex', text_transformer, 'text'),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('lr', LogisticRegression(max_iter=1000, multi_class="multinomial")) #solver="sag",
        ])

        # labels
        label2int = convert_labels(df_train["label"])
        train_labels = df_train['label'].apply(lambda x: label2int[x])

        # fit
        pipeline.fit(train, train_labels)
        print(pipeline.feature_names_in_)

        return pipeline

        #except Exception:
        #    logging.error(f'directory or model is invalid or does not exist: {self._model}')