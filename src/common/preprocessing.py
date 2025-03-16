from typing import Iterator

import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from src.common import FEATURES_TRUE_SHOWER, FEATURES_HILLAS, FEATURES_STEREO


def preprocess(
    data: pd.DataFrame,
    normalize_params=FEATURES_TRUE_SHOWER + FEATURES_HILLAS + FEATURES_STEREO,
    train_portion=0.7,
    validation_portion=0.2,
    stratify_column_name: str = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Split and preprocess telescope data

    Parameters train_portion and validation_portion can be used to adjust the train-validation-test split.
    test_portion is automatically calculated as 1.0 - train_portion - validation_portion.
    """

    assert train_portion + validation_portion < 1.0

    data.dropna(inplace=True)

    stratify = None
    if stratify_column_name is not None:
        stratify = data[[stratify_column_name]]

    train, validation_test = train_test_split(data, train_size=train_portion, stratify=stratify)


    stratify = None
    if stratify_column_name is not None:
        stratify = validation_test[[stratify_column_name]]
    validation, test = train_test_split(
        validation_test, train_size=validation_portion, stratify=stratify
    )

    if len(normalize_params) > 1:
        scaler = StandardScaler()

        train[normalize_params] = scaler.fit_transform(train[normalize_params])
        validation[normalize_params] = scaler.transform(validation[normalize_params])
        test[normalize_params] = scaler.transform(test[normalize_params])

    return train, validation, test
