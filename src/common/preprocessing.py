from typing import Iterator

import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from src.common import PARAMS_TRUE_SHOWER, PARAMS_HILLAS, PARAMS_STEREO


def preprocess(
    data: pd.DataFrame,
    normalize_params=PARAMS_TRUE_SHOWER + PARAMS_HILLAS + PARAMS_STEREO,
    train_portion=0.7,
    validation_portion=0.2,
    **kwargs
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Split and preprocess telescope data

    Parameters train_portion and validation_portion can be used to adjust the train-validation-test split.
    test_portion is automatically calculated as 1.0 - train_portion - validation_portion.
    """

    assert train_portion + validation_portion < 1.0

    data.dropna(inplace=True)

    train, validation_test = train_test_split(data, train_size=train_portion, **kwargs)
    validation, test = train_test_split(
        validation_test, train_size=validation_portion, **kwargs
    )

    if len(normalize_params) > 1:
        scaler = StandardScaler()

        train[normalize_params] = scaler.fit_transform(train[normalize_params])
        validation[normalize_params] = scaler.transform(validation[normalize_params])
        test[normalize_params] = scaler.transform(test[normalize_params])

    return train, validation, test
