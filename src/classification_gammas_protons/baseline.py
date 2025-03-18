from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle

BaselineModel = LogisticRegression | LinearSVC | RandomForestClassifier

def predict(model: BaselineModel, data: pd.DataFrame):
    return model.predict(data)

def evaluate(model: BaselineModel, x: pd.DataFrame, y_true: pd.DataFrame):
    predicted = model.predict(x)    

    return accuracy_score(y_true, predicted)


def train(
        model_type: BaselineModel,
        hyperparams: dict[str, list[float]],
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        features: list[str],
        class_feature: str,
        **kwargs
) -> BaselineModel:
    optimizer = RandomizedSearchCV(
        model_type(**kwargs),
        hyperparams,
        cv=2
    )

    X_train, y_train = train_data[features], train_data[[class_feature]].values.flatten()
    X_validate, y_validate = validation_data[features], validation_data[[class_feature]].values.flatten()

    optimizer.fit(X_train, y_train)
    best_model = optimizer.best_estimator_

    train_predicted = best_model.predict(X_train)
    validate_predicted = best_model.predict(X_validate)

    print(f"Training accuracy: {accuracy_score(train_predicted, y_train)}")
    print(f"Validation accuracy: {accuracy_score(validate_predicted, y_validate)}")

    return best_model

def train_random_forest(
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        features: list[str],
        class_feature: str
) -> RandomForestClassifier:
    return train(
        RandomForestClassifier,
        {
            "max_depth": [5, 10, 20, 40, 75],
            "n_estimators": [10, 20, 50, 100]
        },
        train_data,
        validation_data,
        features,
        class_feature
    )

def train_logistic_regression(
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        features: list[str],
        class_feature: str
) -> LogisticRegression:
    return train(
        LogisticRegression,
        {
            "C": [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
        },
        train_data,
        validation_data,
        features,
        class_feature,
        max_iter=200,
    )

def train_linear_svc(
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        features: list[str],
        class_feature: str,
        additional_features: list[str]=None
) -> SVC:
    X_train = train_data[features]
    X_val = validation_data[features]
    y_train = train_data[[class_feature]].values.flatten()
    y_val = validation_data[[class_feature]].values.flatten()

    pca = PCA(n_components=50)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)


    if additional_features is not None:
        X_train = np.concatenate((X_train, train_data[additional_features].values), axis=1)
        X_val = np.concatenate((X_val, validation_data[additional_features].values), axis=1)

    _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=0.10, stratify=y_train)
    _, X_val, _, y_val = train_test_split(X_val, y_val, test_size=0.10, stratify=y_val)


    optimizer = GridSearchCV(
        LinearSVC(),
        {
            "C": [1e-3, 5e-2, 1, 5e1, 1e3],
            "tol": [1e-4, 1e-3, 1e-2, 1e-1]
        },
        cv=2
    )

    optimizer.fit(X_train, y_train)
    svc = optimizer.best_estimator_

    svc.pca = pca

    print(f"Training accuracy: {accuracy_score(svc.predict(X_train), y_train)}")
    print(f"Validation accuracy: {accuracy_score(svc.predict(X_val), y_val)}")

    return svc

def save(model: BaselineModel, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load(path) -> BaselineModel:
    with open(path, "rb") as f:
        return pickle.load(f)