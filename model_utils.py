import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)

def linear_model(data_frame, target, predictors):
    X = data_frame[predictors]

    # Measurement we're trying to predict
    y = data_frame[target]

    # Physical tooth grouping
    groups = data_frame["Specimen #"]

    # Cannot train a model to predict a measurement if measurement is missing
    model_df = data_frame[data_frame[target].notna()].copy()

    # 1. Fill missing predictor values with the median
    # 2. Fit ordinary linear regression
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regression", LinearRegression())
    ])

    # 5 fold cross validation
    cv = GroupKFold(n_splits=5)
    predictions = cross_val_predict(model, X, y, cv=cv, groups=groups)

    # Evaluate model performance
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    print("Target measurement:", target)
    print("Rows used:", len(model_df))
    print("Physical tooth groups:", groups.nunique())

    print()

    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    return (predictions, y)


def plot_model(target, predictions, y):
    # Chat GPT used to generate the following code for plotting actual vs predicted values

    plt.scatter(
        y,
        predictions
    )

    minimum = min(
        y.min(),
        predictions.min()
    )

    maximum = max(
        y.max(),
        predictions.max()
    )


    # Perfect-prediction line
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        f"Actual {target}"
    )

    plt.ylabel(
        f"Predicted {target}"
    )

    plt.title(
        f"Linear Regression: Predicting {target}"
    )

    plt.show()