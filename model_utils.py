import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error)
from sklearn.base import BaseEstimator, TransformerMixin


class TaxonMedianImputer(BaseEstimator, TransformerMixin):

    def __init__(self, predictors):
        self.predictors = predictors

    def fit(self, X, y=None):
        # Calculate medians using TRAINING data only
        self.taxon_medians_ = (X.groupby("Taxon")[self.predictors].median())

        self.overall_medians_ = X[self.predictors].median()

        # Check for completely missing predictor columns
        if self.overall_medians_.isna().any():
            raise ValueError("A predictor has no observed values in this fold.")

        return self

    def transform(self, X):
        result = X[self.predictors].copy()

        for col in self.predictors:
            # Taxon-specific median from training data
            taxon_values = X["Taxon"].map(self.taxon_medians_[col])

            result[col] = (
                result[col]
                .fillna(taxon_values)
                .fillna(self.overall_medians_[col])
            )

        return result


def linear_model(data_frame, target, predictors):

    # Cannot train a model to predict a measurement
    # if the target measurement is missing
    model_df = data_frame[data_frame[target].notna()].copy()

    # Predictor measurements and taxon for imputation
    X = model_df[list(predictors) + ["Taxon"]]

    # Measurement we're trying to predict
    y = model_df[target]

    # Physical tooth grouping
    groups = model_df["Specimen #"]

    # 1. Fill missing predictor values with taxon median
    # 2. Fit ordinary linear regression
    
    model = Pipeline([
        ("imputer", TaxonMedianImputer(predictors)),
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

    return (model, predictions, y)


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