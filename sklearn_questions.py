"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = check_X_y(X, y)

        # Check if y is for classification (discrete) vs regression
        # Classifiers should reject continuous targets
        y_type = type_of_target(y)
        valid_types = ['binary', 'multiclass', 'multiclass-multioutput',
                       'multilabel-indicator', 'multilabel-sequences']
        if y_type not in valid_types:
            raise ValueError(
                f"Unknown label type: {y_type!r}"
            )

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # Compute pairwise distances between test and training data
        distances = pairwise_distances(
            X, self.X_train_, metric='euclidean'
        )

        # Find k nearest neighbors for each test sample
        n_neighbors = min(self.n_neighbors, len(self.X_train_))
        nearest_indices = np.argsort(distances, axis=1)[:, :n_neighbors]

        # Get labels of nearest neighbors
        nearest_labels = self.y_train_[nearest_indices]

        # Predict using majority vote
        if n_neighbors == 1:
            y_pred = nearest_labels.flatten()
        else:
            # Count votes for each class (handle non-consecutive labels)
            y_pred = np.zeros(len(X), dtype=self.y_train_.dtype)
            for i in range(len(X)):
                labels = nearest_labels[i]
                # Use mode (most frequent value) for tie-breaking
                # In case of tie, sklearn uses the smallest class value
                unique_labels, counts = np.unique(labels, return_counts=True)
                max_count = np.max(counts)
                # Get all labels with max count, then take smallest one
                candidates = unique_labels[counts == max_count]
                # Handle both numeric and non-numeric labels
                if candidates.dtype.kind in ['i', 'u', 'f']:
                    y_pred[i] = np.min(candidates)
                else:
                    # For non-numeric labels, use Python's min
                    # which works on any comparable type
                    y_pred[i] = min(candidates)

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        # Convert to DataFrame if needed (handle Series case)
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        # Get time column or index
        if self.time_col == 'index':
            if not isinstance(X_df.index, pd.DatetimeIndex):
                raise ValueError(
                    "Index must be a DatetimeIndex when time_col='index'"
                )
            time_series = X_df.index.to_series()
        else:
            if self.time_col not in X_df.columns:
                raise ValueError(
                    f"Column '{self.time_col}' not found in X"
                )
            time_series = X_df[self.time_col]
            if not pd.api.types.is_datetime64_any_dtype(time_series):
                raise ValueError(
                    f"Column '{self.time_col}' must be of datetime type"
                )

        # Group by year-month and count unique months
        time_df = pd.DataFrame({'time': time_series})
        time_df['year_month'] = time_df['time'].dt.to_period('M')
        unique_months = time_df['year_month'].unique()
        unique_months = np.sort(unique_months)

        # Number of splits is number of consecutive month pairs
        n_splits = len(unique_months) - 1
        return max(0, n_splits)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        # Convert to DataFrame if needed (handle Series case)
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # Get time column or index
        if self.time_col == 'index':
            if not isinstance(X_df.index, pd.DatetimeIndex):
                raise ValueError(
                    "Index must be a DatetimeIndex when time_col='index'"
                )
            time_series = X_df.index.to_series()
        else:
            if self.time_col not in X_df.columns:
                raise ValueError(
                    f"Column '{self.time_col}' not found in X"
                )
            time_series = X_df[self.time_col]
            if not pd.api.types.is_datetime64_any_dtype(time_series):
                raise ValueError(
                    f"Column '{self.time_col}' must be of datetime type"
                )

        # Create DataFrame with original indices and time
        time_df = pd.DataFrame({
            'original_idx': range(len(X_df)),
            'time': time_series
        })
        time_df['year_month'] = time_df['time'].dt.to_period('M')

        # Sort by time to handle shuffled data
        time_df = time_df.sort_values('time')

        # Group by year-month
        unique_months = time_df['year_month'].unique()
        unique_months = np.sort(unique_months)

        # Generate splits: train on month i, test on month i+1
        n_splits = len(unique_months) - 1
        for i in range(n_splits):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            # Get indices for train and test months
            train_mask = time_df['year_month'] == train_month
            test_mask = time_df['year_month'] == test_month

            idx_train = time_df[train_mask]['original_idx'].values
            idx_test = time_df[test_mask]['original_idx'].values

            # Convert to integer array
            idx_train = np.asarray(idx_train, dtype=np.intp)
            idx_test = np.asarray(idx_test, dtype=np.intp)

            yield idx_train, idx_test
