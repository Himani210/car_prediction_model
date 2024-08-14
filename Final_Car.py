from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self
    
    def transform(self, X):
        # Apply log transformation to the specified column
        X_copy = X.copy()
        X_copy[self.column_name] = np.log(X_copy[self.column_name].astype(float))
        return X_copy


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        # Compute quantiles and IQR during fit
        self.quantile1, self.quantile3= np.percentile(X[self.column_name], [25, 75])
        self.iqr = self.quantile3 - self.quantile1
        self.lb = self.quantile1 - (1.5 * self.iqr)
        self.ub = self.quantile3 + (1.5 * self.iqr)
        return self
    
    def transform(self, X):
        # Apply the bounds
        X[self.column_name] = np.where(X[self.column_name] < self.lb, self.lb,
                                        np.where(X[self.column_name] > self.ub, self.ub, X[self.column_name]))
        return X


class DebugTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, name):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # print(f"After '{self.name}':")
        # print(X[:5])  # Print the first 5 rows for brevity
        return X
