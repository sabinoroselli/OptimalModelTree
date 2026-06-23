from sklearn.base import BaseEstimator, ClassifierMixin
from .ModelTreeClassifier import optimal_OMT
import pandas as pd
import numpy as np



class OptimalModelTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        splits=1,
        C=1.0,
        timeout=60,
        split_type="Parallel",
        model_tree=True,
        random_seed=7,
        super_sparse_integers=True,
        meta=False,
        ww=False,
        sum_to_zero=False,
        console_log=True,
        solver="SCIP"
    ):
        self.splits = splits
        self.C = C
        self.timeout = timeout
        self.split_type = split_type
        self.model_tree = model_tree
        self.random_seed = random_seed
        self.super_sparse_integers = super_sparse_integers
        self.meta = meta
        self.ww = ww
        self.sum_to_zero = sum_to_zero
        self.console_log = console_log
        self.solver = solver

    def fit(self, X, y):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # FORCE y INTO 1D
        y = np.asarray(y).ravel()
        df = X.copy()
        df["target"] = y

        features = list(X.columns)
        labels = ("target", np.unique(y))

        config = {
            "RandomSeed": self.random_seed,
            "ProbType": "Classification",
            "ModelTree": self.model_tree,
            "SplitType": self.split_type,
            "Timeout": self.timeout,
            "Meta": self.meta,
            "WW": self.ww,
            "SuperSparseIntegers": self.super_sparse_integers,
            "SumToZero": self.sum_to_zero,
            "ConsoleLog": self.console_log,
            "df_name": "sklearn",
        }

        self.tree_, self.runtime_ = optimal_OMT(
            df=df,
            features=features,
            labels=labels,
            Splits=self.splits,
            C=self.C,
            config=config,
            solver=self.solver
        )

        self.classes_ = np.unique(y)
        self.model_ = self.tree_
        self.built_tree_ = self.tree_.build_tree(self.tree_.root.value)

        return self

    def predict(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_dict = X.to_dict("index")

        preds = self.model_.predict_class(
            X_dict,
            self.built_tree_,
            None,
        )

        return np.asarray(preds)

