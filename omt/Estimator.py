from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import r2_score, accuracy_score

from .ModelTreeClassifier import optimal_OMT as OMT_Class
from .ModelTreeRegressor import optimal_OMT as OMT_Regr

from .ModelTreeClassifierOblique import optimal_OMT as OMT_Class_oblique
from .ModelTreeRegressorOblique import optimal_OMT as OMT_Regr_oblique

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
        console_log=True,
        solver="SCIP"
    ):
        self.splits = splits
        self.C = C
        self.timeout = timeout
        self.split_type = split_type
        self.model_tree = model_tree
        self.random_seed = random_seed
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
            "ConsoleLog": self.console_log
        }

        if self.split_type == "Parallel":

            self.tree_, self.runtime_ = OMT_Class(
                df=df,
                features=features,
                labels=labels,
                Splits=self.splits,
                C=self.C,
                config=config,
                solver=self.solver
            )

        elif self.split_type == "Oblique":
            self.tree_, self.runtime_ = OMT_Class_oblique(
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
        self.built_tree_ = self.tree_.build_tree(self.tree_.root)

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

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class OptimalModelTreeRegressor(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            splits=1,
            C=1.0,
            timeout=60,
            split_type="Parallel",
            model_tree=True,
            random_seed=7,
            console_log=True,
            solver="SCIP"
    ):
        self.splits = splits
        self.C = C
        self.timeout = timeout
        self.split_type = split_type
        self.model_tree = model_tree
        self.random_seed = random_seed
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
            "ConsoleLog": self.console_log
        }

        if self.split_type == "Parallel":
            self.tree_, self.runtime_ = OMT_Regr(
                df=df,
                features=features,
                labels=labels,
                Splits=self.splits,
                C=self.C,
                config=config,
                solver=self.solver
            )
        elif self.split_type == "Oblique":
            self.tree_, self.runtime_ = OMT_Regr_oblique(
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
        self.built_tree_ = self.tree_.build_tree(self.tree_.root)

        return self

    def predict(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_dict = X.to_dict("index")

        preds = self.model_.predict_regr(
            X_dict,
            self.built_tree_,
            None,
        )

        return np.asarray(preds)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)