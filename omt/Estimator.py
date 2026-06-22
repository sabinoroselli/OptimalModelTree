from sklearn.base import BaseEstimator, ClassifierMixin
from .ORToolsClassifier import optimal_OMT
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

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV, cross_val_score
# from sklearn.metrics import accuracy_score
# from sklearn.utils import shuffle
# from DatabaseParser import DataParser

# if __name__ == "__main__":
#     clf = OptimalModelTreeClassifier(
#         splits=1,
#         C=1
#     )
#
#     ProbType = 'Classification'
#     file = 'blogger'
#     df = DataParser(f'{file}.arff', ProbType, one_hot=True)
#     df = shuffle(df,random_state=7)
#
#     X = df.loc[ : , df.columns != 'class']
#     y = df['class']

    # data = load_df(as_frame=True)
    # X = data.data
    # y = data.target

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2,
    #     stratify=y,
    #     random_state=7
    # )
    #
    # clf.fit(X_train,y_train)
    # y_pred = clf.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("Test accuracy:", acc)

    # cross_val_score(clf, X, y, cv=3)
    # grid = GridSearchCV(
    #     OptimalModelTreeClassifier(),
    #     {"C": [0.1, 1, 10], "splits": [1, 2]},
    #     cv=3
    # )
    # print('Grid Search')
    # grid.fit(X, y)
    # y_pred = clf.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("Test accuracy:", acc)