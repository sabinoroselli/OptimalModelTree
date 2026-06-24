problemType = "classification"

if problemType == "classification":
    from sklearn.datasets import load_breast_cancer as df
    X, y = df(return_X_y=True)
elif problemType == "regression":
    from sklearn.datasets import load_diabetes as df
    X, y = df(return_X_y=True)
# elif problemType == "made_up":
#     from sklearn.datasets import make_classification
#     X, y = make_classification(
#         n_samples=80,
#         n_features=5,
#         n_informative=3,
#         n_redundant=0,
#         n_classes=3,
#         random_state=42
#     )

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if problemType == "classification":

    from omt import OptimalModelTreeClassifier
    clf = OptimalModelTreeClassifier(splits=2, timeout=1, solver='SCIP',split_type='Parallel')
    clf.fit(X_train, y_train)
    print("accuracy:", clf.score(X_test, y_test))

elif problemType == "regression":

    from omt import OptimalModelTreeRegressor
    clf = OptimalModelTreeRegressor(splits=1, timeout=1,solver='SCIP',split_type='Oblique')
    clf.fit(X_train, y_train)
    print("r2score:", clf.score(X_test, y_test))



