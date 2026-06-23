from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from omt.DatabaseParser import DataParser
from omt.Estimator import OptimalModelTreeClassifier

if __name__ == "__main__":
    clf = OptimalModelTreeClassifier(
        splits=1,
        C=1,
        solver= "GUROBI"
    )

    ProbType = 'Classification'
    file = 'blogger'
    df = DataParser(f'{file}.arff', ProbType, one_hot=True)
    df = shuffle(df,random_state=7)

    X = df.loc[ : , df.columns != 'class']
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=7
    )

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)

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
