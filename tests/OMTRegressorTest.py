
from omt.TreeStructure import RAE,RRSE
from omt.ModelTreeRegressor import optimal_OMT

from sklearn.utils import shuffle
from omt.DatabaseParser import DataParser

ProbType = 'Regression'
file = 'AutoMpg'
Splits = 1

config = {
    'RandomSeed': 7,
    'ProbType': ProbType,
    "ModelTree": True,
    'SplitType': 'Parallel',
    'TestSize': 0.4,
    'df_name': file,
    'Timeout': 60,  # for the single iteration (IN MINUTES
    'ConsoleLog': True
}

df = DataParser(f'{file}.arff', ProbType, one_hot=True)

df = shuffle(df, random_state=config['RandomSeed'])

Test_df = df.iloc[:round(len(df) * config['TestSize'])]
Train_df = df.iloc[len(Test_df):]

# ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
for i in Train_df.columns:
    if Train_df[i].nunique() == 1:
        Train_df = Train_df.drop(columns=[i])
        Test_df = Test_df.drop(columns=[i])

features = list(Train_df.columns.drop(['class']))
labels = df['class'].unique()
labels = ('class', labels)

for C in [1]:  # [0.1, 1, 10, 100]:
    ODT, runtime = optimal_OMT(
        df=Train_df,
        features=features,
        labels=labels,
        Splits=Splits,
        C=C,
        config=config
    )

    print('Runtime:', round(runtime, 3), end=" ")
    print('C:', C, end=' ')
    the_tree = ODT.build_tree(ODT.root.value)
    # ODT.print_tree(the_tree)

    # split train into features and labels
    X_train = Train_df.drop(columns='class')
    X_train = X_train.to_dict('index')
    Y_train = Train_df['class']

    # split test set into features and labels
    X_test = Test_df.drop(columns='class')
    X_test = X_test.to_dict('index')
    Y_test = Test_df['class']

    # Predict the train se
    train_pred = ODT.predict_regr(X_train, the_tree, None)
    print('Train -- RAE:', RAE(Y_train, train_pred), 'RRSE:', RRSE(Y_train, train_pred))

    # Predict the test set
    test_pred = ODT.predict_regr(X_test, the_tree, None)
    print('Test -- RAE:', RAE(Y_test, test_pred), 'RRSE:', RRSE(Y_test, test_pred))



