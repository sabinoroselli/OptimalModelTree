from omt.DatabaseParser import DataParser
from omt.ModelTreeClassifier import optimal_OMT

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

if __name__ == "__main__":

    ProbType = 'Classification'
    TestSize = 0.2
    file = 'blogger'
    Splits = 1

    config ={
        'RandomSeed':7,
        'SplitType': 'Parallel',
        'label_name': 'class',
        'Timeout': 60,  # for the single iteration (IN MINUTES)
        'ConsoleLog':False
    }

    df = DataParser(f'{file}.arff',ProbType, one_hot=True)

    df = shuffle(df,random_state=config['RandomSeed'])

    Test_df = df.iloc[:round(len(df) * TestSize)]
    Train_df = df.iloc[len(Test_df):]

    # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
    for i in Train_df.columns:
        if Train_df[i].nunique() == 1:
            Train_df = Train_df.drop(columns=[i])
            Test_df = Test_df.drop(columns=[i])

    features = list(Train_df.columns.drop(['class']))
    labels = df['class'].unique()
    labels = ('class', labels)

    for C in [1]:#[0.1, 1, 10, 100]:
        ODT,runtime = optimal_OMT(
            df= Train_df,
            features= features,
            labels= labels,
            Splits= Splits,
            C= C,
            config=config
        )

        print('Runtime:',round(runtime,3),end=" ")
        print('C:',C,end=' ')
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

        # Predict the train set
        train_pred = ODT.predict_class(X_train, the_tree, None)
        print('Train:', round(accuracy_score(Y_train, train_pred) * 100, 2), '%',end=' ')

        # Predict the test set
        test_pred = ODT.predict_class(X_test, the_tree,None)
        print('Test:', round(accuracy_score(Y_test, test_pred)*100,2),'%')

