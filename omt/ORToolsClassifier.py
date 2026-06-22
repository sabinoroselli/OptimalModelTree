import numpy as np
from time import process_time as tm
from ortools.linear_solver import pywraplp
from binarytree import build
from .TreeStructure import OptimalTree,Parent


def optimal_OMT(df, features, labels, Splits, C, config):

    gamma = 1 # this is the margin of the SVMs

    df = df.reset_index(drop=True)
    I = df.index.values

    classes = df[labels[0]].unique()

    LabelsPerClass = {
        c: {
            i: 1 if df.loc[i, labels[0]] == c else -1
            for i in I
        }
        for c in classes
    }

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }

    mu_min = min(mu.values())

    # depth of the tree DOES NOT include root level
    nodes = [i for i in range(2 ** (int(np.ceil(np.log2(Splits + 1))) + 1) - 1)]
    binary_tree = build(nodes)
    root = binary_tree.levels[0][0]

    # print(binary_tree)

    T_L = [i.value for i in binary_tree.leaves]  # leave nodes
    T_B = [i for i in binary_tree.values if i not in T_L] # branch nodes

    A_l = {
        i: [j.value for j in list(root) if j != i and j.left != None and i in j.left.values] for i in binary_tree.values
    }

    A_r = {
        i: [j.value for j in list(root) if j != i and j.left != None and i in j.right.values] for i in
        binary_tree.values
    }

    D_l = {
            i : [k.value for k in j.left.leaves]
            for i in T_B
            for j in list(root)
            if j.value == i
            }

    D_r = {
        i: [k.value for k in j.right.leaves]
        for i in T_B
        for j in list(root)
        if j.value == i
    }


    P = {
        i: Parent(root, i) for i in binary_tree.values
    }

    m = pywraplp.Solver.CreateSolver("SCIP")

    if m is None:
        raise RuntimeError("SCIP not available")

    INF = m.infinity()

    # m.setParam('Threads',1)
    m.SetTimeLimit( int(config["Timeout"] * 60 * 1000) )

    feature_bounds = {}

    for f in features:
        mn = float(df[f].min())

        mx = float(df[f].max())

        feature_bounds[f] = (mn, mx)

    M = (
        sum(max(abs(df[f].min()), abs(df[f].max())) for f in features)
        +
        max(abs(df[f].min()), abs(df[f].max()))
    )

    # ==================================================
    # VARIABLES
    # ==================================================

    d = { t: m.BoolVar(f"d[{t}]") for t in T_B }

    # Parallel split variables
    a = {
        (f, t): m.BoolVar(f"a[{f},{t}]")
          for f in features
        for t in T_B
    }

    b = {
        t: m.NumVar(-INF, INF, f"b[{t}]")
        for t in T_B
    }

    z = {
        (i, l): m.BoolVar(f"z[{i},{l}]")
        for i in I
        for l in T_L
    }

    lvar = {
        t: m.BoolVar(f"l[{t}]")
        for t in T_L
    }

    # ==================================================
    # LEAF SVM
    # ==================================================

    binary_case = len(classes) == 2

    if binary_case:
        Beta = {
            (f, t): m.NumVar(-INF, INF, f"Beta[{f},{t}]")
            for f in features
            for t in T_L
        }

        Bet_abs = {
            (f, t): m.NumVar(0, INF, f"Bet_abs[{f},{t}]")
            for f in features
            for t in T_L
        }

        Delta = {
            t: m.NumVar(-INF, INF, f"Delta[{t}]")
            for t in T_L
        }

        e = {
            (i, t): m.NumVar(0, INF, f"e[{i},{t}]")
            for i in I
            for t in T_L
        }

    else:
        Beta = {
            (c, f, t): m.NumVar(
                -INF,
                INF,
                f"Beta[{c},{f},{t}]"
            )
            for c in classes
            for f in features
            for t in T_L
        }
        Bet_abs = {
            (c, f, t): m.NumVar(
                0,
                INF,
                f"Bet_abs[{c},{f},{t}]"
            )
            for c in classes
            for f in features
            for t in T_L
        }
        Delta = {
            (c, t): m.NumVar(
                -INF,
                INF,
                f"Delta[{c},{t}]"
            )
            for c in classes
            for t in T_L
        }
        e = {
            (c, i, t): m.NumVar(
                0,
                INF,
                f"e[{c},{i},{t}]"
            )
            for c in classes
            for i in I
            for t in T_L
        }

    # ==================================================
    # SPLIT STRUCTURE
    # ==================================================

    for t in T_B:
        m.Add(
            sum(a[f, t] for f in features)
            == d[t]
        )
    for t in [i for i in T_B if i != root.value]:
        m.Add(d[t] <= d[P[t]])

    # ==================================================
    # LEAF OCCUPANCY
    # ==================================================

    for t in T_L:
        for i in I:
            m.Add(z[i, t] <= lvar[t])

    for t in T_L:
        m.Add(
            sum(z[i, t] for i in I)
            >= lvar[t]
        )
    for i in I:
        m.Add(
            sum(z[i, t] for t in T_L)
            == 1
        )

    # ==================================================
    # ROUTING CONSTRAINTS
    # Indicator -> Big-M
    # ==================================================

    for i in I:
        for leaf in T_L:
            # LEFT
            for t in A_l[leaf]:
                m.Add(
                    sum(
                        a[f, t]
                        * (df.loc[i, f] + mu[f] - mu_min)
                        for f in features
                    )
                    + mu_min
                    <=
                    b[t] + M * (1 - z[i, leaf])
                )

            # RIGHT
            for t in A_r[leaf]:
                m.Add(
                    sum( a[f, t] * df.loc[i, f] for f in features )
                    >=
                    b[t] - M * (1 - z[i, leaf])
                )

    # ==================================================

    # ACTIVE SPLIT => NONEMPTY DESCENDANTS

    # ==================================================

    for t in T_B:
        m.Add(
            d[t] <= sum( lvar[k] for k in D_l[t] )
        )

        m.Add(
            d[t] <= sum( lvar[k] for k in D_r[t] )
        )

    # ==================================================
    # LEAF CLASSIFIERS
    # ==================================================

    if binary_case:
        # assumes labels encoded {-1,+1}
        for i in I:
            yi = float(df.loc[i, labels[0]])
            for t in T_L:
                svm_expr = (

                        sum(

                            Beta[f, t]

                            * df.loc[i, f]

                            for f in features

                        )

                        + Delta[t]

                )

                m.Add(

                    gamma

                    - e[i, t]

                    <= svm_expr * yi

                    + M * (1 - z[i, t])

                )

        # abs(Beta)

        for f in features:

            for t in T_L:
                m.Add(

                    Bet_abs[f, t]

                    >= Beta[f, t]

                )

                m.Add(

                    Bet_abs[f, t]

                    >= -Beta[f, t]

                )

    else:

        for c in classes:

            for i in I:

                for t in T_L:
                    svm_expr = (

                            sum(

                                Beta[c,f,t]

                                * df.loc[i, f]

                                for f in features

                            )

                            + Delta[c,t]

                    )

                    m.Add(

                        gamma

                        - e[c,i,t]

                        <= svm_expr

                        * LabelsPerClass[c][i]

                        + M * (1 - z[i, t])

                    )

        for c in classes:

            for f in features:

                for t in T_L:
                    m.Add(

                        Bet_abs[c,f,t]

                        >= Beta[c,f,t]

                    )

                    m.Add(

                        Bet_abs[c,f,t]

                        >= -Beta[c,f,t]

                    )

    # ==================================================
    # SPLIT BUDGET
    # ==================================================

    m.Add( sum(d[t] for t in T_B) <= Splits )

    # ==================================================
    # OBJECTIVE
    # ==================================================

    if binary_case:

        objective = (

                sum( Bet_abs[f, t] for f in features for t in T_L )
                +
                C * sum( e[i, t]

            for i in I

            for t in T_L

        )

        )

    else:

        objective = (

                sum(

                    Bet_abs[c,f,t]

                    for c in classes

                    for f in features

                    for t in T_L

                )

                + C * sum( e[c,i,t] for c in classes for i in I for t in T_L )
        )

    m.Minimize(objective)

    start = tm()
    status = m.Solve()
    runtime = tm() - start

    splitting_nodes = {}

    if status != pywraplp.Solver.INFEASIBLE:
        vars = m.variables()
        solution = {
                i.name():i.solution_value()
                for i in vars}

        non_zero_vars = [key for key,value in solution.items() if value > 0]

        if config["SplitType"] == "Parallel":
            splitting_nodes = {
                i:{
                    'a': [f for f in features if solution[f'a[{f},{i}]'] > 0][0],
                    'b': round(solution[f'b[{i}]'],6)
                }
                for i in T_B if f'd[{i}]' in non_zero_vars
            }
        elif config["SplitType"] == "Oblique":
            splitting_nodes = {
                i: {
                    'a': {f: round(solution[f'a[{f},{i}]'], 6)
                          for f in features
                          },
                    'b': round(solution[f'b[{i}]'], 6)
                }
                for i in T_B if f'd[{i}]' in non_zero_vars
            }
        if len(classes) == 2:
            non_empty_nodes = {
                i: {
                    'Beta': {
                        j: round(solution[f'Beta[{j},{i}]'], 6)
                        for j in features
                    },
                    'Delta': round(solution[f'Delta[{i}]'], 6)
                }
                for i in T_L if f'l[{i}]' in non_zero_vars
            }
        else:
            non_empty_nodes = {
                i:{
                    c:{
                        'Beta':{
                            j: round(solution[f'Beta[{c},{j},{i}]'],6)
                            for j in features
                        },
                        'Delta':round(solution[f'Delta[{c},{i}]'],6)
                    }
                    for c in classes
                }
                for i in T_L if f'l[{i}]' in non_zero_vars
            }

        ODT = OptimalTree(
            non_empty_nodes,
            splitting_nodes,
            int(np.ceil(np.log2(Splits + 1))),
            config["SplitType"],
            True,
            classes
        )

    else:
        print('MODEL IS INFEASIBLE')
        ODT = None

    return ODT,runtime

# from sklearn.metrics import accuracy_score
# from sklearn.utils import shuffle
# from DatabaseParser import DataParser
# if __name__ == "__main__":
#
#     ProbType = 'Classification'
#     TestSize = 0.2
#     file = 'blogger'
#     Splits = 1
#
#     config ={
#         'RandomSeed':7,
#         'SplitType': 'Parallel',
#         'label_name': 'class',
#         'Timeout': 60,  # for the single iteration (IN MINUTES)
#         'ConsoleLog':False
#     }
#
#     df = DataParser(f'{file}.arff',ProbType, one_hot=True)
#
#     df = shuffle(df,random_state=config['RandomSeed'])
#
#     Test_df = df.iloc[:round(len(df) * TestSize)]
#     Train_df = df.iloc[len(Test_df):]
#
#     # ELIMIATING A COLUMN FROM ALL DATASETS IF ALL THE VALUES IN IT ARE THE SAME IN THE TRAIN SET
#     for i in Train_df.columns:
#         if Train_df[i].nunique() == 1:
#             Train_df = Train_df.drop(columns=[i])
#             Test_df = Test_df.drop(columns=[i])
#
#     features = list(Train_df.columns.drop(['class']))
#     labels = df['class'].unique()
#     labels = ('class', labels)
#
#     for C in [1]:#[0.1, 1, 10, 100]:
#         ODT,runtime = optimal_OMT(
#             df= Train_df,
#             features= features,
#             labels= labels,
#             Splits= Splits,
#             C= C,
#             config=config
#         )
#
#         print('Runtime:',round(runtime,3),end=" ")
#         print('C:',C,end=' ')
#         the_tree = ODT.build_tree(ODT.root.value)
#         # ODT.print_tree(the_tree)
#
#         # split train into features and labels
#         X_train = Train_df.drop(columns='class')
#         X_train = X_train.to_dict('index')
#         Y_train = Train_df['class']
#
#         # split test set into features and labels
#         X_test = Test_df.drop(columns='class')
#         X_test = X_test.to_dict('index')
#         Y_test = Test_df['class']
#
#         # Predict the train set
#         train_pred = ODT.predict_class(X_train, the_tree, None)
#         print('Train:', round(accuracy_score(Y_train, train_pred) * 100, 2), '%',end=' ')
#
#         # Predict the test set
#         test_pred = ODT.predict_class(X_test, the_tree,None)
#         print('Test:', round(accuracy_score(Y_test, test_pred)*100,2),'%')



