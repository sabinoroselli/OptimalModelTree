import numpy as np
from time import process_time as tm
from ortools.linear_solver import pywraplp
from .BinaryTree import build_full_binary_tree, build_ancestors, collect_descendants
from .TreeStructure import OptimalTree,Parent

def optimal_OMT(df, features, labels, Splits, C, config, solver):


    df = df.reset_index(drop=True)
    I = df.index.values

    classes = df[labels[0]].unique()

    mu = {
        feature: min([abs(first - second)
                      for first, second in zip(df[feature][:-1], df[feature][1:])
                      if second != first
                      ])
        for feature in features
    }

    mu_min = min(mu.values())

    nodes, root, T_B, T_L, P, left, right = build_full_binary_tree(Splits)

    A_l, A_r = build_ancestors(nodes, left, right, nodes)

    D_l = {t: collect_descendants(left(t), left, right, len(nodes)) for t in T_B}
    D_r = {t: collect_descendants(right(t), left, right, len(nodes)) for t in T_B}

    P = {i: (i - 1) // 2 if i != 0 else None for i in nodes}

    m = pywraplp.Solver.CreateSolver(solver)
    if config['ConsoleLog'] == False:
        m.setParam('LogToConsole', 0)
    m.SetTimeLimit(60 * config['Timeout'] * 1000)  # milliseconds

    # =============================================================================

    # VARIABLES

    # =============================================================================

    # d_t = 1 if node t splits
    d = { t: m.IntVar(0, 1, f'd[{t}]') for t in T_B}

    # split variable
    a = {
        (j, t): m.IntVar(0, 1, f'a[{j},{t}]')
        for j in features
        for t in T_B
    }

    # split threshold
    b = {
        t: m.NumVar(-m.infinity(), m.infinity(), f'b[{t}]')
        for t in T_B
    }

    # assignment variables
    z = {
        (i, t): m.IntVar(0, 1, f'z[{i},{t}]')
        for i in I
        for t in T_L
    }

    # leaf active variable
    l = {
        t: m.IntVar(0, 1, f'l[{t}]')
        for t in T_L
    }

    # regression coefficients
    Beta = {
        (f, t): m.NumVar(-m.infinity(), m.infinity(), f'Beta[{f},{t}]')
        for f in features
        for t in T_L
    }

    # absolute value of coefficients
    Bet_abs = {
        (f, t): m.NumVar(0, m.infinity(), f'Bet_abs[{f},{t}]')
        for f in features
        for t in T_L
    }

    # intercept
    Delta = {
        t: m.NumVar(-m.infinity(), m.infinity(), f'Delta[{t}]')
        for t in T_L
    }

    # prediction errors
    e = {
        (i, t): m.NumVar(-m.infinity(), m.infinity(), f'e[{i},{t}]')
        for i in I
        for t in T_L
    }

    # absolute prediction errors
    e_abs = {
        (i, t): m.NumVar(0, m.infinity(), f'e_abs[{i},{t}]')
        for i in I
        for t in T_L
    }

    # =============================================================================

    # CONSTRAINTS

    # =============================================================================

    # Const_1
    for t in T_B:
        m.Add(
            m.Sum(a[j, t] for j in features) == d[t]
        )

    # Const_5
    for t in T_B:
        if t != root:
            m.Add(d[t] <= d[P[t]])

    # Const_6
    for i in I:
        for t in T_L:
            m.Add(z[i, t] <= l[t])

    # Const_7
    for t in T_L:
        m.Add(
            m.Sum(z[i, t] for i in I) >= l[t]
        )

    # Const_8
    for i in I:
        m.Add(
            m.Sum(z[i, t] for t in T_L) == 1
        )

    # =============================================================================
    # BIG-M VALUES FOR INDICATOR CONSTRAINTS
    # =============================================================================

    # You should tighten these if possible
    M_split = len(features) + max(abs(df[features].max().max()),
                                  abs(df[features].min().min())) + 100

    M_split = max(
    max(df[f].max(), abs(df[f].min()))
    for f in features
    )

    x_max = df[features].abs().max().max()
    beta_max = 1.0 / (x_max + 1e-6)
    x_max = df[features].abs().max().max()
    y_max = df[labels[0]].abs().max()

    M_reg = y_max + len(features) * beta_max * x_max

    # =============================================================================
    # Const_12
    # (z[i,l] == 1) => left branch constraint
    # =============================================================================
    for i in I:
        for leaf in T_L:
            for t in A_l[leaf]:
                expr = (
                        m.Sum(
                            a[j, t] *
                            (df.loc[i, j] + mu[j] - mu_min)
                            for j in features
                        )
                        + mu_min
                )

                m.Add(
                    expr <= b[t] + M_split * (1 - z[i, leaf])
                )

    # =============================================================================
    # Const_13
    # (z[i,l] == 1) => right branch constraint
    # =============================================================================
    for i in I:
        for leaf in T_L:
            for t in A_r[leaf]:
                expr = m.Sum(
                    a[j, t] * df.loc[i, j]
                    for j in features
                )
                m.Add(
                    expr >= b[t] - M_split * (1 - z[i, leaf])
                )

    # =============================================================================
    # Const_14
    # =============================================================================
    for t in T_B:
        m.Add(
            d[t] <= m.Sum(l[k] for k in D_l[t])
        )

    # =============================================================================
    # Const_15
    # =============================================================================
    for t in T_B:
        m.Add(
            d[t] <= m.Sum(l[k] for k in D_r[t])
        )

    # =============================================================================
    # Const_16
    # (z[i,t] == 1) => prediction equation
    # =============================================================================
    for i in I:
        for t in T_L:
            pred = (
                    m.Sum(
                        Beta[j, t] * df.loc[i, j]
                        for j in features
                    )
                    + Delta[t]
                    - df.loc[i, labels[0]]
            )
            m.Add(
                pred - e[i, t]
                <= M_reg * (1 - z[i, t])
            )
            m.Add(
                pred - e[i, t]
                >= -M_reg * (1 - z[i, t])
            )

    # =============================================================================
    # Const_17
    # e_abs = |e|
    # =============================================================================
    for i in I:
        for t in T_L:
            m.Add(e_abs[i, t] >= e[i, t])
            m.Add(e_abs[i, t] >= -e[i, t])

    # =============================================================================
    # Const_181
    # Bet_abs = |Beta|
    # =============================================================================
    for f in features:
        for t in T_L:
            m.Add(Bet_abs[f, t] >= Beta[f, t])
            m.Add(Bet_abs[f, t] >= -Beta[f, t])

    # =============================================================================
    # Const_19
    # =============================================================================
    m.Add(
        m.Sum(d[t] for t in T_B) <= Splits
    )

    # =============================================================================
    # OBJECTIVE
    # =============================================================================
    m.Minimize(
        m.Sum(
            Bet_abs[f, t]
            for f in features
            for t in T_L
        )
        +
        C * m.Sum(
            e_abs[i, t]
            for i in I
            for t in T_L
        )
    )

    start = tm()
    status = m.Solve()
    runtime = tm() - start


    if status != pywraplp.Solver.INFEASIBLE:

        vars = m.variables()
        solution = {
                i.name():i.solution_value()
                for i in vars}

        non_zero_vars = [key for key,value in solution.items() if value > 0]

        splitting_nodes = {
            i:{
                'a': [f for f in features if solution[f'a[{f},{i}]'] > 0][0],
                'b': round(solution[f'b[{i}]'],6)
            }
            for i in T_B if f'd[{i}]' in non_zero_vars
        }

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

        ODT = OptimalTree(
            non_empty_nodes,
            splitting_nodes,
            int(np.ceil(np.log2(Splits + 1))),
            config["SplitType"],
            config["ModelTree"],
            classes
        )

    else:
        print('MODEL IS INFEASIBLE')
        ODT = None

    return ODT,runtime

