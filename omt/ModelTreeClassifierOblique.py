import numpy as np
from time import process_time as tm
from ortools.linear_solver import pywraplp
from .BinaryTree import build_full_binary_tree, build_ancestors, collect_descendants
from .TreeStructure import OptimalTree, Parent


def optimal_OMT(df, features, labels, Splits, C, config, solver):
    gamma = 1  # this is the margin of the SVMs

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

    nodes, root, T_B, T_L, P, left, right = build_full_binary_tree(Splits)

    A_l, A_r = build_ancestors(nodes, left, right, nodes)

    D_l = {t: collect_descendants(left(t), left, right, len(nodes)) for t in T_B}
    D_r = {t: collect_descendants(right(t), left, right, len(nodes)) for t in T_B}

    P = {i: (i - 1) // 2 if i != 0 else None for i in nodes}

    m = pywraplp.Solver.CreateSolver(solver)

    if not m:
        raise RuntimeError("SCIP solver not available in OR-Tools")

    m.SetTimeLimit(config["Timeout"] * 60 * 1000)

    # =========================
    # VARIABLES
    # =========================

    d = {t: m.IntVar(0, 1, f"d[{t}]") for t in T_B}

    a = {(j, t): m.NumVar(-m.infinity(), m.infinity(), f"a[{j},{t}]")
         for j in features for t in T_B}

    a_abs = {(j, t): m.NumVar(0, m.infinity(), f"a_abs[{j},{t}]")
             for j in features for t in T_B}

    s = {(j, t): m.IntVar(0, 1, f"s[{j},{t}]")
         for j in features for t in T_B}

    b = {t: m.NumVar(-m.infinity(), m.infinity(), f"b[t]")
         for t in T_B}

    z = {(i, t): m.IntVar(0, 1, f"z[{i},{t}]")
         for i in I for t in T_L}

    l = {t: m.IntVar(0, 1, f"l[{t}]")
         for t in T_L}

    # =========================
    # CLASSIFICATION VARIABLES
    # =========================

    if len(classes) > 2:
        Beta = {(c, j, t): m.NumVar(-m.infinity(), m.infinity(),
                                    f"Beta[{c},{j},{t}]")
                for c in classes for j in features for t in T_L}

        Bet_abs = {(c, j, t): m.NumVar(0, m.infinity(),
                                       f"Beta_abs[{c},{j},{t}]")
                   for c in classes for j in features for t in T_L}

        Delta = {(c, t): m.NumVar(-m.infinity(), m.infinity(),
                                  f"Delta[{c},{t}]")
                 for c in classes for t in T_L}

        e = {(c, i, t): m.NumVar(0, m.infinity(), f"e[{c},{i},{t}]")
             for c in classes for i in I for t in T_L}

    else:
        Beta = {(j, t): m.NumVar(-m.infinity(), m.infinity(),
                                 f"Beta[{j},{t}]")
                for j in features for t in T_L}

        Bet_abs = {(j, t): m.NumVar(0, m.infinity(),
                                    f"Beta_abs[{j},{t}]")
                   for j in features for t in T_L}

        Delta = {t: m.NumVar(-m.infinity(), m.infinity(),
                             f"Delta[{t}]")
                 for t in T_L}

        e = {(i, t): m.NumVar(0, m.infinity(), f"e[{i},{t}]")
             for i in I for t in T_L}

    # =========================
    # CONSTRAINTS
    # =========================

    # a_abs = |a|
    for j in features:
        for t in T_B:
            m.Add(a_abs[j, t] >= a[j, t])
            m.Add(a_abs[j, t] >= -a[j, t])

    # sum constraints (oblique split)
    for t in T_B:
        m.Add(sum(a_abs[j, t] for j in features) <= d[t])
        m.Add(sum(s[j, t] for j in features) >= d[t])
        for j in features:
            m.Add(s[j, t] >= a_abs[j, t])
            m.Add(s[j, t] <= d[t])

    # tree structure
    for t in T_B:
        if t != root:
            m.Add(d[t] <= d[P[t]])

    # assignment constraints
    for i in I:
        m.Add(sum(z[i, t] for t in T_L) == 1)

    for t in T_L:
        m.Add(sum(z[i, t] for i in I) >= l[t])
        for i in I:
            m.Add(z[i, t] <= l[t])

    # routing constraints (left/right split)
    M = max(
        df[j].max() - df[j].min()
        for j in features
    )
    for i in I:
        for lnode in T_L:
            for t in A_l[lnode]:
                expr = sum(a[j, t] * df.loc[i, j] for j in features)
                m.Add(expr + 0.0001 <= b[t] + (1 - z[i, lnode]) * M)

            for t in A_r[lnode]:
                expr = sum(a[j, t] * df.loc[i, j] for j in features)
                m.Add(expr >= b[t] - (1 - z[i, lnode]) * M)

    # leaf non-empty constraints
    for t in T_B:
        m.Add(d[t] <= sum(l[m] for m in D_l[t]))
        m.Add(d[t] <= sum(l[m] for m in D_r[t]))

    # classification loss constraints
    if len(classes) > 2:
        for i in I:
            for t in T_L:
                for c in classes:
                    expr = sum(Beta[c, j, t] * df.loc[i, j] for j in features) + Delta[c, t]

                    m.Add(
                        e[c, i, t] >= gamma - LabelsPerClass[c][i] * expr
                    )

        for c in classes:
            for j in features:
                for t in T_L:
                    m.Add(Bet_abs[c, j, t] >= Beta[c, j, t])
                    m.Add(Bet_abs[c, j, t] >= -Beta[c, j, t])

    else:
        for i in I:
            for t in T_L:
                expr = sum(Beta[j, t] * df.loc[i, j] for j in features) + Delta[t]
                m.Add(e[i, t] >= gamma - df.loc[i, labels[0]] * expr)

        for j in features:
            for t in T_L:
                m.Add(Bet_abs[j, t] >= Beta[j, t])
                m.Add(Bet_abs[j, t] >= -Beta[j, t])

    # split limit
    m.Add(sum(d[t] for t in T_B) <= Splits)

    # =========================
    # OBJECTIVE
    # =========================

    if len(classes) > 2:
        m.Minimize(
            sum(Bet_abs[c, j, t] for c in classes for j in features for t in T_L)
            +
            C * sum(e[c, i, t] for c in classes for i in I for t in T_L)
        )
    else:
        m.Minimize(
            sum(Bet_abs[j, t] for j in features for t in T_L)
            +
            C * sum(e[i, t] for i in I for t in T_L)
        )

    # =========================
    # SOLVE
    # =========================

    start = tm()
    status = m.Solve()
    runtime = tm() - start

    splitting_nodes = {}

    if status != pywraplp.Solver.INFEASIBLE:
        vars = m.variables()
        solution = {
            i.name(): i.solution_value()
            for i in vars}

        non_zero_vars = [key for key, value in solution.items() if value > 0]

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
                i: {
                    c: {
                        'Beta': {
                            j: round(solution[f'Beta[{c},{j},{i}]'], 6)
                            for j in features
                        },
                        'Delta': round(solution[f'Delta[{c},{i}]'], 6)
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

    return ODT, runtime

