import math

def build_full_binary_tree(Splits):
    n_nodes = 2 ** (int(math.ceil(math.log2(Splits + 1))) + 1) - 1
    nodes = list(range(n_nodes))

    root = 0

    def left(i): return 2 * i + 1
    def right(i): return 2 * i + 2
    def parent(i): return (i - 1) // 2 if i > 0 else None

    T_B = [i for i in nodes if left(i) < n_nodes]
    T_L = [i for i in nodes if left(i) >= n_nodes]

    P = {i: parent(i) for i in nodes}

    return nodes, root, T_B, T_L, P, left, right

def build_ancestors(nodes, left, right, n_nodes):
    A_l = {i: [] for i in nodes}
    A_r = {i: [] for i in nodes}

    for leaf in nodes:
        i = leaf
        while i != 0:
            p = (i - 1) // 2
            if 2 * p + 1 == i:
                A_l[leaf].append(p)
            else:
                A_r[leaf].append(p)
            i = p

    return A_l, A_r

def collect_descendants(start, left, right, n_nodes):
    stack = [start]
    leaves = []

    while stack:
        u = stack.pop()

        if left(u) >= n_nodes:  # leaf
            leaves.append(u)
        else:
            if left(u) < n_nodes:
                stack.append(left(u))
            if right(u) < n_nodes:
                stack.append(right(u))

    return leaves