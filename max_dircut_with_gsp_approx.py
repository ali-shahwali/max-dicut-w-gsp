import cvxpy as cp
import os
import scipy as sp
import numpy as np
from typing import List, Tuple
import itertools

"""
    This alogorithm is due to Ageev, Ageev, Hassin and Sviridenko
    see https://epubs.siam.org/doi/pdf/10.1137/S089548010036813X?casa_token=CFKXl1kLnxMAAAAA:rQorxrydVo3UknVN0C627EETeqaJEMuxmzy71eVJFiNGGFERSOKQ9ijD1vVUxw6RgzqIZhTu

    Future comments will refer to this as [AHS01]
"""

excluded_tests = ["test1.txt", "test2.txt"]
np.set_printoptions(precision=16)


class Graph:
    def __init__(self, n: int, edges: List[Tuple[int, int, float]], name: str) -> None:
        self.n = n
        self.edges = edges
        self.name = name


def compute_cut_weight(graph: Graph, solution):
    # Initialize the cut weight.
    cut_weight = 0.0

    # For each edge in the graph, check if it crosses the cut. If it does, add
    # its weight to the cut weight.
    for edge in graph.edges:
        u, v, weight = edge
        if solution[u] == 1 and solution[v] == 0:
            cut_weight += weight

    return cut_weight


def read_input(testsdir="./tests"):
    graphs: List[Graph] = []
    graph_count = 0
    for idx, filename in enumerate(
        filter(lambda f: f.endswith(".txt"), os.listdir(testsdir))
    ):
        if filename in excluded_tests:
            continue
        with open(os.path.join(testsdir, filename), "r") as file:
            graphs.append(Graph(int(file.readline()), [], filename))

            lines = file.readlines()
            for line in lines:
                [i, j, w] = line.split(" ")

                graphs[graph_count].edges.append((int(i), int(j), float(w)))
            graph_count += 1

    return graphs


def approx_max_cut(graph: Graph):
    nr_nodes = graph.n
    edges = graph.edges
    X = cp.Variable((nr_nodes, nr_nodes), symmetric=True)
    constraints = [X >> 0]
    constraints += [X[i, i] == 1 for i in range(nr_nodes)]

    objective = sum(0.5 * (1 - X[i, j]) for (i, j, _) in edges)
    prob = cp.Problem(cp.Maximize(objective), constraints)

    prob.solve()

    x = sp.linalg.sqrtm(X.value)

    # generate random hyper plane
    u = np.random.randn(nr_nodes)

    approximation = np.sign(np.dot(x, u))

    return approximation


def approx_max_dicut_with_gsp(graph: Graph, part_size: int):
    edges = graph.edges
    nr_nodes = graph.n
    X = cp.Variable(nr_nodes)

    # See eq (3.7) [AHS01]
    objective = sum(w * cp.minimum(X[i], 1 - X[j]) for (i, j, w) in edges)

    # See eq (3.4) - (3.5) [AHS01]
    constraints = [
        sum(X[i] for i in range(nr_nodes)) == part_size,
    ]
    constraints += [X[i] >= 0 for i in range(nr_nodes)]
    constraints += [X[i] <= 1 for i in range(nr_nodes)]

    prob = cp.Problem(cp.Maximize(objective), constraints)

    # Find basic feasible solution first
    prob.solve(solver=cp.MOSEK, bfs=True)
    x: np.ndarray = X.value

    # See section 5 [AHS01]
    x_prime = round2(x)
    round2_approx = pipage_rounding(x_prime)
    round1_approx = pipage_rounding(x)

    weight_round1 = compute_cut_weight(graph, round1_approx)
    weight_round2 = compute_cut_weight(graph, round2_approx)

    if weight_round2 > weight_round1:
        return (round2_approx, weight_round2)

    return (round1_approx, weight_round1)


def pipage_rounding(x):
    binary_slt = np.array(x, dtype=np.float64)

    n = binary_slt.size
    satisfy = np.zeros(n, dtype=np.int8)

    # Check if there already exists some solved values
    for i in range(n):
        if binary_slt[i].is_integer():
            satisfy[i] = 1

    for i in range(n - 1):
        binary_slt = round_small_numbers(binary_slt)
        # print(f"NEW SOLUTION:  {binary_slt}")
        # We have a binary solution
        if all([el.is_integer() for el in binary_slt]):
            return binary_slt

        # print("SATISFIED VARS: ", satisfy)
        idx = np.random.choice(np.where(satisfy == 0)[0], 2, replace=False)
        x_select = binary_slt[idx]

        xi, xj = x_select
        epsilon = min(1 - xi, xj)
        if (not np.float32(xi + epsilon).is_integer()) and (
            not np.float32(xj - epsilon).is_integer()
        ):
            epsilon = -min(xi, 1 - xj)

        binary_slt[idx[0]] = xi + epsilon
        binary_slt[idx[1]] = xj - epsilon

        if binary_slt[idx[0]].is_integer():
            satisfy[idx[0]] = 1

        if binary_slt[idx[1]].is_integer():
            satisfy[idx[1]] = 1

    return binary_slt


def round2(x: List[float]):
    v1 = list(filter(lambda xi: xi > 0 and xi < 0.5, x))
    v2 = list(filter(lambda xi: xi > 0.5 and xi < 1, x))
    v3 = list(filter(lambda xi: xi == 0.5, x))
    v4 = list(filter(lambda xi: xi == 0 or xi == 1, x))

    v3_u_v4 = v3 + v4

    delta = 0
    if len(v1) > 0:
        delta = v1[0]
    else:
        delta = (v2[0] - 1) * -1

    x_prime = np.zeros(len(x), dtype=np.float64)

    # See equation (5.3) [AHS01]
    for i in range(len(x)):
        if x[i] in v3_u_v4:
            x_prime[i] = x[i]
        elif x[i] in v1:
            x_prime[i] = min(1, delta + (1 - delta) * len(v2) / len(v1))
        elif x[i] in v2:
            x_prime[i] = max(0, (1 - delta) - (1 - delta) * len(v1) / len(v2))

    return x_prime


def round_small_numbers(numbers, tolerance=1e-6):
    rounded_numbers = []
    for number in numbers:
        if number < tolerance:
            rounded_numbers.append(0)
        elif 1 - number < tolerance:
            rounded_numbers.append(1)
        else:
            rounded_numbers.append(number)

    rounded_numbers = np.array(rounded_numbers, dtype=np.float32)
    return rounded_numbers


def brute_force_max_dicut_with_gsp(graph: Graph, part_size: int):
    permutations = list(
        itertools.permutations([1] * part_size + [0] * (graph.n - part_size))
    )

    opt_weight = 0
    opt_cut = permutations[0]
    for permutation in permutations:
        weight = compute_cut_weight(graph, permutation)
        if weight > opt_weight:
            opt_weight = weight
            opt_cut = permutation

    return (opt_cut, opt_weight)


def pprint_cut(graph: Graph, solution):
    print("Vertice ", end="")
    for i in range(len(solution)):
        if solution[i] == 1:
            print(f"{i}, ", end="")

    print("should be in the cut.")
    cut_weight = compute_cut_weight(graph, solution)
    print(f"The cut has a weight of {cut_weight}")


if __name__ == "__main__":
    graphs = read_input()

    for idx, graph in enumerate(graphs):
        (approx, approx_weight) = approx_max_dicut_with_gsp(graph, 7)
        (opt_solution, opt_weight) = brute_force_max_dicut_with_gsp(graph, 7)

        print(f"---- GRAPH {graph.name} DONE ----")
        print("*** APPROXIMATE SOLUTION ***")
        pprint_cut(graph, approx)
        print("***OPTIMAL SOLUTION***")
        pprint_cut(graph, opt_solution)
        print(f"APPROXIMATION RATIO: {approx_weight / opt_weight}")
        print("------------------------------")
