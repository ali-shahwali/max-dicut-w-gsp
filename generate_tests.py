from max_dircut_with_gsp_approx import Graph
from random import choice, uniform
from math import floor


def generate_test(n, max_weight=10.0, name="test8.txt"):
    graph: Graph = Graph(n, [], name)
    for i in range(n):
        outdegree_cnt = choice([k for k in range(0, floor(n / 2)) if k not in [i]])
        has_edge_to = [i]
        for j in range(outdegree_cnt):
            v = choice([i for i in range(0, n) if i not in has_edge_to])
            has_edge_to.append(v)
            weight = round(uniform(0.0, max_weight), 3)
            graph.edges.append((i, v, weight))

    f = open(f"{graph.name}", "w")

    f.write(f"{graph.n}\n")

    for i, j, w in graph.edges:
        f.write(f"{i} {j} {w}\n")

    f.close()


if __name__ == "__main__":
    generate_test(10)
