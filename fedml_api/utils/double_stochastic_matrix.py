import numpy as np
import networkx as nx
import cvxpy as cvx


def asymmetric_fdla_matrix(G, m):
    n = G.number_of_nodes()

    ind = nx.adjacency_matrix(G).toarray() + np.eye(n)
    ind = ~ind.astype(bool)

    average_vec = m / m.sum()
    average_matrix = np.ones((n, 1)).dot(average_vec[np.newaxis, :]).T
    one_vec = np.ones(n)

    W = cvx.Variable((n, n))

    if ind.sum() == 0:
        prob = cvx.Problem(
            cvx.Minimize(cvx.norm(W - average_matrix)),
            [cvx.sum(W, axis=1) == one_vec, cvx.sum(W, axis=0) == one_vec],
        )
    else:
        prob = cvx.Problem(
            cvx.Minimize(cvx.norm(W - average_matrix)),
            [W[ind] == 0, cvx.sum(W, axis=1) == one_vec, cvx.sum(W, axis=0) == one_vec],
        )
    prob.solve()

    W = W.value
    # W = (W + W.T) / 2
    W[ind] = 0
    W -= np.diag(W.sum(axis=1) - 1)
    alpha = np.linalg.norm(W - average_matrix, 2)

    return W, alpha


def symmetric_fdla_matrix(G):
    n = G.number_of_nodes()

    ind = nx.adjacency_matrix(G).toarray() + np.eye(n)
    ind = ~ind.astype(bool)

    average_matrix = np.ones((n, n)) / n
    one_vec = np.ones(n)

    W = cvx.Variable((n, n))

    if ind.sum() == 0:
        prob = cvx.Problem(
            cvx.Minimize(cvx.norm(W - average_matrix)),
            [W == W.T, cvx.sum(W, axis=1) == one_vec],
        )
    else:
        prob = cvx.Problem(
            cvx.Minimize(cvx.norm(W - average_matrix)),
            [W[ind] == 0, W == W.T, cvx.sum(W, axis=1) == one_vec],
        )
    prob.solve()

    W = W.value
    W = (W + W.T) / 2
    W[ind] = 0
    W -= np.diag(W.sum(axis=1) - 1)
    alpha = np.linalg.norm(W - average_matrix, 2)

    return np.array(W), alpha


def generate_expander_graph(self, N, N_selected):
    G = nx.Graph()

    # Add nodes to the graph
    nodes = list(range(N))
    G.add_nodes_from(nodes)

    # Connect each node to N_selected neighbors in a cyclic manner
    for node in nodes:
        for i in range(1, N_selected + 1):
            neighbor = (node + i) % N
            G.add_edge(node, neighbor)

    return G
