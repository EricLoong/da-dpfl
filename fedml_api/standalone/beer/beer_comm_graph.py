import numpy as np
import networkx as nx

import fedml_api.utils.logger as logging_util


class CommunicationGraph:
    def __init__(self, world_size, graph_type="er", graph_params=None):
        logger = logging_util.Logger()
        self.log = logger.get_logger()
        self.log.info(f"Using {graph_type} graph")

        self.world_size = world_size
        self.graph_type = graph_type
        self.graph_params = graph_params

        self.graph = self.generate_graph(
            graph_type=graph_type, graph_params=graph_params
        )
        # if dist.is_initialized():
        #     self.process_group = self.create_process_group(self.graph)

    def has_predecessor(self, u, v):
        return self.graph.has_predecessor(u, v)

    def generate_graph(self, graph_type="expander", graph_params=(5,)):

        if graph_type == "er":
            connected = False
            while connected is False:
                G = nx.erdos_renyi_graph(self.world_size, graph_params[0], seed=0)
                connected = nx.is_connected(G)
        elif graph_type == "ring":
            G = nx.cycle_graph(self.world_size)
        elif graph_type == "complete":
            G = nx.complete_graph(self.world_size)
        elif graph_type == "expander":
            G = nx.Graph()
            # Add nodes to the graph
            nodes = list(range(self.world_size))
            G.add_nodes_from(nodes)

            # Connect each node to N_selected neighbors in a cyclic manner
            for node in nodes:
                for i in range(1, int(graph_params[0]) + 1):
                    neighbor = (node + i) % self.world_size
                    G.add_edge(node, neighbor)
        else:
            raise NotImplementedError(f"Graph type {graph_type} not implemented")

        adjacency_matrix = nx.adjacency_matrix(G).toarray() + np.eye(self.world_size)
        self.log.info(str(adjacency_matrix))
        return G

    def generate_mixing_matrix(self, adj_matrix):
        """
        Generate a symmetric matrix with the same column sums from the adjaciancy matrix.
        """
        mixing_matrix = adj_matrix.astype(float)
        mixing_matrix /= mixing_matrix.sum(axis=1)[0]
        return mixing_matrix

    def neighbors(self, *args, **kwargs):
        return self.graph.neighbors(*args, **kwargs)

    # def create_process_group(self, graph):
    #     group = []
    #     for rank in range(self.world_size):
    #         neighbors = list(graph.neighbors(rank))
    #         log.debug('creating %d\'s predecessoe group from %s',
    #                   rank, neighbors + [rank])
    #         group.append(dist.new_group(ranks=neighbors + [rank]))
    #         # log.debug('%d\'s predecessor group created from %s',
    #                   # rank, predecessors)
    #         log.info(f'process group {rank} created')
    #     return group

    def draw(self):
        nx.draw_circular(self.graph)
