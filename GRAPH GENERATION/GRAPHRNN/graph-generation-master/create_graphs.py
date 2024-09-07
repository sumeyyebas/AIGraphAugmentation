import networkx as nx
import numpy as np
from torch_geometric.utils.convert import to_networkx

from utils import *
from data import *

import torch
def create(args):
### load datasets
    graphs=torch.load('graphs.pt')
    filtered_graphs = []
    for graph in graphs:
        if graph.y == 1:
            nx_graph = to_networkx(graph)
            filtered_graphs.append(nx_graph)
    print("len filtered graphs : ", len(filtered_graphs))
    return filtered_graphs


