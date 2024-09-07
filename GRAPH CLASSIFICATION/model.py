import logging
from creator.CreatorFactory import CreatorFactory
import torch
import random
import networkx as nx
from torch_geometric.utils import from_networkx
import pickle


def load_graphs_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs
def load_labels_from_txt(test_label_path):
    with open(test_label_path, "r") as f:
        labels = f.readlines()
    return labels

class Model:
    _instance = None
    training_data = None
    test_data = None

    def __new__(cls):
        if cls._instance is None:
            logging.info('Creating the Model')
            cls._instance = super(Model, cls).__new__(cls)
        else:
            logging.info('Model already exist')
        return cls._instance

    def initialize(self, args):
        logging.info('Model initializing')
        logging.info('Data loading')
        self.training_data = torch.load('datasets/'+args.training+'/graphs.pt')
        self.test_data = torch.load('datasets/e/graphs.pt')
        logging.info('Data loaded')
        self.creator = CreatorFactory.get_creator(args, self.training_data, self.test_data)
        self.args = args
        logging.info('Model initialized')


    def train(self, args):
        logging.info('Training Started')
        self.creator.train(args)
        logging.info('Training Finished')

    def test(self):
        logging.info('Testing Started')
        self.creator.test()
        # self.evaluate()
        logging.info('Testing Finished')








