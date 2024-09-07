import logging
import sys
from creator.ModelDataset import ModelDataset
from torch.utils.data import Dataset
import torch.optim as optim
import time
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.metrics import classification_report
import torch.nn.functional as F
import torch
sys.path.append("..")
from .Creator import Creator
from torch.nn import (BatchNorm1d as BN, ReLU, Sequential, Linear)
from torch_geometric.nn import (
    SAGEConv, global_add_pool, GraphConv, TopKPooling, global_mean_pool, AttentionalAggregation,
    GCNConv, GINConv, JumpingKnowledge, EdgePooling, Set2Set
)


class ModelCreator(Creator):

    def __init__(self, args, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data
        self.best_models=[]
        self.args = args
    
    def train(self, args):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        log_file = open('datasets/'+args.training+'/logs.txt', "w")
        models = [GraphSAGE(),  GIN0(), GINWithJK()]

        
        for model in models:
            best_val_accuracy, best_loss = 0.0, 100
            best_model = None
            logging.info(f"{model}")
            log_file.write(f"{model}")
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = torch.nn.CrossEntropyLoss()
            train_dataset = ModelDataset(self.training_data)
            train_generator = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True) #self.args.batch_size
            best_model_state_dict = None

            for ep in range(self.args.epoch):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                start_time = time.time()
                for local_batch in train_generator:
                    out = model(local_batch.x, local_batch.edge_index, local_batch.batch)
                    loss = criterion(out, local_batch.y)
                    total_loss += loss.item() / self.args.batch_size
                    _, predicted = torch.max(out, 1)
                    total += local_batch.y.size(0)
                    correct += (predicted == local_batch.y).sum().item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if loss < best_loss:
                        best_loss = loss
                        best_model = model
                        best_model_state_dict = model.state_dict()
                        #log_file.write(f"best train loss {model}: {best_loss}")

                # Training accuracy
                train_accuracy = correct / total
                end_time = time.time()
                epoch_time = (end_time - start_time) / 60
                logging.info('Epoch: %d | Training Loss: %.4f | Training Accuracy: %.4f | Execution Time: %.4f mins', ep, total_loss, train_accuracy, epoch_time)
                log_file.write('Epoch: %d | Training Loss: %.4f | Training Accuracy: %.4f  | Execution Time: %.4f mins\n' % (ep, total_loss, train_accuracy, epoch_time))
            self.best_models.append(best_model)
                
        log_file.close()

    def parse_classification_report(report):
        lines = report.split('\n')
        data = {}
        for line in lines[2:-3]:
            line_data = line.strip().split()
            if len(line_data) == 0:
                continue
            class_name = line_data[0]
            scores = [float(score) for score in line_data[1:]]
            data[class_name] = scores
        return data
    


    def test(self):
        print(self.best_models)
        for model in self.best_models:
            model.eval()
            log_file = open("datasets/"+self.args.training+"/logs.txt", "a")
            y_true = []
            y_pred = []

            correct = 0
            total_samples = 0

            test_set = ModelDataset(self.test_data)
            test_generator = DataLoader(test_set, batch_size=32)

            for data in test_generator:  
                out = model(data.x, data.edge_index, data.batch)  
                pred = out.argmax(dim=1)
                y_true.extend(data.y.tolist())
                y_pred.extend(pred.tolist())
                correct += int((pred == data.y).sum())
                total_samples += len(data.y)

            accuracy = correct / total_samples
            log_file.write(f"Model : {model}\n")
            log_file.write(f"Test Accuracy : {accuracy}")

            logging.info(f"The model used for testing : {model}")
            logging.info(f"Test Accuracy : {accuracy}")

            model_scores_test = classification_report(y_true, y_pred)
            log_file.write(model_scores_test)
            logging.info(model_scores_test)
            log_file.close()




class GIN0(torch.nn.Module):
    def __init__(self, hidden=64, num_layers=8):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(1, hidden),
                ReLU(),
                BN(hidden),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GINWithJK(torch.nn.Module):
    def __init__(self, hidden=64, mode='cat', num_layers=8):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(1, hidden),
                ReLU(),
                BN(hidden),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GCNWithJK(torch.nn.Module):
    def __init__(self, hidden=64, mode='cat', num_layers=8):
        super().__init__()
        self.conv1 = GCNConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self,  x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden=64, num_layers=8):
        super().__init__()
        self.conv1 = SAGEConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__



class TopK(torch.nn.Module):
    def __init__(self, hidden=64, ratio=0.8, num_layers=8):
        super().__init__()
        self.conv1 = GraphConv(1, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [TopKPooling(hidden, ratio) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GlobalAttentionNet(torch.nn.Module):
    def __init__(self, hidden=64, num_layers=16):
        super().__init__()
        self.conv1 = SAGEConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.att = AttentionalAggregation(Linear(hidden, 1))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.att(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class EdgePool(torch.nn.Module):
    def __init__(self, hidden=64, num_layers=16):
        super().__init__()
        self.conv1 = GraphConv(1, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class Set2SetNet(torch.nn.Module):
    def __init__(self, hidden=64, num_layers=16):
        super().__init__()
        self.conv1 = SAGEConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.set2set = Set2Set(hidden, processing_steps=4)
        self.lin1 = Linear(2 * hidden, hidden)
        self.lin2 = Linear(hidden, 3)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.set2set.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.set2set(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
