import copy

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer: optim, train_loader: DataLoader, test_loader: DataLoader,
                 device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.acc_history = []
        self.loss_history = []

    def process(self, epoch):
        # the major part of training process of all model
        self.model.train()
        epoch_loss = []
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            outs = self.model(x)
            loss = self.criterion(outs, y.long())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.item())
        tensor = torch.tensor(epoch_loss)
        average_loss = torch.mean(tensor)
        return average_loss

    def train(self, epochs):
        # the most ordinary training process
        self.model.to(self.device)
        for epoch in range(epochs):
            average_loss = self.process(epoch)
            print('Epoch:{}, Loss:{}'.format(epoch, average_loss))

    def eval_probability(self):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted')

        return accuracy, precision, recall, f1

    def train_with_evaluate(self, epochs):
        # the function train the model and select the state dicts with the best evaluate accuracy
        self.acc_history = []
        self.loss_history = []
        self.model.to(self.device)
        best_evaluate_weights = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(epochs):
            average_loss = self.process(epoch)
            accuracy, _, _, _ = self.eval_probability()
            print('Epoch:{}, Loss:{}, Accuracy:{}%'.format(epoch, average_loss, accuracy * 100))
            self.loss_history.append(average_loss)
            self.acc_history.append(accuracy)
            if accuracy > best_acc:
                best_acc = accuracy
                best_evaluate_weights = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_evaluate_weights)

    def write_to_file(self, path):
        file = open(path, 'w+')
        for epoch, acc, loss in zip(range(len(self.acc_history)), self.acc_history, self.loss_history):
            record = "Epoch:" + str(epoch) + ",Accuracy:" + str(acc) + ",Loss:" + str(loss) + "\n"
            file.writelines(record)
