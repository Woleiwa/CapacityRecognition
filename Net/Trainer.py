import copy
import math

import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision.ops import box_iou


def calculate_accuracy(targets, predictions, iou_threshold=0.75):
    correct_detections = 0

    for target, prediction in zip(targets, predictions):
        target_boxes = target['boxes'].cpu().numpy()
        pred_boxes = prediction['boxes'].cpu().numpy()
        iou_matrix = box_iou(torch.tensor(target_boxes), torch.tensor(pred_boxes)).numpy()
        if iou_matrix.size == 0:
            max_iou_per_target = np.full_like(iou_matrix, 0)
        else:
            max_iou_per_target = np.max(iou_matrix, axis=1)
        correct_detections += np.sum(max_iou_per_target > iou_threshold)

    accuracy = correct_detections / len(targets)
    return accuracy


def calculate_iou(targets, predictions):
    total_iou = 0

    for target, prediction in zip(targets, predictions):
        target_boxes = target['boxes'].cpu().numpy()
        pred_boxes = prediction['boxes'].cpu().numpy()
        iou_matrix = box_iou(torch.tensor(target_boxes), torch.tensor(pred_boxes)).numpy()
        if iou_matrix.size == 0:
            max_iou_per_target = np.full_like(iou_matrix, 0)
        else:
            max_iou_per_target = np.max(iou_matrix, axis=1)
        total_iou += np.sum(max_iou_per_target)

    average_iou = total_iou / len(targets)
    return average_iou


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
        # the major part of training process of merely all model
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

    def train_obj_detection(self, epochs):
        self.model.to(self.device)
        best_distance = 2
        best_state = copy.deepcopy(self.model.state_dict())
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            for image, target in self.train_loader:
                images = list(i.to(self.device) for i in image)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in target]
                images = torch.stack(images, dim=0)
                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
                train_loss += losses.item()
            train_loss = train_loss / len(self.train_loader)
            accuracy, iou, distance = self.evaluate_obj_detection()
            if best_distance < distance:
                best_distance = distance
                best_state = copy.deepcopy(self.model.state_dict())
            print("Epoch:{}, Train Loss:{}, Accuracy:{}, Iou:{}".format(epoch, train_loss, accuracy, iou))

        self.model.load_state_dict(best_state)

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

        print('Best accuracy: {}'.format(best_acc))
        self.model.load_state_dict(best_evaluate_weights)

    def write_to_file(self, path):
        file = open(path, 'w+')
        for epoch, acc, loss in zip(range(len(self.acc_history)), self.acc_history, self.loss_history):
            record = "Epoch:" + str(epoch) + ",Accuracy:" + str(acc) + ",Loss:" + str(loss) + "\n"
            file.writelines(record)

    def evaluate_obj_detection(self):
        self.model.eval()

        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                predictions = self.model(images)

                all_targets.extend(targets)
                all_predictions.extend(predictions)

        accuracy = calculate_accuracy(all_targets, all_predictions)
        iou = calculate_iou(targets, predictions)
        distance = accuracy ** 2 + iou ** 2
        distance = math.pow(distance, 1 / 2)

        return accuracy, iou, distance
