import cv2
import torch

import torch.nn as nn
import numpy as np

from Net.ModelRecorder import Recorder
from torchvision.models import ResNet50_Weights
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def resnet50_initialize():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2))
    return model


def process(cut_model, judge_model, classification_model, transform, device, x, image):
    with torch.no_grad():
        targets = cut_model(x)[0]['boxes'].cpu().numpy()
        res_list = []
        for box in targets:
            box = [int(coord) for coord in box]
            target = image.crop(box)
            target = transform(target)
            target = target.unsqueeze(0).to(device)
            judge = judge_model(target)
            predicted_class = torch.argmax(judge).item()
            if predicted_class == 1:
                res = classification_model(target)
                predicted_class = torch.argmax(res).item()
                if predicted_class == 0:
                    result = 'full'
                else:
                    result = 'unfull'

                res_list.append((box, result))

    return res_list


class Classifier:
    def __init__(self, device):
        self.whole_model = resnet50_initialize()
        self.part_model = resnet50_initialize()

        self.whole_judge_model = resnet50_initialize()
        self.part_judge_model = resnet50_initialize()

        self.whole_cut_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.part_cut_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

        recorder = Recorder(self.whole_model)
        recorder.load_model('Record/whole_full_model.txt')
        self.whole_model = recorder.model

        recorder = Recorder(self.part_model)
        recorder.load_model('Record/part_full_model.txt')
        self.part_model = recorder.model

        recorder = Recorder(self.whole_judge_model)
        recorder.load_model('Record/whole_judge.txt')
        self.whole_judge_model = recorder.model

        recorder = Recorder(self.part_judge_model)
        recorder.load_model('Record/part_judge.txt')
        self.part_judge_model = recorder.model

        recorder = Recorder(self.whole_cut_model)
        recorder.load_model('Record/whole_detection.txt')
        self.whole_cut_model = recorder.model

        recorder = Recorder(self.part_cut_model)
        recorder.load_model('Record/part_detection.txt')
        self.part_cut_model = recorder.model

        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.whole_model.to(device)
        self.part_model.to(device)
        self.whole_judge_model.to(device)
        self.part_judge_model.to(device)
        self.whole_cut_model.to(device)
        self.part_cut_model.to(device)

        self.whole_model.eval()
        self.part_model.eval()
        self.whole_judge_model.eval()
        self.part_judge_model.eval()
        self.whole_cut_model.eval()
        self.part_cut_model.eval()
        self.device = device

    def classify(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        x = transform(image).unsqueeze(0)
        x = x.to(self.device)

        whole_results = process(self.whole_cut_model, self.whole_judge_model, self.whole_model, self.transform, self.device, x, image)
        part_results = process(self.part_cut_model, self.part_judge_model, self.part_model, self.transform, self.device, x, image)

        numpy_image = np.array(image)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        for res in whole_results:
            box = res[0]
            label = res[1]
            cv2.rectangle(opencv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(opencv_image, label, (box[0], box[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        for res in part_results:
            box = res[0]
            label = res[1]
            cv2.rectangle(opencv_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(opencv_image, label, (box[0], box[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return opencv_image
