import copy

import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, transform=None, images=None, labels=None):
        self.transform = transform
        self.images = images
        self.labels = labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label = self.label_encoder.transform([label])[0]
        if self.transform:
            image = self.transform(image)

        return image, label

    def split(self, idx):
        other = CustomDataset(transform=self.transform, images=self.images, labels=self.labels)
        other.labels = []
        other.images = []
        other.label_encoder = copy.deepcopy(self.label_encoder)
        for i in idx:
            other.labels.append(self.labels[i])
            other.images.append(self.images[i])

        return other

    def get_origin(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


class ObjDetectionTransformer:
    def __init__(self, transform):
        self.transform = transform

    def process(self, image, annotation):
        image = self.transform(image)
        target = {}
        target['boxes'] = torch.tensor(annotation['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(annotation['labels'], dtype=torch.int64)
        return image, target

    def __call__(self, image, target):
        return self.process(image, target)


class CustomObjDetectionDataset(Dataset):
    def __init__(self, images, annotations, transform: ObjDetectionTransformer = None):
        self.images = images
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.annotations[idx]

        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def split(self, idx):
        other = CustomObjDetectionDataset(transform=self.transform, images=self.images, annotations=self.annotations)
        other.images = []
        other.annotations = []
        for i in idx:
            other.annotations.append(self.annotations[i])
            other.images.append(self.images[i])

        return other
