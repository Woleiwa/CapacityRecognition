import copy

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