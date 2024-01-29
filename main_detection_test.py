import numpy as np
import torch
import cv2

import torchvision.models as models

from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from DataProcessor.DataFetcher import DataFetcher
from DataProcessor.Dataset import clahe_enhance
from Net.ModelRecorder import Recorder
from torchvision.transforms import transforms


def draw_bounding_box(image, boxes, origin_box):
    for box in boxes:
        box = [int(coord) for coord in box]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    box =[int(coord) for  coord in origin_box]
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)


def corner_detect(image):
    numpy_array = np.array(image)
    cv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80)
    cv2.imshow('Contours', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataFetcher = DataFetcher('Data')
    dataFetcher.read_images()
    images, targets, part_labels = dataFetcher.obj_detection_info()

    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator)

    recorder = Recorder(model)
    recorder.load_model('Record/whole_detection.txt')
    model = recorder.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    cnt = 0

    with torch.no_grad():
        for image, target in zip(images, targets):
            # enhanced_image = clahe_enhance(image)
            x = transform(image).unsqueeze(0)
            x = x.to(device)
            prediction = model(x)
            boxes = prediction[0]['boxes'].cpu().numpy()
            numpy_image = np.array(image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            draw_bounding_box(opencv_image, boxes, target['boxes'][0])
            path = 'DetectionImg/' + str(cnt) + '.jpg'
            cnt += 1
            cv2.imwrite(path, opencv_image)

    dataFetcher = DataFetcher('Detection')
    dataFetcher.read_images()
    images = dataFetcher.get_image()
    with torch.no_grad():
        for image in images:
            x = transform(image).unsqueeze(0)
            x = x.to(device)
            prediction = model(x)
            boxes = prediction[0]['boxes'].cpu().numpy()
            numpy_image = np.array(image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            for box in boxes:
                box = [int(coord) for coord in box]
                cv2.rectangle(opencv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            path = 'DetectionTest/' + str(cnt) + '.jpg'
            cnt += 1
            cv2.imwrite(path, opencv_image)


if __name__ == '__main__':
    main()
