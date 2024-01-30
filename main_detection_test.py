import numpy as np
import torch
import cv2
import os

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from DataProcessor.DataFetcher import DataFetcher
from Net.ModelRecorder import Recorder
from torchvision.transforms import transforms


def draw_bounding_box(image, boxes, origin_boxes):
    for box in boxes:
        box = [int(coord) for coord in box]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    for box in origin_boxes:
        box = [int(coord) for coord in box]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)


def corner_detect(image):
    numpy_array = np.array(image)
    cv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80)
    cv2.imshow('Contours', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_directory(directory_path):
    if os.path.exists(directory_path):
        print(directory_path + " exists")
    else:
        os.makedirs(directory_path)


class Cutter:
    def __init__(self, path):
        self.cut_count = 0
        self.directory_path = path
        detect_directory(self.directory_path)

    def cut_images(self, image, boxes):
        for box in boxes:
            box = [int(coord) for coord in box]
            cut_image = image.crop(box)
            path = self.directory_path + "/true_" + str(self.cut_count) + ".jpg"
            cut_image.save(path)
            self.cut_count += 1


def test_model(model, images, targets, cutter: Cutter, cnt=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for image, target in zip(images, targets):
            # enhanced_image = clahe_enhance(image)
            x = transform(image).unsqueeze(0)
            x = x.to(device)
            prediction = model(x)
            boxes = prediction[0]['boxes'].cpu().numpy()
            numpy_image = np.array(image)
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            draw_bounding_box(opencv_image, boxes, target['boxes'])
            path = 'DetectionImg/' + str(cnt) + '.jpg'
            cnt += 1
            cv2.imwrite(path, opencv_image)
            cutter.cut_images(image, boxes)

    return cnt


def test_on_new_data(model, images, cutter: Cutter, cnt=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

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
            cutter.cut_images(image, boxes)

    return cnt


def main():
    dataFetcher = DataFetcher('Data')
    dataFetcher.read_from_file(224, 224)
    images, targets, part_labels = dataFetcher.obj_detection_info()
    print("Total num{}, Total targets{}".format(len(images), len(targets)))
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    part_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    recorder = Recorder(model)
    recorder.load_model('Record/whole_detection.txt')
    model = recorder.model

    part_recoder = Recorder(part_model)
    part_recoder.load_model('Record/part_detection.txt')
    part_model = part_recoder.model

    cutter = Cutter('CutImage')
    part_cutter = Cutter('PartImage')
    cnt = test_model(model, images, targets, cutter)
    print('cnt:{}'.format(cnt))
    test_model(part_model, images, part_labels, part_cutter, cnt)

    dataFetcher = DataFetcher('Detection')
    dataFetcher.read_images()
    images = dataFetcher.get_image()

    cnt = test_on_new_data(model, images, cutter)
    print('cnt:{}'.format(cnt))
    test_on_new_data(part_model, images, part_cutter, cnt)


if __name__ == '__main__':
    main()
