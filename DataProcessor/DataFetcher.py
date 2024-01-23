import os
import glob
import json

from sklearn.preprocessing import LabelEncoder
from PIL import Image


class Shape:
    def __init__(self, shape):
        self.label = str(shape['label'])
        self.points = shape['points']
        self.shape_type = shape['shape_type']
        self.group_id = shape['group_id']

    def get_label(self):
        return self.label

    def get_points(self):
        return self.points

    def get_shape_type(self):
        return self.shape_type

    def get_group_id(self):
        return self.group_id


class ImageInfo:
    def __init__(self, data):
        self.version = data['version']
        self.shapes = []
        for shape in data['shapes']:
            self.shapes.append(Shape(shape))

    def get_shapes(self):
        return self.shapes


class ImgProcessor:
    def __init__(self, img: Image, height, width):
        self.img = img
        self.height = height
        self.width = width

    def cut_image(self, box):
        new_img = self.img.crop(box)
        new_img = new_img.resize((self.height, self.width))
        return new_img


class DataFetcher:
    def __init__(self, path):
        self.path = path
        self.image = []
        self.info = []
        self.cut_image = []
        self.part_label = []
        self.full_label = []

    def read_from_file(self, height, width):
        for file_path in glob.glob(os.path.join(self.path, '*')):
            file_name, file_extension = os.path.splitext(file_path)

            file_extension = file_extension[1:]
            if file_extension == 'jpg':
                self.image.append(Image.open(file_path))
                json_file_path = file_name + '.json'
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                    img_info = ImageInfo(data)
                    self.info.append(img_info)

        num = 0
        for img, info in zip(self.image, self.info):
            processor = ImgProcessor(img, height, width)
            shapes = info.get_shapes()

            for shape in shapes:
                label = shape.get_label()
                if label != 'unjudge':
                    left = int(shape.get_points()[0][0])
                    upper = int(shape.get_points()[0][1])
                    right = int(shape.get_points()[1][0])
                    lower = int(shape.get_points()[1][1])
                    processed_img = processor.cut_image((left, upper, right, lower))
                    self.cut_image.append(processed_img)
                    processed_img.save('img/' + str(num) + '.jpg')
                    num = num + 1
                    labels = label.split('_')
                    self.part_label.append(labels[0])
                    self.full_label.append(labels[1])
        return self.cut_image, self.part_label, self.full_label

    def get_info(self):
        return self.image, self.info

    def obj_detection_info(self):
        res_image = []
        res_target = []
        label_list = []

        encoder = LabelEncoder()
        encoder.fit(self.part_label)

        for img, info in zip(self.image, self.info):
            shapes = info.get_shapes()
            judge = True
            for shape in shapes:
                label = shape.get_label()
                if label == 'unjudge':
                    judge = False

            if judge:
                res_image.append(img)
                image_dict = {'boxes': [], 'labels': []}
                for shape in info.get_shapes():
                    label = shape.get_label().split('_')[0]
                    label = encoder.transform([label])[0]
                    left = float(shape.get_points()[0][0])
                    upper = float(shape.get_points()[0][1])
                    right = float(shape.get_points()[1][0])
                    lower = float(shape.get_points()[1][1])
                    label_list.append(label)
                    image_dict['labels'].append(label)
                    image_dict['boxes'].append([left, upper, right, lower])

                res_target.append(image_dict)


        print(len(res_image))
        print(len(res_target))
        return res_image, res_target
