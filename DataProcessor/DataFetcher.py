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

    def read_images(self):
        file_names = []
        for file_path in glob.glob(os.path.join(self.path, '*')):
            file_name, file_extension = os.path.splitext(file_path)
            file_extension = file_extension[1:]
            if file_extension == 'jpg':
                self.image.append(Image.open(file_path))
                file_names.append(file_name)

        return file_names

    def get_image(self):
        return self.image

    def read_from_file(self, height, width):
        filenames = self.read_images()
        for file_name in filenames:
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
                    processed_img.save('Img/' + str(num) + '.jpg')
                    num = num + 1
                    labels = label.split('_')
                    self.part_label.append(labels[0])
                    self.full_label.append(labels[1])
        return self.cut_image, self.part_label, self.full_label

    def get_info(self):
        return self.image, self.info

    def obj_detection_info(self):
        whole_image = []
        whole_target = []
        part_target = []

        encoder = LabelEncoder()
        encoder.fit(self.part_label)

        for img, info in zip(self.image, self.info):
            shapes = info.get_shapes()
            flag = False

            for shape in shapes:
                label = shape.get_label().split('_')[0]
                if label == 'whole':
                    flag = True

            if flag:
                part_dict = {'boxes': [], 'labels': []}
                whole_dict = {'boxes': [], 'labels': []}
                whole_image.append(img)

                for shape in shapes:
                    label = shape.get_label().split('_')[0]
                    left = float(shape.get_points()[0][0])
                    upper = float(shape.get_points()[0][1])
                    right = float(shape.get_points()[1][0])
                    lower = float(shape.get_points()[1][1])

                    if label == 'whole':
                        label = 1
                        whole_dict['labels'].append(label)
                        whole_dict['boxes'].append([left, upper, right, lower])

                    else:
                        label = 1
                        part_dict['labels'].append(label)
                        part_dict['boxes'].append([left, upper, right, lower])

                part_target.append(part_dict)
                whole_target.append(whole_dict)

        return whole_image, whole_target, part_target


class BucketFetcher:
    def __init__(self, path, height, width):
        self.path = path
        self.height = height
        self.width = width
        self.images = []
        self.labels = []

    def read_images(self):
        for file_path in glob.glob(os.path.join(self.path, '*')):
            file_name, file_extension = os.path.splitext(file_path)
            file_extension = file_extension[1:]
            if file_extension == 'jpg':
                image = Image.open(file_path)
                image.resize((self.height,self.width))
                self.images.append(image)
                label = file_name.split('_')[0]
                self.labels.append(label)

        return self.images, self.labels
