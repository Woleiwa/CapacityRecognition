import torch
import torch.nn as nn

from torchvision import transforms, models
from DataProcessor.DataFetcher import DataFetcher, ImgProcessor
from Net.ModelRecorder import Recorder


def custom_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def main():
    part_model = models.resnet50(pretrained=True)
    in_features = part_model.fc.in_features
    part_model.apply(custom_weights_init)
    part_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2))

    part_full_model = models.resnet50(pretrained=True)
    part_full_model.apply(custom_weights_init)
    in_features = part_full_model.fc.in_features
    part_full_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2))

    whole_full_model = models.resnet50(pretrained=True)
    whole_full_model.apply(custom_weights_init)
    in_features = whole_full_model.fc.in_features
    whole_full_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2))

    dataFetcher = DataFetcher('Data')
    dataFetcher.read_from_file(224, 224)
    images, infos = dataFetcher.get_info()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    part_recorder = Recorder(part_model)
    part_full_recorder = Recorder(part_full_model)
    whole_full_recorder = Recorder(whole_full_model)

    part_recorder.load_model('Record/part_model_1.txt')
    part_full_recorder.load_model('Record/part_full_model_1.txt')
    whole_full_recorder.load_model('Record/whole_full_model_1.txt')

    part_model = part_recorder.model
    part_full_model = part_full_recorder.model
    whole_full_model = whole_full_recorder.model

    part_model = part_model.to(device)
    part_full_model = part_full_model.to(device)
    whole_full_model = whole_full_model.to(device)

    part_model.eval()
    part_full_model.eval()
    whole_full_model.eval()

    num = 0
    accurate_num = 0
    for image, info in zip(images, infos):
        flag = True
        for shape in info.shapes:
            if shape.get_label() == 'unjudge':
                flag = False
                break
            processor = ImgProcessor(image, 224, 224)

        if flag:
            for shape in info.shapes:
                print(str(num) + ':')
                num += 1
                left = int(shape.get_points()[0][0])
                upper = int(shape.get_points()[0][1])
                right = int(shape.get_points()[1][0])
                lower = int(shape.get_points()[1][1])
                processed_img = processor.cut_image((left, upper, right, lower))
                x = transform(processed_img).unsqueeze(0)
                x = x.to(device)
                with torch.no_grad():
                    y = part_model(x)
                print(shape.label)
                predicted_class = torch.argmax(y).item()
                if predicted_class == 0:
                    with torch.no_grad():
                        res = part_full_model(x)
                    prolix = 'part'
                else:
                    with torch.no_grad():
                        res = whole_full_model(x)
                    prolix = 'whole'

                predicted_class = torch.argmax(res).item()
                if predicted_class == 0:
                    prolix += '_full'
                else:
                    prolix += '_unfull'

                if prolix == shape.get_label():
                    accurate_num += 1
                print(prolix)
                print('\n')

    rate = accurate_num / num
    print("Rate:{:.4f}%".format(rate * 100))


if __name__ == '__main__':
    main()
