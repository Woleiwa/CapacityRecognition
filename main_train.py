import os
import random
import platform
import torch

import torch.nn as nn

from Net.ModelRecorder import Recorder
from DataProcessor.DataFetcher import DataFetcher
from DataProcessor.Dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch import optim
from Net.Trainer import Trainer
from Utils.Options import args_parser


def custom_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train(slogan, train_dataset, test_dataset, args, num_workers, device, path):
    print(slogan)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    model = models.resnet50(pretrained=True)
    model.apply(custom_weights_init)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2))

    criterion = nn.CrossEntropyLoss()
    params_to_update = model.parameters()
    optimizer = optim.Adam(params_to_update, lr=args.lr, )

    trainer = Trainer(model, criterion, optimizer, train_loader, test_loader, device)
    trainer.train_with_evaluate(args.num_epochs)
    trainer.write_to_file(args.record_path)

    val_accuracy, val_precision, val_recall, val_f1 = trainer.eval_probability()
    files = [f for f in os.listdir('Result') if os.path.isfile(os.path.join('Result', f))]
    num = len(files)
    file = open('Result/' + str(num) + '.txt', 'w+')
    result = 'Validation Accuracy:' + str(val_accuracy * 100) + '\nValidation Precision:' + str(
        val_precision * 100) + '\nValidation Recall:' + str(val_recall * 100) + '\nValidation F1-Score' + str(
        val_f1 * 100)
    file.writelines(result)
    print("Validation Accuracy: {:.4f}%".format(val_accuracy * 100))
    print("Validation Precision: {:.4f}%".format(val_precision * 100))
    print("Validation Recall: {:.4f}%".format(val_recall * 100))
    print("Validation F1-Score: {:.4f}%".format(val_f1 * 100))

    recorder = Recorder(model, args.num_epochs, optimizer, criterion)
    recorder.write_to_file(path)
    return model


def main():
    system = platform.system()
    if system == 'Windows':
        num_workers = 0
    else:
        num_workers = 2
    args = args_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataFetcher = DataFetcher('Data')
    images, part_labels, full_labels = dataFetcher.read_from_file(224, 224)
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    origin_dataset = CustomDataset(transform, images, part_labels)
    part_train_list = []
    part_test_list = []
    for i in range(len(images)):
        if random.randint(1, 10) < 8:
            part_train_list.append(i)
        else:
            part_test_list.append(i)

    print("length of train list{}".format(len(part_train_list)))
    print("length of test list{}".format(len(part_test_list)))

    part_train_dataset = origin_dataset.split(part_train_list)
    part_test_dataset = origin_dataset.split(part_test_list)
    part_model = train('Train of part division model', part_train_dataset, part_test_dataset, args, num_workers, device, "Record/part_model.txt")

    img, label = origin_dataset.get_origin(0)
    img = transform(img).unsqueeze(0).to(device)
    output = part_model(img)
    print('label:' + label + 'output:' + str(output))
    predicted_class = torch.argmax(output).item()
    print(predicted_class)

    full_dataset = CustomDataset(transform, images, full_labels)
    part_full_train_list = []
    part_full_test_list = []
    whole_full_test_list = []
    whole_full_train_list = []
    for i in range(len(images)):
        if part_labels[i] == 'part':
            if random.randint(1, 10) < 8:
                part_full_train_list.append(i)
            else:
                part_full_test_list.append(i)
        else:
            if random.randint(1, 10) < 8:
                whole_full_train_list.append(i)
            else:
                whole_full_test_list.append(i)
    part_full_train_dataset = full_dataset.split(part_full_train_list)
    part_full_test_dataset = full_dataset.split(part_full_test_list)
    whole_full_train_dataset = full_dataset.split(whole_full_train_list)
    whole_full_test_dataset = full_dataset.split(whole_full_test_list)
    part_full_model = train('Train of part model', part_full_train_dataset, part_full_test_dataset, args, num_workers,
                            device, 'Record/part_full_model.txt')
    whole_full_model = train('Train of whole model', whole_full_train_dataset, whole_full_test_dataset, args,
                             num_workers, device, 'Record/whole_full_model.txt')

    img, label = part_full_train_dataset.get_origin(0)
    img = transform(img).unsqueeze(0).to(device)
    output = part_full_model(img)
    print('label:' + label + 'output:' + str(output))
    predicted_class = torch.argmax(output).item()
    print(predicted_class)

    img, label = part_full_train_dataset.get_origin(0)
    img = transform(img).unsqueeze(0).to(device)
    output = whole_full_model(img)
    print('label:' + label + 'output:' + str(output))
    predicted_class = torch.argmax(output).item()
    print(predicted_class)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
