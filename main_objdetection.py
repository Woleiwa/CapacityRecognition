import platform
import random
import torch


from torch import optim
from torchvision.models.detection import  fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from Net.ModelRecorder import Recorder
from Utils.Options import args_parser
from torchvision.transforms import transforms
from DataProcessor.Dataset import CustomObjDetectionDataset, ObjDetectionTransformer
from DataProcessor.DataFetcher import DataFetcher
from torch.utils.data import DataLoader
from Net.Trainer import Trainer


def my_collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets


def split_list(num):
    train_list = []
    test_list = []
    for i in range(num):
        if random.randint(1, 10) < 8:
            train_list.append(i)
        else:
            test_list.append(i)

    return train_list, test_list


def train(dataset, train_list, test_list, record_path, num_workers, args):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    recorder = Recorder(model)
    recorder.load_model(record_path)
    model = recorder.model

    train_dataset = dataset.split(train_list)
    test_dataset = dataset.split(test_list)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=num_workers, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True,
                             num_workers=num_workers, collate_fn=my_collate_fn)

    params_to_update = model.parameters()
    optimizer = optim.Adam(params_to_update, lr=args.lr, )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(model, None, optimizer, train_loader, test_loader, device)
    trainer.train_obj_detection(args.num_epochs)

    recorder = Recorder(model, args.num_epochs, optimizer)
    recorder.write_to_file(record_path)


def main():
    system = platform.system()
    args = args_parser()
    if system == 'Windows':
        num_workers = 0
    else:
        num_workers = 2

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transformer = ObjDetectionTransformer(transform)
    dataFetcher = DataFetcher('Data')
    dataFetcher.read_from_file(224, 224)
    images, targets, part_target = dataFetcher.obj_detection_info()
    print("Total num{}, Total targets{}".format(len(images), len(targets)))

    origin_dataset = CustomObjDetectionDataset(images, targets, transformer)
    origin_part_dataset = CustomObjDetectionDataset(images, part_target, transformer)
    train_list, test_list = split_list(len(images))
    part_train_list, part_test_list = split_list(len(images))
    train(origin_dataset, train_list, test_list, 'Record/whole_detection.txt', num_workers, args)
    train(origin_part_dataset, part_train_list, part_test_list, 'Record/part_detection.txt', num_workers, args)


if __name__ == '__main__':
    main()
