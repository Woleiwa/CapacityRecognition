import platform
import random
import torch

import torchvision.models as models

from torch import optim
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

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


def main():
    backbone = models.resnet18(pretrained=True)

    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))

    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator)

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
    images, targets = dataFetcher.obj_detection_info()
    origin_dataset = CustomObjDetectionDataset(images, targets, transformer)

    train_list = []
    test_list = []
    for i in range(len(images)):
        if random.randint(1, 10) < 8:
            train_list.append(i)
        else:
            test_list.append(i)

    train_dataset = origin_dataset.split(train_list)
    test_dataset = origin_dataset.split(test_list)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=num_workers, collate_fn=my_collate_fn)

    params_to_update = model.parameters()
    optimizer = optim.Adam(params_to_update, lr=args.lr, )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(model, None, optimizer, train_loader, test_loader, device)
    trainer.train_obj_detection(args.num_epochs
                                )

    recorder = Recorder(model, args.num_epochs, optimizer)
    recorder.write_to_file('Record/object_detection_model.txt')


if __name__ == '__main__':
    main()
