import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import CUB, CarsDataset, NABirds, dogs, INat2017, CustomDataset, CustomDatasetAlbu
from .autoaugment import AutoAugImageNetPolicy
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as al
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def get_plant_disease_test_transform(input_size):
    return al.Compose(
        [
            al.LongestMaxSize(max_size=round(input_size * 1.1),
                              interpolation=cv2.INTER_AREA),
            al.augmentations.transforms.PadIfNeeded(min_height=round(input_size * 1.1),
                                                    min_width=round(input_size * 1.1),
                                                    border_mode=cv2.BORDER_CONSTANT,
                                                    value=0),
            al.CenterCrop(height=input_size, width=input_size),
            al.Normalize(),
            ToTensorV2()
        ])


def get_plant_disease_train_transform(input_size):
    if not isinstance(input_size, int):
        input_size = (input_size[0], input_size[1])
    return al.Compose(
        [
            al.LongestMaxSize(max_size=round(input_size * 1.1),
                              interpolation=cv2.INTER_AREA),
            al.augmentations.transforms.PadIfNeeded(min_height=round(input_size * 1.1),
                                                    min_width=round(input_size * 1.1),
                                                    border_mode=cv2.BORDER_CONSTANT,
                                                    value=0),
            al.RandomCrop(input_size, input_size),
            al.HorizontalFlip(p=0.5),
            al.ShiftScaleRotate(shift_limit=0.1,
                                scale_limit=0.1,
                                rotate_limit=30,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0,
                                p=0.5),
            al.Rotate(limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.25),
            al.RandomBrightnessContrast(p=0.25),
            al.RandomGamma(p=0.25),
            al.OneOf([
                al.HueSaturationValue(),
                al.RGBShift(),
                al.CLAHE(),
            ], p=0.25),
            al.OneOf([
                al.GaussNoise(),
                al.ISONoise(),
                al.MultiplicativeNoise(),
            ], p=0.25),
            al.Blur(blur_limit=3, p=0.25),
            al.Downscale(p=0.2, interpolation=cv2.INTER_LINEAR),
            al.ImageCompression(quality_lower=80, p=0.2),
            al.Normalize(),
            ToTensorV2()
        ])


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'CUB_200_2011':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform=test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root, 'devkit/cars_train_annos.mat'),
                               os.path.join(args.data_root, 'cars_train'),
                               os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                               # cleaned=os.path.join(data_dir,'cleaned.dat'),
                               transform=transforms.Compose([
                                   transforms.Resize((600, 600), Image.BILINEAR),
                                   transforms.RandomCrop((448, 448)),
                                   transforms.RandomHorizontalFlip(),
                                   AutoAugImageNetPolicy(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                               )
        testset = CarsDataset(os.path.join(args.data_root, 'cars_test_annos_withlabels.mat'),
                              os.path.join(args.data_root, 'cars_test'),
                              os.path.join(args.data_root, 'devkit/cars_meta.mat'),
                              # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                              transform=transforms.Compose([
                                  transforms.Resize((600, 600), Image.BILINEAR),
                                  transforms.CenterCrop((448, 448)),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                              )
    elif args.dataset == 'dog':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                        train=True,
                        cropped=False,
                        transform=train_transform,
                        download=False
                        )
        testset = dogs(root=args.data_root,
                       train=False,
                       cropped=False,
                       transform=test_transform,
                       download=False
                       )
    elif args.dataset == 'nabirds':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                              transforms.RandomCrop((304, 304)),
                                              transforms.RandomHorizontalFlip(),
                                              AutoAugImageNetPolicy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                             transforms.CenterCrop((304, 304)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)
    elif args.dataset == 'plant_disease':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                              transforms.RandomCrop((448, 448)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                             transforms.CenterCrop((448, 448)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CustomDataset(os.path.join(args.data_root, "train"), train_transform)
        testset = CustomDataset(os.path.join(args.data_root, "test"), test_transform)
    else:
        trainset = CustomDatasetAlbu(args.labeled_root, get_plant_disease_train_transform(448))
        testset = CustomDatasetAlbu(args.labeled_root, get_plant_disease_test_transform(448))

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
