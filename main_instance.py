import pdb
from datetime import datetime
from time import time
import cv2
import numpy as np
import torchvision
import os
import torch
from PIL import Image
from loguru import logger
from matplotlib.patches import Rectangle
from torch.hub import download_url_to_file
from tqdm import tqdm  # pip3 install tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from modules.args_utils import ArgsUtils
from modules.csv_utils_2 import CsvUtils2
from modules.data_utils import DataUtils
from modules.file_utils import FileUtils
from modules.loss_functions import IoU, dice_coeficient

from torch.utils.data.distributed import DistributedSampler
import torch.distributed

import albumentations as A

plt.rcParams["figure.figsize"] = (30, 15)
import torch.utils.data
import argparse

# launch:  python -m torch.distributed.launch --nproc_per_node=2 main_instance.py

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-sequence_name', default='instance_d', type=str)
parser.add_argument('-run_name', default='run', type=str)

parser.add_argument('-class_weight', default=0.9598, type=float)

parser.add_argument('-batch_size', default=30, type=int)
parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-learning_rate', default=1e-4, type=float)

parser.add_argument('-augmentation_probability', default=0.9, type=float)

parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-model', default='fasterrcnn_mobilenet_v3_large_320_fpn', type=str)

parser.add_argument('--local_rank', default=0, type=int)

parser.add_argument('-filter_scores_above', default=0.5, type=float)
parser.add_argument('-filter_ios_below_two_predicted_boxes', default=0.1, type=float)

parser.add_argument('-p_color_jitter', default=0.5, type=float)

# -batch_size 32 -model lraspp_mobilenet_v3_large
# -batch_size 32 -model deeplabv3_mobilenet_v3_large

args, args_other = parser.parse_known_args()

args = ArgsUtils.add_other_args(args, args_other)

path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
path_artefacts = f'./artefacts/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_run)
FileUtils.createDir(path_artefacts)

logger.add(f'{path_run}/{args.run_name}.log')

CsvUtils2.create_global(path_sequence)
CsvUtils2.create_local(path_sequence, args.run_name)


args.cuda_devices_count = torch.cuda.device_count()
if not torch.cuda.is_available():
    args.device = 'cpu'
    args.batch_size = 4
    logger.info('CUDA NOT AVAILABLE')
else:
    logger.info(f'{args.cuda_devices_count} CUDA AVAILABLE')
    if args.cuda_devices_count > 1:
        torch.distributed.init_process_group(
            backend='nccl' # https://pytorch.org/docs/stable/distributed.html
        )
        args.device = torch.device('cuda', args.local_rank)
    else:
        args.device = 'cuda'


class DatasetLLUWoodKnots(torch.utils.data.Dataset):
    def __init__(self, part):

        self.part = part
        dataset_name = f'llu_wood_knots_{part}.npy'
        path_dataset = f'./dataset_processed/{dataset_name}'
        path_bounding_boxes = path_dataset.replace(f'llu_wood_knots_{part}.npy', f'llu_wood_knots_bounding_boxes_{part}.json')
        if not os.path.exists(path_dataset):
            FileUtils.createDir('./dataset_processed')
            download_url_to_file(
                f'http://share.yellowrobot.xyz/1651675667-llu/{dataset_name}',
                path_dataset,
                progress=True
            )

            download_url_to_file(
                f'http://share.yellowrobot.xyz/1651675667-llu/llu_wood_knots_bounding_boxes_{part}.json',
                path_bounding_boxes,
                progress=True
            )

        self.np_data = np.load(path_dataset, allow_pickle=False)
        # (RGBK, W, H)
        # K channel means Knot label

        self.list_bounding_boxes = FileUtils.loadJSON(path_bounding_boxes)
        self.len_max_bounding_boxes_length = max([len(bounding_boxes) for bounding_boxes in self.list_bounding_boxes])

        self.augment_transformations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                rotate=(-90, 90),
                p=0.9,
                mode=cv2.BORDER_REFLECT
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
                p=args.p_color_jitter
            ),
        ])

    def __len__(self):
        return len(self.np_data)

    def __getitem__(self, idx):
        np_rgbk = self.np_data[idx] # red, green, blue, knots
        bounding_boxes = self.list_bounding_boxes[idx]

        debug = 'none'
        if self.part == 'train':
            if args.augmentation_probability > 0:
                if True or np.random.random() < args.augmentation_probability:
                    debug = 'augment'
                    result = self.augment_transformations(image=np_rgbk[:3, :, :].transpose(1, 2, 0), mask=np_rgbk[3, :, :])
                    np_rgbk[:3, :, :] = result['image'].transpose(2, 0, 1)
                    np_rgbk[3, :, :] = result['mask']

                    # recalculate bounding boxes
                    bounding_boxes = DataUtils.get_bboxes_from_image(np_rgbk[3])
                    if len(bounding_boxes) == 0:
                        # if aufgmentation failed, use the original image
                        np_rgbk = self.np_data[idx]
                        bounding_boxes = self.list_bounding_boxes[idx]

        bbox_len = len(bounding_boxes)
        bbox = torch.zeros((self.len_max_bounding_boxes_length, 4)).float()
        if len(bounding_boxes) > self.len_max_bounding_boxes_length:
            bounding_boxes = bounding_boxes[:self.len_max_bounding_boxes_length]
        bbox[:bbox_len, :] = torch.from_numpy(np.array(bounding_boxes))

        # plt.cla()
        # plt.clf()
        # plt.imshow(np_rgbk[0:3].transpose(1, 2, 0)/255)
        # for bbox_each in bounding_boxes:
        #     if bbox_each[2] > 0:  # ignore empty boxes
        #         plt.gca().add_patch(Rectangle(
        #             (bbox_each[0], bbox_each[1]),
        #             bbox_each[2] - bbox_each[0],
        #             bbox_each[3] - bbox_each[1],
        #             linewidth=1, edgecolor='r', facecolor='none'
        #         ))
        # #plt.show()
        # plt.savefig(f'./tmp/{debug}_{idx}.png')

        x = np_rgbk[0:3] / 255.0
        y = np_rgbk[3]

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y[np.newaxis, :])

        return x, y, bbox, bbox_len


dataset_train = DatasetLLUWoodKnots(part='train')
dataset_test = DatasetLLUWoodKnots(part='test')

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=args.batch_size,
    drop_last=(len(dataset_train) % args.batch_size == 1),
    num_workers=(0 if args.device == 'cpu' else 8),
    shuffle=(args.cuda_devices_count < 2),
    sampler=(None if args.cuda_devices_count < 2 else DistributedSampler(dataset_train))
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=args.batch_size,
    drop_last=(0 < len(dataset_test) % args.batch_size < 6),
    num_workers=(0 if args.device == 'cpu' else 8),
    sampler=(None if args.cuda_devices_count < 2 else DistributedSampler(dataset_test, shuffle=False))
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        weights = [1-args.class_weight, args.class_weight]
        if args.model == 'fasterrcnn_resnet50_fpn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=True,
                num_classes=2
            )
        elif args.model == 'fasterrcnn_mobilenet_v3_large_fpn':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=False,
                pretrained_backbone=True,
                num_classes=2
            )
        elif args.model == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=False,
                pretrained_backbone=True,
                num_classes=2
            )
        elif args.model == 'fcos_resnet50_fpn':
            self.model = torchvision.models.detection.fcos_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=True,
                num_classes=2
            )
        elif args.model == 'retinanet_resnet50_fpn':
            self.model = torchvision.models.detection.retinanet_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=True,
                num_classes=2
            )
        elif args.model == 'ssd300_vgg16':
            self.model = torchvision.models.detection.ssd300_vgg16(
                pretrained=False,
                pretrained_backbone=True,
                num_classes=2
            )
        elif args.model == 'ssdlite320_mobilenet_v3_large':
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                pretrained=False,
                pretrained_backbone=True,
                num_classes=2
            )

    def forward(self, x, bbox_dict):
        out = self.model.forward(x, bbox_dict)
        return out

model = Model()
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank)

metrics = {}
metrics_epoch = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'loss_classifier',
        'loss_box_reg',
        'loss_objectness',
        'loss_rpn_box_reg',
        'dice_coef',
        'iou_box',
        'classification',
        'bbox_regression',
        'bbox_ctrness',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

    for metric in [
        'loss_best',
        'dice_coef_best',
        'iou_box_best',
        'acc_best'
    ]:
        metrics[f'{stage}_{metric}'] = float('-inf') if metric != 'loss_best' else float('inf')

for epoch in range(1, args.epochs + 1):
    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [data_loader_train, data_loader_test]:
        if data_loader == data_loader_test:
            stage = 'test'
        else:
            stage = 'train'

        for x, y, bbox, bbox_len in tqdm(data_loader, desc=f'{stage} epoch {epoch}'):

            if data_loader == data_loader_train:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)

            x = x.to(args.device)
            y = y.to(args.device).squeeze(dim=1)
            bbox = bbox.to(args.device)

            bbox_dicts = []
            for idx in range(bbox.shape[0]):
                bbox_dict = {}
                bbox_dict['boxes'] = bbox[idx, :bbox_len[idx], :]
                bbox_dict['labels'] = torch.ones(bbox_len[idx]).long().to(args.device)
                bbox_dicts.append(bbox_dict)

            model = model.train()
            out = model.forward(x, bbox_dicts)
            loss = 0
            for key in out.keys():
                loss += out[key]
                metrics_epoch[f'{stage}_{key}'].append(out[key].item())
            metrics_epoch[f'{stage}_loss'].append(loss.item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            torch.set_grad_enabled(False)

            model = model.eval()
            out = model.forward(x, bbox_dicts)
            # out is list of [{'boxes', 'labels', 'scores'}]

            # filter out the predictions that overlap with lower confidence scored boxes
            # boxes already ordered by confidence score
            # filter also out zero labelled boxes as model uses softmax with 2 classes
            bbox_prim_len_max = 0
            bbox_prim_len = []
            for idx in range(bbox.shape[0]):
                bbox_each = out[idx]['boxes']
                label_each = out[idx]['labels']
                scores_each = out[idx]['scores']

                bbox_each = [it for i, it in enumerate(bbox_each) if label_each[i] == 1]
                scores_each = [it for i, it in enumerate(scores_each) if label_each[i] == 1]
                label_each = [it for i, it in enumerate(label_each) if label_each[i] == 1]

                # filter by score above threshold args.filter_scores_above
                bbox_each = [it for i, it in enumerate(bbox_each) if scores_each[i] >= args.filter_scores_above]
                label_each = [it for i, it in enumerate(label_each) if scores_each[i] >= args.filter_scores_above]
                scores_each = [it for i, it in enumerate(scores_each) if scores_each[i] >= args.filter_scores_above]

                if len(bbox_each) > 1:
                    i = 0
                    while i < len(bbox_each) - 1:
                        bbox_main = bbox_each[i:i+1]
                        bbox_other = bbox_each[i+1:]
                        ious_other = torchvision.ops.box_iou(torch.stack(bbox_main), torch.stack(bbox_other)).cpu().data.numpy().squeeze(axis=0)
                        is_deleted = False
                        for i_other, iou_other in enumerate(reversed(ious_other)):
                            i_other = len(ious_other) - i_other - 1
                            if iou_other > args.filter_ios_below_two_predicted_boxes: # remove boxes that overlap higher confidence boxes
                                if iou_other == 0:
                                    is_deleted = True
                                del bbox_each[i + i_other + 1]
                                del label_each[i + i_other + 1]
                                del scores_each[i + i_other + 1]
                        if not is_deleted:
                            i += 1

                bbox_prim_len_max = max(bbox_prim_len_max, len(bbox_each))
                bbox_prim_len.append(len(bbox_each))
                out[idx]['boxes'] = torch.stack(bbox_each) if len(bbox_each) > 0 else torch.zeros(0, 4)
                out[idx]['scores'] = torch.stack(scores_each) if len(scores_each) > 0 else torch.zeros(0)
                out[idx]['labels'] = torch.stack(label_each) if len(label_each) > 0 else torch.zeros(0)

            # prim boxes might be more than ground truth boxes, because labelling might not be perfect
            # also they might be less than ground truth boxes
            bbox_prim = torch.zeros((bbox.shape[0], bbox_prim_len_max, 4), dtype=torch.float32).to(args.device)
            for idx in range(bbox.shape[0]):
                bbox_each = out[idx]['boxes']
                if len(bbox_each) > 0:
                    bbox_prim[idx, :len(bbox_each), :] = bbox_each.cpu()

            # torchvision.ops.box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor)
            # for each ground truth bbox find highest iou for bbox_prim then calculate average iou for whole batch
            batch_ios = []
            for idx in range(bbox.shape[0]):
                if bbox_len[idx] > 0 and bbox_prim_len[idx] > 0:
                    bbox_each = bbox[idx, :bbox_len[idx], :]
                    bbox_prim_each = bbox_prim[idx, :bbox_prim_len[idx], :]
                    ious_all = torchvision.ops.box_iou(bbox_each, bbox_prim_each).cpu().data.numpy()
                    for ious in ious_all:
                        if len(ious) > 1:
                            batch_ios.append(np.max(ious))
                        else:
                            batch_ios.append(ious[0])
                else:
                    batch_ios += [0] * bbox_len[idx]  # log each box that were not found

            batch_acc = []
            for ios in batch_ios:
                if ios > 0.0:
                    batch_acc.append(1)
                else:
                    batch_acc.append(0)
            metrics_epoch[f'{stage}_acc'].append(np.mean(batch_acc))

            batch_io = np.mean(batch_ios)
            metrics_epoch[f'{stage}_iou_box'].append(batch_io)

            loss = loss.cpu()
            x = x.cpu()
            y = y.cpu()
            bbox = bbox.cpu()
            bbox_len = bbox_len.cpu()
            bbox_prim = bbox_prim.cpu()

            np_x = x.data.numpy()
            np_y = y.data.numpy()


            if args.device == 'cpu':
                break # debugging code

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                if '_best' not in key:
                    value = np.mean(metrics_epoch[key])
                    metrics_epoch[key] = value
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')

                    if key + '_best' in metrics.keys():
                        if 'loss' in key:
                            metrics[key + '_best'] = min(metrics[key][-1], metrics[key + '_best'])
                        else:
                            metrics[key + '_best'] = max(metrics[key][-1], metrics[key + '_best'])
                        metrics_epoch[key + '_best'] = metrics[key + '_best']

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if args.local_rank == 0:
        CsvUtils2.add_hparams(
            path_sequence=path_sequence,
            run_name=args.run_name,
            args_dict=args.__dict__,
            metrics_dict=metrics_epoch,
            global_step=epoch
        )

        #plt.subplot(121) # row col idx
        plt.clf()
        plt.cla()
        plt.subplot(321)  # row col idx
        plt.plot(metrics['train_loss'], label='train_loss')
        plt.plot(metrics['test_loss'], label='test_loss')
        plt.legend()

        plt.subplot(323)  # row col idx
        plt.plot(metrics['train_acc'], label='train_acc')
        plt.plot(metrics['test_acc'], label='test_acc')
        plt.legend()

        plt.subplot(325)  # row col idx
        plt.plot(metrics['train_iou_box'], label='train_iou_box')
        plt.plot(metrics['test_iou_box'], label='test_iou_box')
        plt.legend()

        np_x = np_x.transpose(0, 2, 3, 1)
        np_x = np_x * 255
        np_x = np_x.astype(np.uint8)

        for i, j in enumerate([4, 5, 6, 16, 17, 18, 10, 11, 12, 22, 23, 24]):
            if i < len(np_x):
                plt.subplot(4, 6, j)
                plt.title('Blue - True, Red - Predicted')
                plt.imshow(np_x[i], interpolation=None)
                plt.imshow(np_y[i], cmap='Blues', alpha=0.5, interpolation=None)

                bboxes = bbox[i, :].numpy() # ground truth boxes
                for bbox_each in bboxes:
                    if bbox_each[2] > 0: # ignore empty boxes
                        plt.gca().add_patch(Rectangle(
                            (bbox_each[0], bbox_each[1]),
                            bbox_each[2] - bbox_each[0],
                            bbox_each[3] - bbox_each[1],
                            linewidth=1, edgecolor='b', facecolor='none'
                        ))

                bboxes = bbox_prim[i, :].numpy() # predicted boxes
                for bbox_each in bboxes:
                    if bbox_each[2] > 0: # ignore empty boxes
                        plt.gca().add_patch(Rectangle(
                            (bbox_each[0], bbox_each[1]),
                            bbox_each[2] - bbox_each[0],
                            bbox_each[3] - bbox_each[1],
                            linewidth=1, edgecolor='r', facecolor='none'
                        ))

        plt.tight_layout(pad=0)
        plt.savefig(f'{path_run}/epoch_{epoch}.png')
        plt.show()


