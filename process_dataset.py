import os
from copy import copy

import cv2
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from tqdm import tqdm
import seaborn as sns
from modules.file_utils import FileUtils

is_dry_run = True

stats_label_percentage = []
stats_knots_count = []

images_to_examine_rgb_map_boxes = []

for part in ['train', 'test']:
    dataset_files = FileUtils.listSubFiles(f'./dataset_raw_{part}')
    image_files = [file_name for file_name in dataset_files if 'image' in file_name]

    list_images_processed = []

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.filterByColor = False
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)

    bounding_boxes_y = []

    for image_file in tqdm(image_files):
        np_image = np.asarray(PIL.Image.open(image_file))
        label_file = image_file.replace('image', 'label')
        if not os.path.exists(label_file):
            print(f'missing label file: {label_file}')
            exit()
        np_label = np.asarray(PIL.Image.open(label_file))

        percentage_label = np.sum(np_label) / (np_label.shape[0] * np_label.shape[1])
        stats_label_percentage.append(percentage_label)

        cv_label = (np_label * 255).astype(np.uint8)

        contours = cv2.findContours(cv_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        stats_knots_count.append(len(contours))
        print(f'contours: {len(contours)}')

        bounding_boxes_each_image = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes_each_image.append([x, y, x+w, y+h])
        bounding_boxes_y.append(bounding_boxes_each_image)

        np_label = np_label[:, :, np.newaxis]
        np_combined = np.concatenate((np_image, np_label), axis=-1)
        np_permute_cwh = np_combined.transpose((2, 0, 1))
        list_images_processed.append(np_permute_cwh)

        if len(contours) > 4:
            images_to_examine_rgb_map_boxes.append((
                str(len(contours)),
                np_image,
                np_label,
                bounding_boxes_each_image
            ))

        # if len(list_images_processed) > 300:
        #     break

    np_images_processed = np.stack(list_images_processed)
    print(f'np_images_processed.shape: {np_images_processed.shape}')
    if not is_dry_run:
        np.save(f'./dataset_processed/llu_wood_knots_{part}.npy', np_images_processed, allow_pickle=False)
        FileUtils.writeJSON(f'./dataset_processed/llu_wood_knots_bounding_boxes_{part}.json', bounding_boxes_y)

mean_label_percentage = np.mean(stats_label_percentage)
print(f'mean label percentage: {mean_label_percentage}')

stats_label_percentage = np.array(stats_label_percentage)
print('stats_label_percentage')
print(np.min(stats_label_percentage), np.max(stats_label_percentage), np.median(stats_label_percentage))

stats_label_percentage *= 100

fig, ax = plt.subplots()
bins = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
sns.distplot(stats_label_percentage, kde=False, bins=bins, ax=ax)

labels = copy(bins)
labels[0] = '< 1'
ax.set_xticks(bins, labels=labels, rotation=-45)
ax.set_xlabel('Percentage of label area')
ax.set_ylabel('Number of samples')
plt.tight_layout(pad=0)
plt.show()

fig, ax = plt.subplots()
bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sns.distplot(stats_knots_count, kde=False, bins=bins, ax=ax)
ax.set_xticks(bins)
print('stats_knots_count')
print(np.min(stats_knots_count), np.max(stats_knots_count), np.median(stats_knots_count))
ax.set_xlabel('Number of knots')
ax.set_ylabel('Number of samples')
plt.tight_layout(pad=0)
plt.show()

#images_to_examine_rgb_map_boxes = sorted(images_to_examine_rgb_map_boxes, key=lambda x: len(x[3]), reverse=True)

plt.clf()
plt.cla()

for i, (
    name_of_sample,
    np_x,
    np_y,
    bboxes
) in enumerate(images_to_examine_rgb_map_boxes):
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    plt.title(name_of_sample)
    plt.imshow(np_x, interpolation=None)
    plt.imshow(np_y, cmap='Reds', alpha=0.5, interpolation=None)

    for bbox_each in bboxes:
        if bbox_each[2] > 0:  # ignore empty boxes
            plt.gca().add_patch(Rectangle(
                (bbox_each[0], bbox_each[1]),
                bbox_each[2] - bbox_each[0],
                bbox_each[3] - bbox_each[1],
                linewidth=1, edgecolor='r', facecolor='none'
            ))

    if i + 1 >= 9:
        break

plt.tight_layout(pad=1)
plt.savefig(f'./tmp_boxes_masks.png', dpi=300)
plt.show()


plt.clf()
plt.cla()

for i, (
    name_of_sample,
    np_x,
    np_y,
    bboxes
) in enumerate(images_to_examine_rgb_map_boxes):
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    plt.title(name_of_sample)
    plt.imshow(np_x, interpolation=None)
    plt.imshow(np_y, cmap='Reds', alpha=0.5, interpolation=None)

    if i + 1 >= 9:
        break

plt.tight_layout(pad=1)
plt.savefig(f'./tmp_masks.png', dpi=300)
plt.show()


plt.clf()
plt.cla()

for i, (
    name_of_sample,
    np_x,
    np_y,
    bboxes
) in enumerate(images_to_examine_rgb_map_boxes):
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    plt.title(name_of_sample)
    plt.imshow(np_x, interpolation=None)

    plt.imshow(np.ones_like(np_x) * 255, alpha=0.5, interpolation=None)

    for bbox_each in bboxes:
        if bbox_each[2] > 0:  # ignore empty boxes
            plt.gca().add_patch(Rectangle(
                (bbox_each[0], bbox_each[1]),
                bbox_each[2] - bbox_each[0],
                bbox_each[3] - bbox_each[1],
                linewidth=1, edgecolor='r', facecolor='none'
            ))

    if i + 1 >= 9:
        break

plt.tight_layout(pad=1)
plt.savefig(f'./tmp_boxes.png', dpi=300)
plt.show()

for i, (
    name_of_sample,
    np_x,
    np_y,
    bboxes
) in enumerate(images_to_examine_rgb_map_boxes):
    plt.subplot(3, 3, i + 1)
    plt.axis('off')
    plt.title(name_of_sample)
    plt.imshow(np_x, interpolation=None)
    plt.imshow(np.ones_like(np_x) * 255, alpha=0.5, interpolation=None)

    if i + 1 >= 9:
        break

plt.tight_layout(pad=1)
plt.savefig(f'./tmp_clear.png', dpi=300)
plt.show()