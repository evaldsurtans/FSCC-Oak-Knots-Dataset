# FSCC-Oak-Knots-Dataset

Dataset available in raw format:
http://share.yellowrobot.xyz/1652876792-oak-konts-dataset

Pre-processed format

```python
class DatasetWoodKnots(torch.utils.data.Dataset):
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

        if self.part == 'train':
            if args.augmentation_probability > 0:
                if True or np.random.random() < args.augmentation_probability:
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

        x = np_rgbk[0:3] / 255.0
        y = np_rgbk[3]

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y[np.newaxis, :])

        return x, y, bbox, bbox_len
```



Examples of dataset (masks and bounding boxes):
![tmp_boxes_masks](http://share.yellowrobot.xyz/upic/f00fbb92b8eb5d678c47e7c7dbe16048_1652901941.png)

