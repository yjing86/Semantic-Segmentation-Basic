from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import random
import torch
from imgaug import augmenters as iaa



def img_aug(img, mask):
#     mask = np.where(mask > 0, 0, 255).astype(np.uint8)
#     flipper = iaa.Fliplr(0.5).to_deterministic()
#     mask = flipper.augment_image(mask)
#     img = flipper.augment_image(img)
#     vflipper = iaa.Flipud(0.5).to_deterministic()
#     img = vflipper.augment_image(img)
#     mask = vflipper.augment_image(mask)
#     if random.random() < 0.5:
#         rot_time = random.choice([1, 2, 3])
#         for i in range(rot_time):
#             img = np.rot90(img)
#             mask = np.rot90(mask)
#     if random.random() < 0.5:
#         translater = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#                                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                                 shear=(-8, 8),
#                                 cval=(255)
#                                 ).to_deterministic()
#         img = translater.augment_image(img)
#         mask = translater.augment_image(mask)
#     # if random.random() < 0.5:
#     #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     mask = np.where(mask > 0, 0, 255).astype(np.uint8)
#     return img, mask
#
#
# def _count_visible_keypoints(anno):
#     return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)
#
#
# def _has_only_empty_bbox(anno):
#     return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)
#
#
# def has_valid_annotation(anno):
#     # if it's empty, there is no annotation
#     if len(anno) == 0:
#         return False
#     # if all boxes have close to zero area, there is no annotation
#     if _has_only_empty_bbox(anno):
#         return False
#     # keypoints task have a slight different critera for considering
#     # if an annotation is valid
#     if "keypoints" not in anno[0]:
#         return True
#     return False

class Dataset(Dataset):
    """My Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images(numpy): image-npy(需要提前读入)
        masks (numpy): mask-npy(需要提前读入)
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """
    def __init__(self, images, masks, size, augmentation=None, preprocessing=None, mode = 'train'):
        self.images = images
        self.masks = masks
        self.size = size
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode

    def __getitem__(self, i):
        # read data
        image = self.images[i]
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float()
        if self.mode == 'test':
            mask = '0'
        else:
            mask = self.masks[i]
            mask = torch.from_numpy(mask)
            # mask = mask.permute(2, 0, 1).float()

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # image = T.Compose([T.ToPILImage(),T.ToTensor()])(image)
            # mask = mask.astype(np.uint8)
            # mask = T.Compose([T.ToPILImage(),T.ToTensor()])(mask)

        return image, mask #.transpose(2,0,1)

    def __len__(self):
        return len(self.images)

    def augumentor(self,image):
        # 数据增强使用了imgaug
        '''未完善，不可使用
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0,4),[
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            ], random_order=True)
        '''
        image_aug = augment_img.augment_image(image)
        return image_aug
