from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import random
import math

import torch
from torchvision.transforms import *

from skimage.util.noise import random_noise
import operator
from scipy.ndimage.morphology import distance_transform_edt as dt
import skimage.draw as draw
import numpy as np
import cv2

def add_sharpness(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def blur(image):
    return cv2.blur(image,(5,5))

def sharp_blur(image):
    return blur(image) if np.random.random()>0.5 else add_sharpness(image)

def saturation(image):
    hsvImg = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsvImg[..., 1] = hsvImg[..., 1] * np.random.random()*2
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB)

def contrast(img):
    factor = max(0.2, np.random.random()*2.)
    return np.clip(128 + factor * img - factor * 128, 0, 255).astype(np.uint8)


def get_mask(img, minmax=(60, 80)):
    mask = np.zeros_like(img[..., 0])
    x, y = mask.shape
    min_dim = min(x, y)
    if np.random.random() > 0.5: # Circle-shaped masks
        random_r = np.random.randint(int(min_dim / 5), int(min_dim / 2))
        random_r = int(random_r / 2)
        random_x = np.random.randint(random_r, x - random_r)
        random_y = np.random.randint(random_r, y - random_r)
        rr, cc = draw.circle(random_x, random_y, random_r)
    else: # Ellipse-shaped masks
        random_r = np.random.randint(int(min_dim / 5), int(min_dim / 1.5))
        random_r = int(random_r / 2)
        random_x = np.random.randint(random_r, x - random_r)
        random_y = np.random.randint(random_r, y - random_r)
        rr, cc = draw.ellipse(random_x, random_y, random_r, random_r*np.random.uniform(low=0.3, high=0.8, size=1)[0],
                                shape=(x,y),rotation=np.random.random()*np.pi*2-np.pi)
    mask[rr, cc] = 1
    mask = dt(mask)
    rv = np.random.randint(minmax[0], minmax[1])
    mask = mask / np.max(mask) * rv
    return mask, rv


def illumination_augmenter(img, global_mask=(20, 40), local_mask=(60, 80)):
    img = np.squeeze(img)

    if np.random.random()<0.33: img=saturation(img)
    elif np.random.random()<0.66: img=sharp_blur(img)
    else:
        img=contrast(img)
        local_mask=(40,60)
        global_mask=(15,30)

    # Only local changes
    if any(x > 0 for x in local_mask):
        mask, ch = get_mask(img, local_mask)
        mask = np.stack((mask,) * 3, axis=-1)
        sign = '-'
        if np.random.random() > 0.5:
            sign = '+'
            img = img + mask
        else:
            img = img - mask

        # Local and global changes
        if any(x > 0 for x in global_mask):
            if np.random.random() > 0.5:
                sign += '+'
            else:
                sign += '-'
            if sign == '--' or sign == '++':
                global_max = global_mask[1]
                global_min = global_mask[0]
            else:
                global_max = global_mask[1] + ch
                global_min = global_mask[0] + ch

            if sign[1] == '+':
                img = img + np.ones_like(img) * np.random.randint(global_min, global_max)
            elif sign[1] == '-':
                img = img + np.ones_like(img) * np.random.randint(global_min, global_max) * -1

    # Only global changes
    elif any(x > 0 for x in global_mask):
        global_min, global_max = global_mask
        sign = [-1, 1]
        img = img + np.ones_like(img) * np.random.randint(global_min, global_max) * sign[np.random.randint(0, 2)]
    img[img > 255] = 255
    img[img < 0] = 0
    return img[np.newaxis, ...]


class Augmenter:
    def __init__(self, local_mask=(30, 40), global_mask=(10, 20),
                 flip_and_noise=False, augmenting_prob=0.67):

        self.augmenting_prob = augmenting_prob
        self.local_mask = local_mask
        self.global_mask = global_mask
        self.flip_and_noise = flip_and_noise
        self.augment_illumination = any(x > 0 for x in list(local_mask) + list(global_mask))

    def __call__(self, img):
        if torch.is_tensor(img):
            # print(img)
            img = img.permute(1, 2, 0).numpy() * 255

        if self.augment_illumination and np.random.random() < self.augmenting_prob:
            img = illumination_augmenter(img, self.global_mask, self.local_mask)
        # if self.flip_and_noise and np.random.random() < self.augmenting_prob:
        #     img, gt = regular_augmenter(img, gt)
        # print(img.shape)
        return torch.from_numpy(img / 255).squeeze(0).permute(2, 0, 1).float()


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    Origin: https://github.com/zhunzhong07/Random-Erasing
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ColorAugmentation(object):
    """
    Randomly alter the intensities of RGB channels
    Reference:
    Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural Networks. NIPS 2012.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def build_transforms(height,
                     width,
                     illumination_aug=False,
                     random_erase=False,  # use random erasing for data augmentation
                     color_jitter=False,  # randomly change the brightness, contrast and saturation
                     color_aug=False,  # randomly alter the intensities of RGB channels
                     **kwargs):
    # use imagenet mean and std as default
    # TODO: compute dataset-specific mean and std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    # build train transformations
    transform_train = []
    transform_train += [Random2DTranslation(height, width)]
    # transform_train += [RandomHorizontalFlip()]
    if color_jitter:
        transform_train += [ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)]
    transform_train += [ToTensor()]
    # if illumination_aug:
    #     transform_train += [Augmenter()]
    # if color_aug:
    #     transform_train += [ColorAugmentation()]
    transform_train += [normalize]
    if random_erase:
        transform_train += [RandomErasing()]
    transform_train = Compose(transform_train)

    # build test transformations
    transform_test = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_train, transform_test


def build_strong_transforms(height,
                     width,
                     illumination_aug=False,
                     random_erase=False,  # use random erasing for data augmentation
                     color_jitter=False,  # randomly change the brightness, contrast and saturation
                     color_aug=False,  # randomly alter the intensities of RGB channels
                     **kwargs):
    # use imagenet mean and std as default
    # TODO: compute dataset-specific mean and std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    # build train transformations
    transform_train = []
    transform_train += [Random2DTranslation(height, width), RandAugmentMC(2, 10)]
    # transform_train += [RandomHorizontalFlip()]
    # if color_jitter:
    #     transform_train += [ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)]
    transform_train += [ToTensor()]
    # if illumination_aug:
    #     transform_train += [Augmenter()]
    # if color_aug:
    #     transform_train += [ColorAugmentation()]
    transform_train += [normalize]
    # if random_erase:
    #     transform_train += [RandomErasing()]
    transform = Compose(transform_train)

    return transform


def build_additional_transforms(
                     random_erase=False,  # use random erasing for data augmentation
                     color_jitter=False,  # randomly change the brightness, contrast and saturation
                     color_aug=False):
    # use imagenet mean and std as default
    # TODO: compute dataset-specific mean and std

    # build train transformations
    transform_train = [ToPILImage()]
    transform_train += [ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)]
    transform_train += [ColorAugmentation()]
    transform_train = Compose(transform_train)


    return transform_train, transform_test