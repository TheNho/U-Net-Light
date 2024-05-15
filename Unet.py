# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import random

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize, rescale, rotate
from torch.utils.data import Dataset
from torchvision.transforms import Compose

def pad_square_sample(x):
    def pad_image(image):
        H, W = image.shape[:2]
        padded_size = max(W, H)
        padded_mask = np.zeros(shape=(padded_size, padded_size, image.shape[-1]))
        if len(image.shape) < 3:
            padded_mask = np.zeros(shape=(padded_size, padded_size))
        p = ((padded_size-H)//2, (padded_size-W)//2)
        padded_mask[p[0]: p[0] + image.shape[0] , p[1]: p[1] + image.shape[1]] = image
        return padded_mask
    
    image, mask = x
    return pad_image(image), pad_image(mask)

def resize_sample(x, size=256):
    image, mask = x
    mask = resize(
        mask,
        output_shape= (size, size),
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    image = resize(
        image,
        output_shape= (size, size),
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return image, mask

class SegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir = './',
        images_folder = 'images',
        masks_folder = 'masks',
        transform=None,
        image_size=256,
        random_sampling=True,
    ):
        self.image_size= image_size
        self.path_to_images = os.path.join(root_dir, images_folder)
        self.path_to_masks = os.path.join(root_dir, masks_folder)  
        # Read images and mask folder (image: .jpg, mask:.png)
        basename_images = [f[:-4] for f in os.listdir(self.path_to_images) if f.endswith('.jpg')]
        print(f"Found {len(basename_images)} images")
        basename_masks = [f[:-4] for f in os.listdir(self.path_to_masks) if f.endswith('.jpg')]
        print(f"Found {len(basename_masks)} masks")
        # Intersection image and mask name
        self.basename_list=list(set(basename_images).intersection(set(basename_masks)))
        print('There are {} images with ground truth.'.format(len(self.basename_list)))
        # Shuffle
        if random_sampling:
            self.basename_list = sorted(self.basename_list, key=lambda x: random.random())
        self.transform = transform

    def __len__(self):
        return len(self.basename_list)

    def __getitem__(self, idx):
        image_name = self.basename_list[idx] + '.jpg'
        mask_name = self.basename_list[idx] + '.jpg'
        image = imread(os.path.join(self.path_to_images, image_name))
        mask = imread(os.path.join(self.path_to_masks, mask_name), as_gray=True)

        image, mask = pad_square_sample((image, mask))
        image, mask = resize_sample((image, mask), size=256) 

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        image = image/255.0
        mask = mask/255.0 if np.max(mask) > 1 else mask
        mask = np.expand_dims(mask, axis=0) if len(mask.shape) < 3 else mask

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor

def transforms(scale=None, angle=None, flip_prob=None, shift=None):
    transform_list = []
    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))
    if shift is not None:
        transform_list.append(Shift(shift))
    return Compose(transform_list)

class Shift(object):
    def __init__(self, distance):
        self.distance = distance
    def shift_image(self, X, dx, dy):
        X = np.roll(X, dy, axis=0)
        X = np.roll(X, dx, axis=1)
        if dy>0:
            X[:dy, :] = 0
        elif dy<0:
            X[dy:, :] = 0
        if dx>0:
            X[:, :dx] = 0
        elif dx<0:
            X[:, dx:] = 0
        return X
    
    def __call__(self, sample):
        image, mask = sample
        img_size = image.shape[0]
        dx, dy = np.random.randint(-self.distance, self.distance, size=2)
        if np.abs(dx) > img_size//3:
            dx = img_size//3 if dx > 0 else -img_size//3
        if np.abs(dy) > img_size//3:
            dy = img_size//3 if dy > 0 else -img_size//3
        image = self.shift_image(image, dx, dy)
        mask = self.shift_image(mask, dx, dy)
        return image, mask

class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            scale,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            scale,
            order=0,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding[:-1], mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask

class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask

class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

def log_images(x, y_true, y_pred, channel=1):
    images = []
    x_np = x[:, channel].cpu().numpy()
    y_true_np = y_true[:, 0].cpu().numpy()
    y_pred_np = y_pred[:, 0].cpu().numpy()
    for i in range(x_np.shape[0]):
        image = gray2rgb(np.squeeze(x_np[i]))
        image = outline(image, y_pred_np[i], color=[255, 0, 0])
        image = outline(image, y_true_np[i], color=[0, 255, 0])
        images.append(image)
    return images

def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image))
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret

def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image

def data_loaders(root_dir, batch_size, workers, image_size, aug_scale, aug_angle, shift):
    dataset_train, dataset_valid = datasets( root_dir, image_size, aug_scale, aug_angle, shift)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last= False,
        num_workers=workers,
        
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
    )

    return loader_train, loader_valid

def datasets(root_dir, image_size, aug_scale, aug_angle, shift):
    train = SegmentationDataset(
        root_dir = os.path.join(root_dir, 'train'),
        image_size=image_size,
        transform=transforms(scale=aug_scale, angle=aug_angle, flip_prob=0.5, shift=shift),
    )
    valid = SegmentationDataset(
        root_dir= os.path.join(root_dir, 'valid'),
        image_size=image_size,
        random_sampling=False,
    )
    return train, valid

def dsc(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def predict(model: UNet, image):
    image = image/255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.from_numpy(image.astype(np.float32)).to(next(model.parameters()).device)
    output  = model(image_tensor)
    out_image = output.detach().cpu().numpy()[0]
    out_image += 0.5
    out_image //= 1
    out_image *= 255
    out_image = out_image.astype(np.uint8)
    return out_image

def train_validate():
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    # Change your setting here
    data_dir = './dataset'
    batch_size = 8
    epochs = 50
    lr = 1e-3
    workers = 4
    save_folder = "./saved_model"
    image_size = 256
    aug_scale = 0.05
    aug_angle = 15
    shift = 50 #pixel

    os.makedirs(save_folder, exist_ok=True)
    loader_train, loader_valid = data_loaders(data_dir, batch_size, workers, image_size, aug_scale, aug_angle, shift)
    loaders = {"train": loader_train, "valid": loader_valid}
    print('Len of data loader train:', len(loader_train))
    print('Len of data loader valid:', len(loader_valid))
    
    unet = UNet(in_channels= 3, out_channels= 1, init_features=32)
    unet.to(device)
    
    dsc_loss = DiceLoss()
    
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    loss_train = []
    loss_valid = []
    
    step = 0
    
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()
    
            for data in tqdm(loaders[phase]):
                if phase == "train":
                    step += 1
    
                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)
    
                    loss = dsc_loss(y_pred, y_true)
    
                    if phase == "valid":
                        loss_valid.append(loss.item())
                        
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
            
            if phase == "train":
                print("Train - epoch {} | {}: {}".format(epoch + 1,"loss", np.mean(loss_train)))
                loss_train = []
            else:
                print("Valid - epoch {} | {}: {}".format(epoch + 1,"loss", np.mean(loss_valid)))
                loss_valid = []
        scheduler.step()
        # Draw result to visualize test data
        unet.eval()
        path_to_images_test = os.path.join(data_dir, 'test', 'images')
        list_image_test = [f for f in os.listdir(path_to_images_test) if f.endswith('.jpg')]
        print('Save test result ....')
        for image_test in tqdm(list_image_test):
            image_path = os.path.join(path_to_images_test, image_test)
            image = imread(image_path)
            out_image = predict(unet, image)
            path_save_image = image_path[:-4] + '_mask.png'
            imsave(path_save_image, out_image)

        # Save check point
        torch.save(unet, os.path.join(save_folder, 'unet.pt'))


if __name__ == '__main__':
    train_validate()
    