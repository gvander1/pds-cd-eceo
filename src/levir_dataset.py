import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# Root of LEVIR-CD on the workstation. Override with:  export LEVIR_DATA_FOLDER=/some/other/path
DATA_FOLDER = os.getenv("LEVIR_DATA_FOLDER", "/data/gaetane/LEVIR-CD")

# Expected layout (official release, after unzipping train / val / test):
#   DATA_FOLDER/
#     train/  A/*.png  B/*.png  label/*.png     (445 pairs)
#     val/    A/*.png  B/*.png  label/*.png     (64 pairs)
#     test/   A/*.png  B/*.png  label/*.png     (128 pairs)
# A = image at t1, B = image at t2, label = binary building change (0 = no change, 255 = change)
# All images are 1024 x 1024 RGB at 0.5 m/px.


class LEVIRCD_Dataset(Dataset):
    """Binary change detection dataset for LEVIR-CD.

    Returns (image1, image2, changemap) or (image1, image2, changemap, name) if get_name=True
      image1, image2 : float tensor (3, cropsize, cropsize), values in [0, 1]
      changemap      : long tensor (cropsize, cropsize), values in {0, 1}

    Cropping:
      - train : one random crop per image per epoch (+ optional flips / 90° rotations)
      - val / test : deterministic grid of non-overlapping tiles covering the whole image
        (1024 / 256 -> 16 tiles per image), so evaluation is reproducible and covers every pixel.
    """

    def __init__(self, cropsize=256, type='train', data_folder=DATA_FOLDER,
                 augment=None, small=False, size=10, get_name=False):
        if type not in ('train', 'val', 'test'):
            raise ValueError(f"type must be 'train', 'val' or 'test', got {type!r}")

        root = os.path.join(data_folder, type)
        self.folder_i1 = os.path.join(root, 'A')
        self.folder_i2 = os.path.join(root, 'B')
        self.folder_cd = os.path.join(root, 'label')

        def list_png(folder):
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Folder not found: {folder}")
            return sorted(f for f in os.listdir(folder) if f.lower().endswith('.png'))

        names1 = list_png(self.folder_i1)
        names2 = list_png(self.folder_i2)
        names_cd = list_png(self.folder_cd)

        # A, B and label must contain exactly the same file names, otherwise pairs get mixed up
        if not (names1 == names2 == names_cd):
            only = set(names1) ^ set(names2) | set(names1) ^ set(names_cd)
            raise ValueError(f"A / B / label file names do not match in {root}. "
                             f"Examples of mismatches: {sorted(only)[:5]}")
        if len(names1) == 0:
            raise ValueError(f"No .png file found in {self.folder_i1}")

        self.names = names1[:size] if small else names1

        self.type = type
        self.cropsize = cropsize
        self.get_name = get_name
        self.augment = (type == 'train') if augment is None else augment

        # Image size read from the first file (1024 for LEVIR-CD); all images are assumed to share it
        with Image.open(os.path.join(self.folder_i1, self.names[0])) as im:
            self.img_w, self.img_h = im.size
        if cropsize > min(self.img_w, self.img_h):
            raise ValueError(f"cropsize={cropsize} larger than image size {self.img_w}x{self.img_h}")

        # Grid of tiles for val / test (pixels beyond the last full tile are dropped if not divisible)
        self.tiles_y = self.img_h // cropsize
        self.tiles_x = self.img_w // cropsize
        self.tiles_per_image = 1 if type == 'train' else self.tiles_y * self.tiles_x

    def __len__(self):
        return len(self.names) * self.tiles_per_image

    def _load(self, name):
        im1 = Image.open(os.path.join(self.folder_i1, name)).convert('RGB')
        im2 = Image.open(os.path.join(self.folder_i2, name)).convert('RGB')
        lab = Image.open(os.path.join(self.folder_cd, name)).convert('L')  # robust if label saved as RGB
        return im1, im2, lab

    def transform(self, im1, im2, lab, tile=None):
        cs = self.cropsize
        if tile is None:  # random crop (train)
            i = random.randint(0, self.img_h - cs)
            j = random.randint(0, self.img_w - cs)
        else:             # deterministic tile (val / test)
            r, c = divmod(tile, self.tiles_x)
            i, j = r * cs, c * cs

        im1 = TF.crop(im1, i, j, cs, cs)
        im2 = TF.crop(im2, i, j, cs, cs)
        lab = TF.crop(lab, i, j, cs, cs)

        im1 = TF.to_tensor(im1)                        # (3, H, W) float in [0, 1]
        im2 = TF.to_tensor(im2)
        lab = (TF.pil_to_tensor(lab)[0] > 127).long()  # (H, W) {0, 1}; 255 -> 1

        if self.augment:  # same geometric transform for the 3 tensors
            if random.random() < 0.5:
                im1, im2, lab = im1.flip(-1), im2.flip(-1), lab.flip(-1)
            if random.random() < 0.5:
                im1, im2, lab = im1.flip(-2), im2.flip(-2), lab.flip(-2)
            k = random.randint(0, 3)
            if k:
                im1 = torch.rot90(im1, k, dims=(-2, -1))
                im2 = torch.rot90(im2, k, dims=(-2, -1))
                lab = torch.rot90(lab, k, dims=(-2, -1))

        return im1, im2, lab

    def __getitem__(self, idx):
        if self.type == 'train':
            img_idx, tile = idx, None
        else:
            img_idx, tile = divmod(idx, self.tiles_per_image)

        name = self.names[img_idx]
        im1, im2, lab = self._load(name)
        image1, image2, changemap = self.transform(im1, im2, lab, tile)

        if self.get_name:
            return image1, image2, changemap, name if tile is None else f"{name}#tile{tile}"
        return image1, image2, changemap