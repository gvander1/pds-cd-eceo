"""Change-detection datasets: Hi-UCD mini and LEVIR-CD.
 
HiUCD_Dataset
  everything=True  -> (im1, im2, changemap, landcover1, landcover2)
  everything=False -> (im1, landcover1)                                (default)
LEVIRCD_Dataset    -> (im1, im2, changemap)
get_name=True appends the file name (+ "#tileK" for LEVIR val/test) at the end.
 
Augmentation (same default for both): augment=None -> on for 'train', off for 'val'/'test'.
Random flips H/V + rotation 0/90/180/270, applied identically to every image and label map.
 
Randomness (crops + augmentation) uses the torch RNG only, so a single
torch.manual_seed(0) right before iterating a loader reproduces exactly the same samples.
"""
import os
 
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
 
# Override with: export BCD_DATA_FOLDER=/path   /   export LEVIR_ROOT=/path
HiUCD_ROOT = os.getenv("BCD_DATA_FOLDER", "/data/gaetane")
LEVIR_ROOT = os.getenv("LEVIR_ROOT", "/data/gaetane/LEVIR-CD")
 
SPLITS = ("train", "val", "test")
 
 
# ---------------------------------------------------------------- shared helpers
 
def list_png(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
 
 
def check_pairs(names1, names2, names_cd, where):
    """T1 / T2 / label folders must hold exactly the same file names, otherwise the
    i-th files of the three sorted lists are not the same tile and pairs get mixed up."""
    if not (names1 == names2 == names_cd):
        diff = (set(names1) ^ set(names2)) | (set(names1) ^ set(names_cd))
        raise ValueError(f"T1 / T2 / label file names do not match in {where}, e.g. {sorted(diff)[:5]}")
    if not names1:
        raise ValueError(f"No .png file found in {where}")
 
 
def paired_augment(*tensors):
    """Same random flips / 90° rotation for all tensors (images (3,H,W) and maps (H,W))."""
    if torch.rand(1).item() < 0.5:
        tensors = [t.flip(-1) for t in tensors]
    if torch.rand(1).item() < 0.5:
        tensors = [t.flip(-2) for t in tensors]
    k = int(torch.randint(0, 4, (1,)))
    if k:
        tensors = [torch.rot90(t, k, dims=(-2, -1)) for t in tensors]
    return tuple(tensors)
 
 
# ---------------------------------------------------------------- Hi-UCD mini
 
HIUCD_FOLDERS = {
    "train": ("HiUCD_mini/train/image/2017/9", "HiUCD_mini/train/image/2018/9", "HiUCD_mini/train/mask_merge/2017_2018/9"),
    "val":   ("HiUCD_mini/val/image/2017/9",   "HiUCD_mini/val/image/2018/9",   "HiUCD_mini/val/mask_merge/2017_2018/9"),
    # NOTE: test is a different time pair (2018 -> 2019) than train / val (2017 -> 2018)
    "test":  ("HiUCD_mini/test/image/2018/9",  "HiUCD_mini/test/image/2019/9",  "HiUCD_mini/test/mask_merge/2018_2019/9"),
}
 
 
class HiUCD_Dataset(Dataset):
    # cropsize modifié à 224 pour que ce soit un multiple de 14 pour l'encoder DINOv2
    def __init__(self, cropsize=224, type="train", root=HiUCD_ROOT, augment=None,
                 small=False, size=10, everything=False, get_name=False):
        if type not in SPLITS:
            raise ValueError(f"type must be one of {SPLITS}, got {type!r}")
 
        f1, f2, fcd = (os.path.join(root, f) for f in HIUCD_FOLDERS[type])
        names1, names2, names_cd = list_png(f1), list_png(f2), list_png(fcd)
        check_pairs(names1, names2, names_cd, os.path.dirname(f1))
        names = names1[:size] if small else names1
 
        self.im1 = [os.path.join(f1, n) for n in names]
        self.im2 = [os.path.join(f2, n) for n in names]
        self.lab = [os.path.join(fcd, n) for n in names]
 
        self.type = type
        self.cropsize = cropsize
        self.everything = everything
        self.get_name = get_name
        self.augment = (type == "train") if augment is None else augment
 
    def fixed_domain_shift(self, img):
        """Kept for domain-shift experiments (not used by default)."""
        img = TF.adjust_brightness(img, 1.2)  # brighter
        img = TF.adjust_contrast(img, 1.2)    # higher contrast
        img = TF.adjust_saturation(img, 2)    # more saturated
        img = TF.adjust_hue(img, -0.05)       # slight hue shift
        return img
 
    def transform(self, im1, im2, lab):
        # Random crop (all splits; seed the torch RNG before iterating for reproducible val)
        i, j, h, w = v2.RandomCrop.get_params(im1, output_size=(self.cropsize, self.cropsize))
        im1 = TF.to_tensor(TF.crop(im1, i, j, h, w))
        im2 = TF.to_tensor(TF.crop(im2, i, j, h, w))
        lab = TF.to_tensor(TF.crop(lab, i, j, h, w))  # mask NOT converted: keeps class ids
        return im1, im2, lab
 
    def __len__(self):
        return len(self.im1)
 
    def __getitem__(self, idx):
        image1 = Image.open(self.im1[idx]).convert("RGB")  # no-op if already RGB
        image2 = Image.open(self.im2[idx]).convert("RGB")
        masks = Image.open(self.lab[idx])
        image1, image2, masks = self.transform(image1, image2, masks)
 
        # *255 recovers the uint8 ids exactly (checked: float32 round-trip is exact for 0..255)
        landcover1 = (masks[0] * 255).long()
        landcover2 = (masks[1] * 255).long()
        changemap = ((masks[2] * 255) == 2).int()  # binary: 1 = change
 
        if self.augment:
            image1, image2, changemap, landcover1, landcover2 = paired_augment(
                image1, image2, changemap, landcover1, landcover2)
 
        if self.everything:
            out = (image1, image2, changemap, landcover1, landcover2)
        else:
            out = (image1, landcover1)
        if self.get_name:
            out = out + (os.path.basename(self.im1[idx]),)
        return out
 
 
# ---------------------------------------------------------------- LEVIR-CD
 
class LEVIRCD_Dataset(Dataset):
    """Binary change detection dataset for LEVIR-CD (1024x1024 RGB, 0.5 m/px).
 
    Layout: root/{train,val,test}/{A,B,label}/*.png   (445 / 64 / 128 pairs)
    A = t1, B = t2, label: 0 = no change, 255 = building change.
 
    Returns (image1, image2, changemap) [+ name if get_name]
      image1, image2 : float (3, cropsize, cropsize) in [0, 1]
      changemap      : long (cropsize, cropsize) in {0, 1}
 
    Cropping:
      - train      : one random crop per image per epoch
      - val / test : deterministic grid of non-overlapping tiles (1024 / 256 -> 16 per image)
    """
 
    EXCLUDE = {"train_76.png"}  # label unreadable on the workstation disk
 
    def __init__(self, cropsize=256, type="train", root=LEVIR_ROOT, augment=None,
                 small=False, size=10, get_name=False):
        if type not in SPLITS:
            raise ValueError(f"type must be one of {SPLITS}, got {type!r}")
 
        split_root = os.path.join(root, type)
        self.folder_i1 = os.path.join(split_root, "A")
        self.folder_i2 = os.path.join(split_root, "B")
        self.folder_cd = os.path.join(split_root, "label")
 
        names1 = list_png(self.folder_i1)
        check_pairs(names1, list_png(self.folder_i2), list_png(self.folder_cd), split_root)
        names = [n for n in names1 if n not in self.EXCLUDE]
        self.names = names[:size] if small else names
 
        self.type = type
        self.cropsize = cropsize
        self.get_name = get_name
        self.augment = (type == "train") if augment is None else augment
 
        with Image.open(os.path.join(self.folder_i1, self.names[0])) as im:  # all images assumed same size
            self.img_w, self.img_h = im.size
        if cropsize > min(self.img_w, self.img_h):
            raise ValueError(f"cropsize={cropsize} larger than image size {self.img_w}x{self.img_h}")
 
        self.tiles_y = self.img_h // cropsize  # pixels beyond the last full tile are dropped
        self.tiles_x = self.img_w // cropsize
        self.tiles_per_image = 1 if type == "train" else self.tiles_y * self.tiles_x
 
    def __len__(self):
        return len(self.names) * self.tiles_per_image
 
    def _load(self, name):
        im1 = Image.open(os.path.join(self.folder_i1, name)).convert("RGB")
        im2 = Image.open(os.path.join(self.folder_i2, name)).convert("RGB")
        lab = Image.open(os.path.join(self.folder_cd, name)).convert("L")  # robust if label saved as RGB
        return im1, im2, lab
 
    def transform(self, im1, im2, lab, tile=None):
        cs = self.cropsize
        if tile is None:  # random crop (train)
            i = int(torch.randint(0, self.img_h - cs + 1, (1,)))
            j = int(torch.randint(0, self.img_w - cs + 1, (1,)))
        else:             # deterministic tile (val / test)
            r, c = divmod(tile, self.tiles_x)
            i, j = r * cs, c * cs
 
        im1 = TF.to_tensor(TF.crop(im1, i, j, cs, cs))
        im2 = TF.to_tensor(TF.crop(im2, i, j, cs, cs))
        lab = (TF.pil_to_tensor(TF.crop(lab, i, j, cs, cs))[0] > 127).long()  # 255 -> 1
 
        if self.augment:
            im1, im2, lab = paired_augment(im1, im2, lab)
        return im1, im2, lab
 
    def __getitem__(self, idx):
        if self.type == "train":
            img_idx, tile = idx, None
        else:
            img_idx, tile = divmod(idx, self.tiles_per_image)
 
        name = self.names[img_idx]
        image1, image2, changemap = self.transform(*self._load(name), tile)
 
        if self.get_name:
            return image1, image2, changemap, (name if tile is None else f"{name}#tile{tile}")
        return image1, image2, changemap
 
 
# ---------------------------------------------------------------- entry point for scripts
 
def build_dataset(cfg, split):
    """Used by train.py / evaluate.py: every parameter comes from the config (and is logged to W&B)."""
    if cfg["dataset"] == "hiucd":
        return HiUCD_Dataset(cropsize=cfg["crop"], type=split, root=cfg.get("root", HiUCD_ROOT),
                             augment=(cfg.get("augment") if split == "train" else False), everything=True)
    if cfg["dataset"] == "levir":
        return LEVIRCD_Dataset(cropsize=cfg["crop"], type=split, root=cfg.get("root", LEVIR_ROOT),
                               augment=(cfg.get("augment") if split == "train" else False))
    raise ValueError(f"Unknown dataset {cfg['dataset']!r}")