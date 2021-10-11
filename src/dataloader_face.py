import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import cv2
from torch.utils.data.dataloader import default_collate
from PIL import Image


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, skflist=None, irr_mask_path=None,
                 seg_mask_path=None, fix_mask_path=None, training=True):
        super(FaceDataset, self).__init__()
        self.config = config
        self.training = training
        self.data = self.load_flist(flist)
        self.data = sorted(self.data, key=lambda x: x.split('/')[-1])
        self.sketch_data = self.load_flist(skflist)
        self.sketch_data = sorted(self.sketch_data, key=lambda x: x.split('/')[-1])
        self.irr_mask_data = self.load_flist(irr_mask_path)
        self.seg_mask_data = self.load_flist(seg_mask_path)
        self.fix_mask_data = self.load_flist(fix_mask_path)
        self.mask_rates = []
        if self.training and self.config.mask_rates is not None:
            # accumulate mask rates for different situations
            mskr = 0
            for m in self.config.mask_rates:
                mskr += m
                self.mask_rates.append(mskr)
            assert mskr == 1.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.load_item(index)

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.config.input_size
        # load image
        img = cv2.imread(self.data[index])[:, :, ::-1]
        img = self.resize(img, size, size, center_crop=self.config.center_crop)
        if len(self.sketch_data) == len(self.data):
            sketch = cv2.imread(self.sketch_data[index])[:, :, 0]
        else:
            sketch = None

        # load mask for finetuning
        has_mask = len(self.irr_mask_data) > 0 or len(self.seg_mask_data) > 0 or len(self.fix_mask_data) > 0
        if has_mask:
            if self.training is False:
                mask = cv2.imread(self.fix_mask_data[index], cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.uint8) * 255
            else:  # train mode: 50% mask with random brush, 50% mask with
                if random.random() < 0.5:
                    mask = cv2.imread(np.random.choice(self.irr_mask_data, 1)[0], cv2.IMREAD_GRAYSCALE)
                else:
                    mask = cv2.imread(np.random.choice(self.seg_mask_data, 1)[0], cv2.IMREAD_GRAYSCALE)
                if mask.shape[0] != size or mask.shape[1] != size:
                    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
        else:
            mask = None

        # augment data
        if self.training is True and self.config.flip is True:
            if random.random() < 0.5:
                img = img[:, ::-1, ...]
                if sketch is not None:
                    sketch = sketch[:, ::-1, ...]
                if mask is not None:
                    mask = mask[:, ::-1, ...]

        img = self.to_tensor(img, norm=True)  # norm to -1~1
        meta = {'img': img}
        if sketch is not None:
            sketch = self.sketch_to_tensor(sketch)
            meta['sketch'] = sketch
        if mask is not None:
            mask = self.to_tensor(mask)
            # Set multi-mask for training
            if self.training and len(self.mask_rates) == 3 and has_mask:
                rdv = random.random()
                if rdv < self.mask_rates[0]:
                    mask = torch.zeros_like(mask)
                elif self.mask_rates[0] <= rdv < self.mask_rates[1]:
                    pass
                else:
                    mask = torch.ones_like(mask)
            meta['mask'] = mask
        else:
            meta['mask'] = None
        return meta

    def to_tensor(self, img, norm=False):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        if norm:
            img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img_t

    def sketch_to_tensor(self, sketch):
        sketch = Image.fromarray(sketch)
        sketch_t = F.to_tensor(sketch).float()
        sketch_t = F.normalize(sketch_t, mean=[0.5,], std=[0.5,])
        return sketch_t

    def resize(self, img, height, width, center_crop=False):
        imgh, imgw = img.shape[0:2]

        if center_crop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        if imgh > height and imgw > width:
            inter = cv2.INTER_AREA
        else:
            inter = cv2.INTER_LINEAR
        img = cv2.resize(img, (height, width), interpolation=inter)

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort(key=lambda x: x.split('/')[-1])
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                collate_fn=self.collate_fn
            )

            for item in sample_loader:
                yield item

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        res = {}
        for k in keys:
            temp_ = []
            for b in batch:
                if b[k] is not None:
                    temp_.append(b[k])
            if len(temp_) > 0:
                res[k] = default_collate(temp_)
            else:
                res[k] = None

        return res
