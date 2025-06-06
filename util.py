import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A

from skimage import measure
from sklearn.metrics import auc

import time, os, logging, random
import os.path as osp
from glob import glob
import numpy as np
from PIL import Image

def fix_randseed(seed):
    """ 
        Set random seeds for reproducibility 
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)# use deterministic algorithms.
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)

    loader_dict = { 'worker_init_fn':seed_worker, 'generator':g}
    return loader_dict

class PadToSquare:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        img = TF.resize(img, (new_h, new_w))

        # Pad to center
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top

        img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        return img

class MOAIDataloader(Dataset):
    """
        MOAI Dataset.
        x_size is input image size, y_size is output mask size.
    """
    def __init__(self, mode='test', x_size=1024, y_size=256, datapath='../moai', hyp=None) -> None:
        super().__init__()
        self.mode = mode
        
        self.datapath = datapath
        self.img_mean = [0.485, 0.456, 0.406] 
        self.img_std = [0.229, 0.224, 0.225]
        
        extensions = ['bmp', 'png', 'jpg', 'jpeg', 'tif', 'tiff']

        # self.files = sorted(sum([glob(osp.join(self.datapath, 'good', f'*.{ext}')) for ext in extensions],[]))
        
        if mode == 'train':
            self.files = sorted(sum([glob(osp.join(self.datapath, 'good',f'*.{ext}')) for ext in extensions],[]))
            self.aug = A.Compose([
                        A.HorizontalFlip(p=hyp.hflip), A.VerticalFlip(p=hyp.vflip), A.RandomRotate90(p=hyp.rotate90),
                        A.ColorJitter(brightness=hyp.brightness, contrast=hyp.contrast, saturation=hyp.saturation, hue=hyp.hue)
                        #A.RGBShift(p=0.5),
                        #A.CLAHE(p=0.5),
                        #A.ChannelShuffle(p=0.5)
                        ])
        elif mode == 'val':
            self.files = sorted(sum([glob(osp.join(self.datapath, 'good',f'*.{ext}')) for ext in extensions],[]))
        elif mode == 'test':
            self.files = sorted(sum([glob(osp.join(self.datapath, f'*.{ext}')) + 
                                     glob(osp.join(self.datapath, '*', f'*.{ext}'))
                                     for ext in extensions], []))

        self.img2tensor = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((x_size,x_size), interpolation=Image.BILINEAR),
                                            # PadToSquare(x_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)
                                            ])
        self.label2tensor = transforms.Compose([transforms.ToPILImage(), 
                                                transforms.Resize((y_size,y_size), interpolation=Image.NEAREST),
                                                transforms.ToTensor()])

    def __getitem__(self, index):
        # neg_files
        file_path = self.files[index]

        fname = file_path.rsplit('.', 1)[0].rsplit('/', 1)[1]
        # fname = file_path.rsplit('.', 1)[0].rsplit('\\', 1)[1]
        
        img = np.array(Image.open(file_path).convert('RGB'))
        label = np.zeros(img.shape[:2], dtype=np.float32)# h,w
        if self.mode == 'test':
            label_path = file_path.replace('valid', 'ground_truth')
            label_path = label_path.rsplit('.', 1)[0] + '_mask.png'
            if osp.exists(label_path):
                label = np.array(Image.open(label_path).convert('L'), dtype=np.float32)/255.
        if self.mode == 'train' :
            augresults = self.aug(image=img, mask=label)
            img, label = augresults['image'], augresults['mask']
        img, label = self.img2tensor(img), self.label2tensor(label)
        return {'images': img, 'labels': label.round().int(), 'fname': fname}

    def __len__(self):
        return len(self.files)


def kornia_filter2d(input, kernel):
    """
        conv2d function from kornia.
    """
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    tmp_kernel = tmp_kernel.expand(-1, 64, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output


class BatchDiffLoss(nn.Module):
    """
        Difference Loss within a batch.
    """
    def __init__(self, batchsize, p=2) -> None:
        super().__init__()
        self.idx0, self.idx1 = np.triu_indices(n=batchsize, k=1)
        self.p = p
    def forward(self, pyramid):
        diffes = []
        for input in pyramid:
            diff = (input[self.idx0] - input[self.idx1]).abs()**self.p
            diffes.append(diff)
        return diffes

def getLogger(work_dir):
    time_now = time.strftime('%Y-%m-%d-%H.%M', time.localtime())
    logger = logging.getLogger(work_dir+f'_{time_now}')
    logger.setLevel(logging.INFO)  # Logging level
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    # FileHandler
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(osp.join(work_dir, f"{time_now}.txt"), mode='w')
    fHandler.setLevel(logging.DEBUG) 
    fHandler.setFormatter(formatter) 
    logger.addHandler(fHandler) 
    return logger, time_now


def compute_pro_score_fast(_amaps, _masks):
    """
        Compute PRO score fastly
            amaps: b,h,w
            masks: b,h,w
    """
    amaps = (_amaps-_amaps.min())/(_amaps.max()-_amaps.min())
    amaps, masks = amaps.cpu().numpy(), _masks.cpu().numpy()
    pro_lst, fpr_lst = [], []
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    max_step = 200
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / max_step
    weights, num_pros = [], 0
    for mask in masks:
        label = measure.label(mask, background=0)
        weight = np.zeros_like(label, dtype=np.float32)
        for k in range(1, np.max(label)+1):
            weight_mask = (label==k)
            weight[weight_mask] = 1/np.sum(weight_mask)
        weights.append(weight)
        num_pros += np.max(label)
    weights = np.stack(weights, axis=0)/num_pros
    inverse_masks = 1 - masks
    for th in np.arange(min_th, max_th, delta):
        binary_amaps = np.where(amaps>th, 1, 0)
        pro = (binary_amaps*weights).sum()
        pro_lst.append(pro)
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()
        fpr_lst.append(np.mean(fpr))
    return auc(fpr_lst, pro_lst)
