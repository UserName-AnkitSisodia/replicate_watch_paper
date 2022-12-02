import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from shutil import copyfile

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1

def get_img_size(dataset):
    """Return the correct image size."""
    return (3, 128, 128) 

def get_background(dataset):
    """Return the image background color."""
    return COLOUR_WHITE

def split_dataset(model_name):
    """Split the dataset."""

    root=os.path.join(DIR, '../data/watches/') 
    data = os.path.join(root, "christies.npz")

    dataset_zip = np.load(data)
    watches = dataset_zip['watches']
 
    wtp= dataset_zip['wtp']
    location = dataset_zip['location']
    brand = dataset_zip['brand']
    circa = dataset_zip['circa']
    movement = dataset_zip['movement']
    diameter = dataset_zip['diameter']
    material = dataset_zip['material']
    timetrend = dataset_zip['timetrend']
    modelname = dataset_zip['modelname']
    filenames = dataset_zip['filenames']
   
    sequence = np.arange(0,filenames.shape[0])
    df = pd.DataFrame(data=np.column_stack((sequence,modelname,filenames)),columns=['seq','model','file'])
    df['model'] = df['model'].str.encode('ascii', 'ignore').str.decode('ascii')

    df_mod = df.groupby(['model'])["seq"].count().reset_index(name="count")
    r = np.random.uniform(size=df_mod.shape[0])
    r = np.where(r>=0.9,1,0) ## split ratio
    df_mod['r'] = r.tolist()
    result = pd.merge(df, df_mod, on="model")
    train_idx = result[result['r']==0]
    valid_idx = result[result['r']==1]
    train_idx = train_idx['seq'].to_numpy()
    valid_idx = valid_idx['seq'].to_numpy()
    train_idx = train_idx.astype(np.int)
    valid_idx = valid_idx.astype(np.int)
    np.savez( os.path.join(DIR, "../results",model_name,"christies_train.npz"),watches=watches[train_idx,:,:,],wtp=wtp[train_idx],diameter=diameter[train_idx],timetrend=timetrend[train_idx],location=location[train_idx,],brand=brand[train_idx,],circa=circa[train_idx,],movement=movement[train_idx,],material=material[train_idx,],filenames=filenames[train_idx])
    np.savez( os.path.join(DIR, "../results",model_name,"christies_validation.npz"),watches=watches[valid_idx,:,:,],wtp=wtp[valid_idx],diameter=diameter[valid_idx],timetrend=timetrend[valid_idx],location=location[valid_idx,],brand=brand[valid_idx,],circa=circa[valid_idx,],movement=movement[valid_idx,],material=material[valid_idx,],filenames=filenames[valid_idx])
    copyfile(os.path.join(root, "christies_test1.npz"),os.path.join(DIR, "../results",model_name,"christies_test1.npz"))
    copyfile(os.path.join(root, "christies_test2.npz"),os.path.join(DIR, "../results",model_name,"christies_test2.npz"))
    return 0

def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128,eval_batchsize=10000,model_name="temp",sup_signal="brand",logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset :   Name of the dataset to load
    root : str  Path to the dataset root. If `None` uses the default one.
    kwargs :    Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available

    # temp = split_dataset(model_name)

    Train_Dataset = Watches(split="all",model_name=model_name,sup_signal=sup_signal,logger=logger)
    # Validation_Dataset = Watches(split="validation",model_name=model_name,logger=logger)
    Test1_Dataset = Watches(split="test1",model_name=model_name,sup_signal=sup_signal,logger=logger)
    Test2_Dataset = Watches(split="test2",model_name=model_name,sup_signal=sup_signal,logger=logger)

    train_loader = DataLoader(Train_Dataset,batch_size=batch_size,shuffle=True,pin_memory=pin_memory,**kwargs)
    # validation_loader = DataLoader(Validation_Dataset,batch_size=eval_batchsize,shuffle=True,pin_memory=pin_memory,**kwargs)
    test1_loader = DataLoader(Test1_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs)
    test2_loader = DataLoader(Test2_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs)
    train_loader_all = DataLoader(Train_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs) 
    train_loader_one = DataLoader(Train_Dataset,batch_size=1,shuffle=True,pin_memory=pin_memory,**kwargs)
    
    return train_loader, test1_loader, test2_loader, train_loader_all, train_loader_one 

class Watches(Dataset):
    """
    """
    files = {"train": "christies_train.npz", "validation": "christies_validation.npz", "test1": "christies_test1.npz", "test2": "christies_test2.npz", "all":"christies.npz"} 
    img_size = (3, 128, 128)
    background_color = COLOUR_WHITE
    def __init__(self, root=os.path.join(DIR, '../data/watches'), transforms_list=[transforms.ToTensor()], logger=logging.getLogger(__name__), split="train",model_name="temp",sup_signal="brand",**kwargs):
        self.model_name = model_name
        self.sup_signal = sup_signal
        self.data = os.path.join(DIR,'../data/watches', type(self).files[split])

        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        dataset_zip = np.load(self.data)
        self.imgs = dataset_zip['watches']
        self.location = dataset_zip['location']
        self.brand = dataset_zip['brand']
        self.circa = dataset_zip['circa']
        self.movement = dataset_zip['movement']
        self.diameter = dataset_zip['diameter']
        self.diameter=(self.diameter-np.mean(self.diameter))/np.std(self.diameter)
        self.material = dataset_zip['material']
        self.timetrend = dataset_zip['timetrend']
        self.timetrend=(self.timetrend-np.mean(self.timetrend))/np.std(self.timetrend)
        # self.modelname = dataset_zip['modelname']
        self.filenames = dataset_zip['filenames']


        if self.sup_signal == 'brand':
           self.wtp = self.brand
           self.wtp = np.argmax(self.wtp,axis=1)
        elif self.sup_signal == 'circa':
           self.wtp = self.circa
           self.wtp = np.argmax(self.wtp,axis=1)
        elif self.sup_signal == 'movement':
           self.wtp = self.movement
           self.wtp = np.argmax(self.wtp,axis=1)
        elif self.sup_signal == 'material':
           self.wtp = self.material
           self.wtp = np.argmax(self.wtp,axis=1)
        elif self.sup_signal == 'location':
           self.wtp = self.location
           self.wtp = np.argmax(self.wtp,axis=1)
        elif self.sup_signal == 'timetrend':
           self.wtp = self.timetrend
           self.wtp = np.argmax(self.wtp,axis=1)
        elif self.sup_signal == 'price':
           self.wtp = dataset_zip['wtp']
           self.wtp = (self.wtp-np.mean(self.wtp))/np.std(self.wtp)


    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        img = self.transforms(self.imgs[idx])
        wtp = self.wtp[idx]
        location = self.location[idx] 
        brand = self.brand[idx]
        circa = self.circa[idx]
        movement = self.movement[idx]
        diameter = self.diameter[idx]
        material = self.material[idx]
        timetrend = self.timetrend[idx]
        filenames = self.filenames[idx]
        return img, 0, wtp, location, brand, circa, movement, diameter, material, timetrend, filenames

# HELPERS
def preprocess(root, size=(128, 128), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)
