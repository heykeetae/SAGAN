import os
import glob
import torch
from PIL import Image
from os.path import join
from os.path import basename
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.datasets as dsets


class DataLoaderSegmentation(Dataset):
    def __init__(self,path, img_transform=None, mask_transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.mask_transform = mask_transform 
        self.img_files = glob.glob(join(self.path, 'CelebA-HQ-img','*.jpg'))
        self.mask_files = []
        for i, img_path in enumerate(self.img_files):
            img_val = int(img_path.split('/')[-1][:-4])
            self.mask_files.append(join(self.path,'mask','{}.png'.format(img_val)))
    
    def __getitem__(self,index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image,mask
    
    def __len__(self):
        return len(self.mask_files)

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        path = join(self.path, 'CelebA')
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(root=path, transform=transforms)
        return dataset
    
    def load_imagenet(self):
        if self.train:
            path = join(self.path, 'train')
        else:
            path = self.path
        transforms = self.transform(True, True, True, False)
        dataset = dsets.ImageFolder(root=path,transform=transforms)
        return dataset
    
    def load_celebA_semantic(self):
        img_transform = self.transform(True,True,True,False)
        mask_transform = self.transform(True, True, False, False)
        dataset = DataLoaderSegmentation(path,img_transform, mask_transform)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'imagenet':
            dataset = self.load_imagenet()
        elif self.dataset == 'semantic':
            dataset = self.load_custom()
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=True)
        return loader

