import cv2
import numpy as np

import os
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    def __init__(
            self, 
            images_path=None, 
            frame_size=None,
            mean=None,
            std=None,
            polygon=None,
            crop=None,
            transpose=True
    ):
        self.images = self.dataset(images_path)
        self.frame_size = frame_size
        self.mean = mean
        self.std = std
        self.polygon = polygon
        self.crop = crop
        self.transpose = transpose


    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images[i])
        if image is None:
            raise ValueError(f"Image {self.images[i]} is not found")
        # mask river surface
        if self.masking:
            image = self.masking(image)
        # crop image to remove black regions
        if self.cropping:
            image = self.cropping(image)
        # resize image
        if self.frame_size:
            image = cv2.resize(image, self.frame_size, interpolation = cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # normalize image
        if not self.mean is None:
                image = self.normalizing(image)
        # transpose image
        if self.transpose:
            image = self.transposing(image)
        return image
    

    def dataset(self, images_path):
        # List all files in the directory
        all_files = os.listdir(images_path)
        full_paths = [os.path.join(images_path, file) for file in all_files]

        return sorted(full_paths)


    def masking(self, x):
        color = [255,255,255]
        stencil = np.zeros(x.shape).astype('uint8')
        stencil = cv2.fillPoly(stencil, self.polygon, color)
        return cv2.bitwise_and(x, stencil)


    def cropping(self, x):
        return x[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], :]
        

    def normalizing(self, x):
        if x.max() > 1:
            x = x / 255.0
        x = x - self.mean
        x = x / self.std
        return x


    def transposing(self, x):
        return x.transpose(2, 0, 1).astype('float32')
        
    
    def __len__(self):
        return len(self.images)





    


    


    


