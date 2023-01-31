import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms


class FASDataset(Dataset):
    """ A data loader for Face PAD where samples are organized in this way
    Args:
        root_dir (string): Root directory path
        csv_file (string): csv file to dataset annotation
        map_size (int): size of pixel-wise binary supervision map. The paper uses map_size=14
        transform: A function/transform that takes in a sample and returns a transformed version
        smoothing (bool): Use label smoothing
    """
    def __init__(self, root_dir, csv_file, depth_map_size, transform, smoothing,depth_map_default):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.depth_map_size = depth_map_size
        self.transform = transform
        
        if smoothing:
            self.label_weight = 1.0
        else:
            self.label_weight = 0.99


    def __getitem__(self, index):
        """ Get image, output map and label for a given index
        Args:
            index (int): index of image
        Returns:
            img (PIL Image): 
            mask: output map (32x32)
            label: 1 (genuine), 0 (fake) 
        """
        img_name = self.data.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_name)

        img = Image.open(img_path)

        label = self.data.iloc[index, 1].astype(np.float32)
        label = np.expand_dims(label, axis=0)

        if label == 1:
            if self.depth_map_default==1:
                depth_map = np.ones((self.depth_map_size[0], self.depth_map_size[1]), dtype=np.float32) * self.label_weight
            else:
                depth_name = self.data.iloc[index, 2]
                depth_path = os.path.join(self.root_dir, depth_name)
                # depth_path=os.path.join(self.root_dir, f"depth/{img_name.split('/')[1]}")
                depth_map =Image.open(depth_path)
                depth_map=np.array(depth_map)
        else:
            depth_map = np.ones((self.depth_map_size[0], self.depth_map_size[1]), dtype=np.float32) * (1.0 - self.label_weight)

        if self.transform:
            img = self.transform(img)

        return img, depth_map, label

    
    def __len__(self):
        return len(self.data)