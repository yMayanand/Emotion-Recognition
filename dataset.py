import os
import pandas as pd
import numpy as np
from glob import glob
from utils import read_image

class EmotionDataset:
    """This is an Dataset Class for Emotion Recognition

    Args:
        root(str): path to the root directory of the dataset

        split(str): dataset split one of the: (train, valid, test)

    Returns
        (EmotionDataset): the dataset class which return images and labels.
    """

    def __init__(self, root='./', split='train'):
        
        if split == 'train':
            path = os.path.join(root, "FER2013Train")
        elif split == 'valid':
            path = os.path.join(root, "FER2013Valid")
        elif split == 'test':
            path = os.path.join(root, "FER2013Test")
        else:
            raise ValueError(f'{split} is not valid value of split')
        
        self.path = path

        self.image_files = glob(f'{path}/*.png')
        label_file = glob(f'{path}/*.csv')
        self.label_csv = pd.read_csv(label_file[0], header=None)

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        temp = self.label_csv.iloc[idx, :]
        fname = temp[0]

        file_path = os.path.join(self.path, fname)
        image = read_image(file_path)

        label = temp[2:]
        label = np.array(label)
        label = (label > 0).astype('float')
        return image, label


        


