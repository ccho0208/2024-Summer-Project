from skimage import io

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, meta, labels, transform=None):
        """ Custom dataset for image files
            meta {list}: contains meta info that consists of filenames and labels as {dict: 2}
            labels {dict}: definition of labels used in the dataset
        """
        self.meta = meta
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get filename and its label info
        fname, label = self.meta[idx]['file'], self.meta[idx]['label']

        # read image data & convert label {str} to label {int}
        feat = io.imread(fname)
        label = int(self.labels[label])

        if self.transform:
            feat = self.transform(feat)
        return {'feat':feat, 'label':label}

    def __len__(self):
        return len(self.meta)
