# Find mean and standard deviation for Dataset Normalization
#
#  CiFAR10: Default dataset from `torchvision`
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from proc.data_processing import *
from proc.data_loader import *


def main():
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #
    # 1) Set Parameters ------------------------------------------------------------------------------------------------
    f_choose = 2


    #
    # 2) Prep Data-loaders ---------------------------------------------------------------------------------------------
    if f_choose == 1:
        dpath = os.path.join(HOME, 'Unzipped/0_prac/2_pytorch/b1_resnet/data')

        trans = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.CIFAR10(root=dpath, train=True, download=True, transform=trans)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), num_workers=1)

        # iter() calls the __iter__() method on the `loader`` which returns an iterator.
        # next() then calls the __next__() method on that iterator to get the first iteration.
        # Running next() again will get the second item of the iterator, etc.
        data = next(iter(loader))

        # convert torch to numpy ndarray
        data2 = data[0].numpy()

        # save histogram
        plot_histogram(data2, 'Histogram: CiFAR10')

    elif f_choose == 2:
        dpath = os.path.join(HOME, 'Unzipped/0_prac/2_pytorch/b1_resnet')

        # load parameters from config file (YAML) and do pre-processing
        fname = os.path.join(dpath, 'config.yaml')
        params = load_parameters(fname)
        params = process_parameters(params)

        # get meta data from meta files
        params['general']['meta_train'] = fetch_meta_data(params['general']['fname_train'], params)
        params['general']['labels'] = get_event_labels2(params['general']['meta_train'])

        trans = transforms.Compose([transforms.ToTensor()])
        dataset = ImageDataset(params['general']['meta_train'], params['general']['labels'], trans)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), num_workers=1)

        # convert torch to numpy ndarray
        data = next(iter(loader))
        data2 = data['feat'].numpy()

        # save histogram
        plot_histogram(data2, 'Histogram: CiFAR10: Custom DataLoader')

    # print out the means and stds
    print('mean: {:.4f}, {:.4f}, {:.4f}'.format(np.mean(data2[:,0,:,:]),np.mean(data2[:,1,:,:]),np.mean(data2[:,2,:,:])))
    print('std: {:.4f}, {:.4f}, {:.4f}'.format(np.std(data2[:,0,:,:]),np.std(data2[:,1,:,:]),np.std(data2[:,2,:,:])))

    elapsed = time.time() - start
    print('Elapsed {:.2f} minutes'.format(elapsed/60.0))
    return 0


def plot_histogram(data, str=''):
    data_1d = data.flatten()
    n, bins, patches = plt.hist(x=data_1d, bins=50, color='#0504aa', alpha=0.7, rwidth=0.9)
    plt.xlabel('value'); plt.ylabel('frequency'); plt.grid()
    plt.title(str)
    plt.show()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,IOError) as e:
        sys.exit(e)
