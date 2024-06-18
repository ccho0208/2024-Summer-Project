# CNN Pipeline Framework: CIFAR10
#
#  1) CIFAR10 dataset: Custom (PNG files)
#  2) Dataloader: Custom
#     - Define a dataset class for custom PyTorch dataloader
#     - Use default transform components: ToTensor & Normalize
#  3) Data augmentation: RandomCrop, RandomHorizontalFlip
#  4) Learning rate: 3 (Pytorch supported) + 1 (Custom) schedulers added
#     - CosineAnnealingLR, CyclicLR, CosineAnnealingWarmRestarts
#     - Custom CosineAnnealingWarmUpRestarts
#  5) Train w/ train-set & Val w/ test-set
#  6) Others
#     - Save the best model
#     - Logging
#     - GPU used
#
import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn

from proc.data_processing import *
from proc.data_loader2 import ImageDataset
from proc.model import *
from proc.learning import *


def main():
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    title('Image Classification Pipeline: CIFAR10')

    #
    # 1) Set Parameters ------------------------------------------------------------------------------------------------
    # load parameters from config file (YAML) and do pre-processing
    fname = os.path.join(os.path.dirname(__file__), 'config.yaml')
    params = load_parameters(fname)
    params = process_parameters(params)
    make_folders(params)

    # get meta data from meta files
    params['general']['meta_train'] = fetch_meta_data(params['general']['fname_train'], params['path']['data_train'])
    params['general']['meta_test'] = fetch_meta_data(params['general']['fname_test'], params['path']['data_test'])
    params['general']['labels'] = get_event_labels(params['general']['meta_train'])

    # parameters
    f_resume = False                                                    # flag: to resume training from checkpoint
    start_epoch = 0                                                     # start from epoch 0 or last checkpoint epoch

    # log-system
    fid = open(params['general']['log_fname'], 'wt')
    params['general']['log_fid'] = fid

    # print train/test meta info
    print_head('Set parameters', fid)
    log_parameters(params, fid)


    #
    # 2) Prep Data-loaders ---------------------------------------------------------------------------------------------
    print_head('Prepare data-loaders', fid)

    # transforms
    if params['general']['f_augment']:
        train_trans = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
        print_text('[INFO] data-augment being used', fid)
    else:
        train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
        print_text('[INFO] NO data-augment used', fid)

    test_trans = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])

    # train-set
    train_dataset = ImageDataset(params['general']['meta_train'], params['general']['labels'], train_trans)
    train_loader = DataLoader(train_dataset, batch_size=params['classifier']['batch_size'], shuffle=True, num_workers=2)

    # test-set
    test_dataset = ImageDataset(params['general']['meta_test'], params['general']['labels'], test_trans)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    print_text('[INFO] custom data-loaders for train-set and test-set ready')


    #
    # 3) Define CNN model and loss/optimizer ---------------------------------------------------------------------------
    print_head('Define CNN model and loss function/optimizer', fid)

    # 3-1) define CNN architecture
    net = define_cnn_model(params['classifier']['model'])

    fn_model = os.path.join(params['path']['models'],'model_ckpt_'+params['classifier']['model']+'.pt')
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        print_text('[INFO] GPU {:d} being used'.format(torch.cuda.device_count()), fid)

    if f_resume:
        print_text('[INFO] resuming from saved checkpoint', fid)
        assert os.path.isdir(params['path']['checkpoint']), 'Error: No checkpoint directory found'
        checkpoint = torch.load(fn_model)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        net.load_state_dict(checkpoint['net'])

    # 3-2) specify loss function, optimizer, and learn-rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=params['classifier']['lr'],
                          momentum=0.9,
                          weight_decay=5e-4)

    if params['classifier']['f_lr']:
        # CosineAnnealingLR
        if params['classifier']['lr_scheduler'] == 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=int(params['classifier']['epochs']/5))
            print_text('[INFO] learn-rate scheduler: CosineAnnealingLR', fid)

        # CyclicLR
        elif params['classifier']['lr_scheduler'] == 2:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,
                                                          step_size_up=int(params['classifier']['epochs']/10),
                                                          mode='triangular2')
            print_text('[INFO] learn-rate scheduler: CyclicLR', fid)

        # CosineAnnealingWarmRestarts
        elif params['classifier']['lr_scheduler'] == 3:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=5, T_mult=2, eta_min=0.001)
            print_text('[INFO] learn-rate scheduler: CosineAnnealingWarmRestarts', fid)

        # Custom CosineAnnealingWarmUpRestarts
        # params['classifier']['lr'] for `optimizer` should be zero
        elif params['classifier']['lr_scheduler'] == 4:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmUpRestarts(optimizer,
                                                                               T_0=50, T_mult=2, eta_max=0.1,
                                                                               T_up=1, gamma=0.5)
            print_text('[INFO] learn-rate scheduler: Custom CosineAnnealingWarmUpRestarts', fid)
        else:
            raise IOError('Error: Incorrect learn-rate chosen [%d]' % params['classifier']['lr_scheduler'])
    else:
        print_text('[INFO] NO learn-rate scheduler', fid)


    #
    # 4) Train the network ---------------------------------------------------------------------------------------------
    print_head('Train the network', fid)
    train_loss, val_loss, train_acc, val_acc, val_loss_min = [], [], [], [], np.Inf
    for epoch in range(start_epoch, start_epoch+params['classifier']['epochs']):
        # 4-1) training
        train_loss_t, train_acc_t = run_train(net, train_loader, criterion, optimizer, device)

        # 4-2) validation
        val_loss_t, val_acc_t = run_validate(net, test_loader, criterion, device)

        # 4-3) update learning rate
        if params['classifier']['f_lr']:
            scheduler.step()

        # 4-4) save average losses & accuracies
        train_loss.append(train_loss_t)
        val_loss.append(val_loss_t)
        train_acc.append(train_acc_t)
        val_acc.append(val_acc_t)

        # 4-5) save model only if val-acc has increased
        s = '.epoch [{:3d}/{}], train-acc: {:.4f}, val-acc: {:.4f}, train-loss: {:.5f}, val-loss: {:.5f}, lr: {:.5f}'. \
            format(epoch+1, params['classifier']['epochs'],
                   train_acc_t, val_acc_t, train_loss_t, val_loss_t, optimizer.param_groups[0]['lr'])

        if val_loss_t < val_loss_min:
            val_loss_min = val_loss_t
            state = {'epoch': epoch, 'acc': val_acc_t, 'net': net.state_dict()}
            torch.save(state, fn_model)
            print_text(s+' [CKPT]', fid)
        else:
            print_text(s, fid)


    #
    # 5) Test the network ----------------------------------------------------------------------------------------------
    print_head('Test the network', fid)

    # load the best trained model
    checkpoint = torch.load(fn_model)
    net.load_state_dict(checkpoint['net'])

    # test the network with the best model
    correct, total = run_test(net, test_loader, device)
    print_text('[EVAL] accuracy on the {:d} test images: {}%'.format(total,100*correct/total), fid)
    print_text('[EVAL] {} / {}'.format(total-correct,total), fid)
    print_text('', fid)

    # save training curves
    fn_fig = os.path.join(params['path']['results'], 'train_curves_'+params['classifier']['model']+'.png')
    print_text('[INFO] save training curves at {}'.format(fn_fig))
    summarize_learning(train_loss, val_loss, train_acc, val_acc, fn_fig, params['general']['test_id'])

    elapsed = time.time() - start
    print_head('Elapsed {:.2f} minutes'.format(elapsed/60.0), fid)
    fid.close()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError,IOError) as e:
        sys.exit(e)
