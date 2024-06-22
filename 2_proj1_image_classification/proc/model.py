from pytorch_model_summary import summary
from models import *


def define_cnn_model(model_type, f_print=False):
    if model_type == 'vgg16':
        net = VGG('VGG16')
    elif model_type == 'resnet18':
        net = ResNet18()
    elif model_type == 'lenet':
        net = LeNet()
    else:
        raise IOError('Error: Incorrect model-type [%s]' % model_type)
    if f_print:
        print(summary(net, torch.zeros((1,3,32,32)), show_input=True))
    return net
