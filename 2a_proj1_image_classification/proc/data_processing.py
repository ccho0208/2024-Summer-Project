import yaml, csv
from proc.files import *


def load_parameters(fname):
    """ load configuration parameters from YAML file """
    if os.path.isfile(fname):
        with open(fname,'r') as f:
            return yaml.safe_load(f)
    else:
        raise IOError('Error: Configuration file not found [%s]' % fname)


def process_parameters(params):
    """ post-process parameters """
    # HOME directory
    if params['path']['home'] == 'local':
        HOME = os.path.expanduser('~')
    elif params['path']['home'] == 'ssd':
        HOME = '/Volumes/DB'
    else:
        raise IOError('Error: setting for HOME folder [%d]' % params['path']['home'])

    params['path']['data_train'] = os.path.join(HOME, params['path']['data_train'])
    params['path']['data_test'] = os.path.join(HOME, params['path']['data_test'])
    params['path']['exp_model'] = os.path.join(HOME, params['path']['exp_model'])
    params['path']['exp_eval'] = os.path.join(HOME, params['path']['exp_eval'])

    params['path']['models'] = os.path.join(params['path']['exp_model'], params['path']['models'])
    params['path']['results'] = os.path.join(params['path']['exp_eval'], params['path']['results'])

    params['general']['fname_train'] = os.path.join(params['path']['data_train'], params['general']['fname_train'])
    params['general']['fname_test'] = os.path.join(params['path']['data_test'], params['general']['fname_test'])
    params['general']['log_fname'] = os.path.join(params['path']['results'], params['general']['log_fname'])
    return params


def log_parameters(params, fid):
    len_train, len_test = len(params['general']['meta_train']), len(params['general']['meta_test'])
    print_text('.train-meta: {}'.format(params['general']['fname_train']), fid)
    print_text('.model-path: {}'.format(params['path']['exp_model']), fid)
    print_text('.test-meta: {}'.format(params['general']['fname_test']), fid)
    print_text('.eval-path: {}'.format(params['path']['exp_eval']), fid)
    print_text('.labels: {}'.format(params['general']['labels']), fid)
    print_text('.train: {}, test: {} and total: {}'.format(len_train, len_test, len_train+len_test), fid)
    print_text('.model-type: {}'.format(params['classifier']['model']), fid)
    print_text('.epochs: {}, batch_size: {}'.format(params['classifier']['epochs'],
                                                    params['classifier']['batch_size']), fid)
    print_text('', fid)


def fetch_meta_data(fname, spath):
    """ get a list of files from a meta file """
    l_data = []
    if os.path.isfile(fname):
        with open(fname, 'rt') as f:
            for row in csv.reader(f, delimiter='\t'):
                l_data.append(
                    {'file':get_abspath(row[0],spath), 'label':row[1]}
                )
    else:
        raise IOError('Error: Configuration file not found [%s]' % fname)
    return l_data


def get_event_labels(data, format=True):
    """ returns a list that contains labels: a list of strings """
    labels = []
    for item in data:
        if 'label' in item and item['label'] not in labels:
            labels.append(item['label'])
    labels.sort()

    if format:
        data = {}
        for idx, item in enumerate(labels):
            data[item] = idx
    else: data = labels
    return data


def get_event_labels2(data):
    """ returns a dict that contains labels (string) and their indices (int) """
    labels = []
    for item in data:
        if 'label' in item and item['label'] not in labels:
            labels.append(item['label'])
    labels.sort()

    data = {}
    for idx, item in enumerate(labels):
        data[item] = idx
    return data
