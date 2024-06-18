import os, sys, pickle, itertools
from datetime import datetime
import numpy as np

spinner = itertools.cycle(['-', '\\', '|', '/'])


def title(text):
    print('-'*len(text))
    print(text)
    print('-'*len(text))


def progress(title=None, perc=None, note=None, str=None):
    """ print progress line """
    if title!=None and perc!=None and note!=None:
        msg = '{:2s} {:20s} [{:3.2f}%] [ {:s} ]'.format(next(spinner), title, perc*100.0, note)
    elif title!=None and perc==None and note!=None:
        msg = '{:2s} {:20s} [ {:20s} ]'.format(next(spinner), title, note)
    elif title!=None and perc!=None and note==None:
        msg = '{:2s} {:20s} [{:3.2f}%]'.format(next(spinner), title, perc*100.0)
    print(str + msg, end='')
    str = '\b' * len(msg)
    return str


def make_folders(params):
    """ create folders for training """
    check_path(params['path']['models'])
    check_path(params['path']['results'])


def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_abspath(f, ref):
    return os.path.join(ref, f)


def get_feature_fname(fname, tpath, ext='cpickle'):
    fn = os.path.split(fname)[1]
    return os.path.join(tpath, os.path.splitext(fn)[0]+'.'+ext)


def save_data(fname, data):
    """ save data into pickle file """
    pickle.dump(data, open(fname,'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_data(fname):
    """ load data from pickle file """
    return pickle.load(open(fname,'rb'))


def get_datetime():
    """ get current date & time for logging """
    now = datetime.now()
    return now.strftime('[%H:%M:%S %d/%m/%y]: ')


def print_head(text, fid=None):
    now = get_datetime()
    if fid is not None:
        fid.write(now + text + '\n')
    print(now + text)


def print_text(text, fid=None):
    if fid is not None:
        fid.write(text + '\n')
    print(text)
