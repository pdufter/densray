import argparse
import os
import pickle
import json
import logging


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dump_class(cls, fname):
    assert not os.path.exists(fname), "File " + fname + " exists already."
    outfile = open(fname, 'wb')
    pickle.dump(cls, outfile)


def load_class(fname):
    infile = open(fname, 'rb')
    return pickle.load(infile)


def store(path):
    assert os.path.exists(path) == 0, path + " exists already."
    return open(path, 'w', encoding='utf-8')


def read(path):
    return open(path, 'r', encoding='utf-8')


def dump_dict(dict_, fname):
    assert os.path.exists(fname) == 0, fname + " exists already."
    outfile = open(fname, 'w')
    json.dump(dict_, outfile, default=lambda x: "<not_serializable>")
    outfile.close()


def invdict(dict_):
    if len(set(dict_.values())) != len(dict_.values()):
        raise ValueError("Values in dict are not unique, cannot reverse.")
    return dict([(v, k) for k, v in dict_.items()])


def get_logger(name, filename, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fh = logging.FileHandler(filename)
    ch = logging.StreamHandler()

    fh.setLevel(level)
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
