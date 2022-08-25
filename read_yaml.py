from addict import Dict
import numpy as np
import yaml


def read_yaml(fpath='./model.yaml'):
    with open(fpath, mode='r') as file:
        yml = yaml.load(file, Loader=yaml.Loader)
#        print('Transform:', Dict(yml).Transform)
        return Dict(yml)


if __name__ == '__main__':
    d = read_yaml()
    print(d.Augmentation)

