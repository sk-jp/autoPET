import monai.transforms as mt


def get_transform(conf_augmentation):
    """ Get augmentation function
        Args:
            conf_augmentation (Dict): dictionary of augmentation parameters
    """
    def get_object(trans):
#        print('trans:', trans)
        if trans.name in {'Compose', 'OneOf'}:
            augs_tmp = [get_object(aug) for aug in trans.member]
            return getattr(mt, trans.name)(augs_tmp, **trans.params)

        if hasattr(mt, trans.name):
            return getattr(mt, trans.name)(**trans.params)
        else:
            return eval(trans.name)(**trans.params)

    if conf_augmentation is None:
        augs = list()
    else:
        augs = [get_object(aug) for aug in conf_augmentation]

#    print('augs:', augs)

    return mt.Compose(augs)

if __name__ == '__main__':
    from read_yaml import read_yaml
    config = './dyn_unet.yaml'    
    cfg = read_yaml(fpath=config)
    trans = get_transform(cfg.Transform.train)
    