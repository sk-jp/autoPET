import csv
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from monai.data import (
#    DataLoader,
    Dataset,
    CacheDataset,
    PersistentDataset,
    load_decathlon_datalist,
#    decollate_batch,
    list_data_collate,
)

from get_transform import get_transform


class VolumeDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(VolumeDataModule, self).__init__()

        # configs
        self.cfg_dataset = cfg.Data.dataset
        self.cfg_dataloader = cfg.Data.dataloader
        self.cfg_transform = cfg.Transform
        
        self.top_dir = self.cfg_dataset.top_dir
        self.csv_path = self.cfg_dataset.csv_path_predict

    # called once from main process
    # called only within a single process
    def prepare_data(self):
        # prepare data
        pass
    
    # perform on every GPU
    def setup(self, stage):
        # read a csv file (pos/neg)
        test_datalist = []
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                d = {}
                for idx, c in enumerate(row):
                    d[header[idx]] = f'{self.top_dir}/{c}'
                test_datalist.append(d)
                       
        # transforms
        """
        train_transforms = get_transform(self.cfg_transform.train)
        valid_transforms = get_transform(self.cfg_transform.valid)
        """
        test_transforms = get_transform(self.cfg_transform.test)

        # dataset
        self.dataset = {}
        """
        self.dataset['train'] = PersistentDataset(
            data=self.train_datalist,
            transform=train_transforms,
            cache_dir=self.cfg_dataset.cache_dir
        )
        self.dataset['valid'] = PersistentDataset(
            data=self.valid_datalist,
            transform=valid_transforms,
            cache_dir=self.cfg_dataset.cache_dir
        )
        """
        self.dataset['test'] = Dataset(
            data=test_datalist,
            transform=test_transforms,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.dataset['train'],
            batch_size=self.cfg_dataloader.batch_size,
            shuffle=self.cfg_dataloader.train.shuffle,
            num_workers=self.cfg_dataloader.num_workers,
            pin_memory=False,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset['valid'],
            batch_size=self.cfg_dataloader.batch_size,
            shuffle=self.cfg_dataloader.valid.shuffle,
            num_workers=self.cfg_dataloader.num_workers,
            pin_memory=False
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.dataset['test'],
            batch_size=self.cfg_dataloader.batch_size,
            shuffle=self.cfg_dataloader.valid.shuffle,
            num_workers=self.cfg_dataloader.num_workers,
            pin_memory=False
        )
        return test_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.dataset['test'],
            batch_size=self.cfg_dataloader.batch_size,
            shuffle=self.cfg_dataloader.valid.shuffle,
            num_workers=self.cfg_dataloader.num_workers,
            pin_memory=False
        )
        return predict_loader
