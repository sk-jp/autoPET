import argparse
import os
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
import SimpleITK
import warnings

from get_transform import get_transform
from lightning_module import LightningModule
from volume_datamodule import VolumeDataModule
from read_yaml import read_yaml


warnings.filterwarnings("ignore")


def make_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--config", default="dyn_unet.yaml", type=str, help="Path to config file")
    arg("--gpus", default="0", type=str, help="GPU IDs")
    return parser


def predict(cfg):
    """ Training main function
    """
    
    # == initial settings ==
    # random seed
    if isinstance(cfg.General.seed, int):
        seed_everything(seed=cfg.General.seed, workers=True)

    # == plugins ==
    plugins = None
    if not cfg.General.lr_tune:
        if cfg.General.strategy == "ddp":
            strategy = DDPStrategy(find_unused_parameters=False)

    # == Trainer ==
    default_root_dir = os.getcwd()
    trainer = Trainer(
#        max_epochs=cfg.General.epoch,
        strategy=strategy,
#        num_nodes=cfg.General.num_nodes,
        accelerator='gpu',
        devices=len(cfg.General.gpus),
        gpus=cfg.General.gpus,
        precision=cfg.General.precision,
        deterministic=False,
        benchmark=False,
        default_root_dir=default_root_dir,
        auto_select_gpus=False,
        callbacks=[],
        logger=[],
        replace_sampler_ddp=True,
        plugins=plugins,
#        num_sanity_val_steps=0,
#        accumulate_grad_batches=cfg.Optimizer.accumulate_grad_batches
    )

    # Lightning module and data module
    if cfg.Model.arch == 'dyn_unet':
        # transforms
        test_transforms = get_transform(cfg.Transform.test)

        # post transforms
        from monai.transforms import (
            Activationsd,
            AsDiscreted,
            Compose,
            Invertd,
#            SaveImaged,
            EnsureTyped,
        )
        post_test_transforms = Compose([
            EnsureTyped(keys="pred"),
#            Activationsd(keys="pred", sigmoid=True),
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="CTres",
                meta_keys="pred_meta_dict",
                orig_meta_keys="CTres_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True),
        ])                    
    
        model = LightningModule(cfg, test_transforms, post_test_transforms)
        datamodule = VolumeDataModule(cfg)
    else:
        raise ValueError(f'{cfg.Model.arch} is not supported.')
    
    # run prediction
    print('*** Start prediction ***')
    pred = trainer.predict(model, datamodule=datamodule)
                
    # inverse transform
#    post_transforms = datamodule.post_test_transforms
#    pred = post_transforms(pred)
#    print('pred(post):', pred.shape)

    return pred


def convert_mha_to_nii(mha_input_path, nii_out_path):
    print('  Reading:', mha_input_path)
    img = SimpleITK.ReadImage(mha_input_path, imageIO="MetaImageIO")
    SimpleITK.WriteImage(img, nii_out_path, True)

    
def convert_nii_to_mha(nii_input_path, mha_out_path):
    img = SimpleITK.ReadImage(nii_input_path)
    SimpleITK.WriteImage(img, mha_out_path, True)

        
def load_inputs(cfg):
    """
    Read from /input/
    Check https://grand-challenge.org/algorithms/interfaces/
    """
    ct_mha = os.listdir(os.path.join(cfg.input_path, 'images/ct/'))[0]
    pet_mha = os.listdir(os.path.join(cfg.input_path, 'images/pet/'))[0]
    uuid = os.path.splitext(ct_mha)[0]
    convert_mha_to_nii(os.path.join(cfg.input_path, 'images/pet/', pet_mha),
                       os.path.join(cfg.nii_path, 'SUV.nii.gz'))
    convert_mha_to_nii(os.path.join(cfg.input_path, 'images/ct/', ct_mha),
                       os.path.join(cfg.nii_path, 'CTres.nii.gz'))
  
    return uuid


def write_outputs(cfg, uuid):
    """
    Write to /output/
    Check https://grand-challenge.org/algorithms/interfaces/
    """
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    convert_nii_to_mha(os.path.join(cfg.output_path, "PRED.nii.gz"), os.path.join(cfg.output_path, uuid + ".mha"))
    print('Output written to: ' + os.path.join(cfg.output_path, uuid + ".mha"))


def main():
    # parse args
#    parser = make_parse()
#    args = parser.parse_args()
#    print('args:', args)

    # Read config
    cfg = read_yaml(fpath='./dyn_unet.yaml')
#    cfg = read_yaml(fpath=args.config)
#    if args.gpus is not None:
#        cfg.General.gpus = list(map(int, args.gpus.split(",")))

    # input and output path
#    cfg.input_path = '/input/'  # according to the specified grand-challenge interfaces
#    cfg.output_path = '/output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
#    cfg.nii_path = '/opt/algorithm/'  # where to store the nii files
#    cfg.ckpt_path = '/opt/algorithm/dyn_unet-epoch=253-valid_loss=0.16.ckpt'
#    cfg.input_path = './input/'  # according to the specified grand-challenge interfaces
#    cfg.output_path = './output/images/automated-petct-lesion-segmentation/'  # according to the specified grand-challenge interfaces
#    cfg.nii_path = './'  # where to store the nii files
#    cfg.ckpt_path = './dyn_unet-epoch=273-valid_loss=0.33.ckpt'
    if not os.path.exists(cfg.Path.output_path):
        os.makedirs(cfg.Path.output_path)
#    cfg.Model.pretrained = cfg.ckpt_path

    # load input data (convert mha to nii)
    print('Start processing')
    uuid = load_inputs(cfg.Path)

    cfg.Data.dataset.top_dir = cfg.Path.nii_path
    cfg.Data.dataset.csv_path_predict = f'{cfg.Path.nii_path}/datalist.csv'
            
    # Prediction
    print('Start prediction')
    predict(cfg)
#    pred = predict(cfg)
#    print(pred)

    # Write output
    write_outputs(cfg.Path, uuid)


if __name__ == '__main__':
    main()
