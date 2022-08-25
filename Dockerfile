#FROM python:3.9-slim
FROM pytorch/pytorch


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output/images/automated-petct-lesion-segmentation \
    && chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/

RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm fix_model_state_dict.py /opt/algorithm/
COPY --chown=algorithm:algorithm lightning_module.py /opt/algorithm/
COPY --chown=algorithm:algorithm volume_datamodule.py /opt/algorithm/
COPY --chown=algorithm:algorithm get_kernels_strides.py /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm post_process.py /opt/algorithm/
COPY --chown=algorithm:algorithm get_transform.py /opt/algorithm/
COPY --chown=algorithm:algorithm read_yaml.py /opt/algorithm/

COPY --chown=algorithm:algorithm dyn_unet.yaml /opt/algorithm/
COPY --chown=algorithm:algorithm datalist.csv /opt/algorithm/

# COPY --chown=algorithm:algorithm dyn_unet-epoch=273-valid_loss=0.33.ckpt /opt/algorithm/
# COPY --chown=algorithm:algorithm dyn_unet-epoch=253-valid_loss=0.16.ckpt /opt/algorithm/
COPY --chown=algorithm:algorithm dyn_unet-epoch223-valid_loss0.45-CV0.ckpt /opt/algorithm/
COPY --chown=algorithm:algorithm dyn_unet-epoch251-valid_loss0.47-CV1.ckpt /opt/algorithm/
COPY --chown=algorithm:algorithm dyn_unet-epoch102-valid_loss0.48-CV2.ckpt /opt/algorithm/
COPY --chown=algorithm:algorithm dyn_unet-epoch248-valid_loss0.51-CV3.ckpt /opt/algorithm/
COPY --chown=algorithm:algorithm dyn_unet-epoch266-valid_loss0.48-CV4.ckpt /opt/algorithm/

ENTRYPOINT python -m process $0 $@
