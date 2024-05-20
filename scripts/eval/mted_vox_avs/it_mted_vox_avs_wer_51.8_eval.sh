#!/bin/bash
ROOT=PATO_TO_ROOT # ex : /home/jh/projects/vsr-low
CUDA_VISIBLE_DEVICES=0 python test.py \
data.modality=video \
ckpt_path=${ROOT}/pretrained_models/it/mted_vox_avs/it_mted_vox_avs_wer_51.8.pth \
trainer.num_nodes=1 \
sp_model_path=${ROOT}/spm/mted_vox_avs/it/unigram1000.model \
dict_path=${ROOT}/spm/mted_vox_avs/it/unigram1000_units.txt \
data/dataset=mted_vox_avs/mted_vox_avs_it \
log_wandb=False logger=it \
intialize_dict=False