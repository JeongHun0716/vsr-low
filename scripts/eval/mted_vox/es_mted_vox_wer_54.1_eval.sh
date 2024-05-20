#!/bin/bash
ROOT=PATO_TO_ROOT # ex : /home/jh/projects/vsr-low
CUDA_VISIBLE_DEVICES=0 python test.py \
data.modality=video \
ckpt_path=${ROOT}/pretrained_models/es/mted_vox/es_mted_vox_wer_54.1.pth \
trainer.num_nodes=1 \
sp_model_path=${ROOT}/spm/mted_vox/es/unigram1000.model \
dict_path=${ROOT}/spm/mted_vox/es/unigram1000_units.txt \
data/dataset=mted_vox/mted_vox_es \
log_wandb=False logger=es \
intialize_dict=False