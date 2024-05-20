#!/bin/bash
ROOT=PATO_TO_ROOT # ex : /home/jh/projects/vsr-low
CUDA_VISIBLE_DEVICES=0 python test.py \
data.modality=video \
ckpt_path=${ROOT}/pretrained_models/es/mted/es_mted_wer_59.9.pth \
trainer.num_nodes=1 \
sp_model_path=${ROOT}/spm/mted/es/unigram1000.model \
dict_path=${ROOT}/spm/mted/es/unigram1000_units.txt \
data/dataset=mted/mted_es \
log_wandb=False logger=es \
intialize_dict=False