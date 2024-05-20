#!/bin/bash
ROOT=PATO_TO_ROOT # ex : /home/jh/projects/vsr-low
CUDA_VISIBLE_DEVICES=0 python test.py \
data.modality=video \
ckpt_path=${ROOT}/pretrained_models/fr/mted/fr_mted_wer_65.3.pth \
trainer.num_nodes=1 \
sp_model_path=${ROOT}/spm/mted/fr/unigram1000.model \
dict_path=${ROOT}/spm/mted/fr/unigram1000_units.txt \
data/dataset=mted/mted_fr \
log_wandb=False logger=fr \
intialize_dict=False