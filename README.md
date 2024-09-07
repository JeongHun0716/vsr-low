# Visual Speech Recognition for Languages with Limited Labeled Data Using Automatic Labels from Whisper

This repository contains the Official PyTorch implementation code of the following paper:

> **Visual Speech Recognition for Languages with Limited Labeled Data Using Automatic Labels from Whisper**<br>  
> \*Jeong Hun Yeo, \*Minsu Kim, Shinji Watanabe, and Yong Man Ro<br> 
> [[Paper]](https://ieeexplore.ieee.org/abstract/document/10446720) 

<div align="center"><img width="80%" src="img/img.PNG?raw=true" /></div>


We release the automatic labels of the four low-resource languages([French](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/French), [Italian](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/Italian), [Portuguese](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/Portuguese), and [Spanish](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/Spanish)). 

To generate the automatic labels, we identify the languages of all videos in VoxCeleb2 and AVSpeech, and then the transcription (automatic labels) is produced by the pretrained ASR model. In this project, we use a "[whisper/large-v2](https://github.com/openai/whisper)" model to conduct these processes.  

## Environment Setup
```bash
conda create -n vsr-low python=3.9 -y
conda activate vsr-low
git clone https://github.com/JeongHun0716/vsr-low
cd vsr-low
```

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
pip install hydra-core==1.3.0
pip install omegaconf==2.3.0
pip install pytorch-lightning==1.5.10
pip install sentencepiece
pip install av
```


## Dataset preparation
Multilingual TEDx(mTEDx), VoxCeleb2, and AVSpeech Datasets. 
  1. Download the mTEDx dataset from the [mTEDx link](https://www.openslr.org/100) of the official website.
  2. Download the VoxCeleb2 dataset from the [VoxCeleb2 link](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) of the official website.
  3. Download the AVSpeech dataset from the [AVSpeech link](https://looking-to-listen.github.io/avspeech/) of the official website.

When you are interested in training the model for a specific target language VSR, we recommend using language-detected files (e.g., [link](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/blob/main/French/AVSpeech-Fr/train.txt) provided in this project instead of video lists of the AVSpeech dataset provided on the official website to reduce the dataset preparation time. Because of the huge amount of AVSpeech dataset, it takes a lot of time.

## Preprocessing 
After downloading the datasets, you should detect the facial landmarks of all videos and crop the mouth region using these facial landmarks. We recommend you preprocess the videos following [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages).  

  
## Training the Model
The training code is available soon.


## Inference
Download the checkpoints from the below links and move them to the `pretrained_models` directory. You can evaluate the performance of each model using the scripts available in the `scripts` directory.

## Pretrained Models

<details open>

<summary>mTEDx Fr</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/oiptq2pxwv386v80ym4kk/fr_mted_wer_65.3.pth?rlkey=b6by808fog6xw1ofkvmpvj68s&st=6k5gmhsz&dl=0) |       mTEDx        |        85           |    65.25    | Fr  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/pjhxyredi12bmz12ea8jl/fr_mted_vox_wer_60.6.pth?rlkey=nvsew9e3zc3vxdydk0nmund7k&st=sqkw3rl9&dl=0) |        mTEDx + VoxCeleb2            |        209          |    60.61    | Fr  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/6pzmsmmvx2fjrjlvx6gkr/fr_mted_vox_avs_wer_58.3.pth?rlkey=sfqbsxcrfplzsumroyarw0e43&st=rv4ull1x&dl=0) |        mTEDx + VoxCeleb2 + AVSpeech       |        331         |    58.30    | Fr  |



<details open>

<summary>mTEDx It</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/fk5pvmyfo2fek19ama9iw/it_mted_wer_60.4.pth?rlkey=76bb83e0p42o5v7bij3sddwmf&st=elxnmxz0&dl=0) |       mTEDx        |        46           |    60.40    | It  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/1x257dhxhdti7esrm29p1/it_mted_vox_wer_56.5.pth?rlkey=r1l0h6cw10gg8i5e82bqcnok9&st=2lc9areg&dl=0) |        mTEDx + VoxCeleb2            |        84          |    56.48    | It  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/609u2o3ulc35ziceoxpqs/it_mted_vox_avs_wer_51.8.pth?rlkey=hqmczohabyo2ixij92w2podj0&st=scisjf0t&dl=0) |        mTEDx + VoxCeleb2 + AVSpeech       |        152         |    51.79    | It  |

<details open>

<summary>mTEDx Es</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/sb3p26t21fi7h38u1uxrk/es_mted_wer_59.9.pth?rlkey=rd4hfffu6gqg3oiswvqu8585k&st=zkpmpn1j&dl=0) |       mTEDx        |        72           |    59.91    | Es  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/c9avezj9a2k1usdeb3xta/es_mted_vox_wer_54.1.pth?rlkey=agra6ao6563pyabwe24888geg&st=5eb0eug8&dl=0) |        mTEDx + VoxCeleb2            |        114          |    54.05    | Es  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/tzddk1nl8yylirq4ja79j/es_mted_vox_avs_wer_45.7.pth?rlkey=ldml1kaexags7zm2jplw7j4jk&st=d7tyezgf&dl=0) |        mTEDx + VoxCeleb2 + AVSpeech       |        384         |    45.71    | Es  |



<details open>

<summary>mTEDx Pt</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/kzvzk99061b4yt0ccmuzd/pt_mted_wer_59.5.pth?rlkey=ncm145vzy2vp02eaoij1wncy5&st=40qouw20&dl=0) |       mTEDx        |        82           |    59.45    | Pt  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/s9n4aba92avvf6scdsl7t/pt_mted_vox_wer_58.8.pth?rlkey=q777mbgaphzm2vkr9wxij3hhf&st=2abb5z28&dl=0) |        mTEDx + VoxCeleb2            |        91          |    58.82    | Pt  |
| [ckpt.pt](https://www.dropbox.com/scl/fi/c8wyurx379zhbebwxn763/pt_mted_vox_avs_wer_47.9.pth?rlkey=2h3o4xcazmx4si2347gbrxwkv&st=r8h5g2wf&dl=0) |        mTEDx + VoxCeleb2 + AVSpeech       |        420         |    47.89    | Pt  |


## Citation
If you find this work useful in your research, please cite the paper:

```bibtex
@inproceedings{yeo2024visual,
  title={Visual Speech Recognition for Languages with Limited Labeled Data Using Automatic Labels from Whisper},
  author={Yeo, Jeong Hun and Kim, Minsu and Watanabe, Shinji and Ro, Yong Man},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10471--10475},
  year={2024},
  organization={IEEE}
}
```


## Acknowledgement

This project is based on the auto-avsr code. We would like to acknowledge and thank the original developers of auto-avsr for their contributions and the open-source community for making this work possible.

auto-avsr Repository: [auto-avsr GitHub Repository](https://github.com/mpc001/auto_avsr)


