# Visual-Speech-Recognition-for-Low-Resource-Languages
Visual Speech Recognition For Low-Resource Languages with Automatic Labels From Whisper Model

This project is still under review process. 

After the review process, we will provide all VSR models, training code, and inference code for low-resource languages.

We release the automatic labels of the four low-resource languages([French](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/French), [Italian](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/Italian), [Portuguese](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/Portuguese), and [Spanish](https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages/tree/main/Spanish)). 

To generate the automatic labels, we identify the languages of all videos in VoxCeleb2 and AVSpeech, and then the transcription (automatic labels) is produced by the pretrained ASR model. In this project, we use a "[whisper/large-v2](https://github.com/openai/whisper)" model to conduct these processes.  

## Dataset Prepare
Multilingual TEDx Dataset(mTEDx)
  1. Download the mTEDx dataset from the [mTEDx link](https://www.openslr.org/100) of offical website.
  2.  

VoxCeleb2



AVSpeech

## Training the Model



## Inference

The code is avaiable soon.


## Models

<details open>

<summary>mTEDx Fr</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| best_ckpt.pt |       mTEDx        |        85           |    65.25    | Fr  |
| best_ckpt.pt |        mTEDx + VoxCeleb2            |        209          |    60.61    | Fr  |
| best_ckpt.pt |        mTEDx + VoxCeleb2 + AVSpeech       |        331         |    58.30    | Fr  |



<details open>

<summary>mTEDx It</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| best_ckpt.pt |       mTEDx        |        46           |    60.40    | It  |
| best_ckpt.pt |        mTEDx + VoxCeleb2            |        84          |    56.48    | It  |
| best_ckpt.pt |        mTEDx + VoxCeleb2 + AVSpeech       |        152         |    51.79    | It  |

<details open>

<summary>mTEDx Es</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| best_ckpt.pt |       mTEDx        |        72           |    59.91    | Es  |
| best_ckpt.pt |        mTEDx + VoxCeleb2            |        114          |    54.05    | Es  |
| best_ckpt.pt |        mTEDx + VoxCeleb2 + AVSpeech       |        384         |    45.71    | Es  |



<details open>

<summary>mTEDx Pt</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| best_ckpt.pt |       mTEDx        |        82           |    59.45    | Pt  |
| best_ckpt.pt |        mTEDx + VoxCeleb2            |        91          |    58.82    | Pt  |
| best_ckpt.pt |        mTEDx + VoxCeleb2 + AVSpeech       |        420         |    47.89    | Pt  |
