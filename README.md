# Visual-Speech-Recognition-for-Low-Resource-Languages
Visual Speech Recognition For Low-Resource Languages with Automatic Labels


## Models

In the following table, we provide all end-to-end trained models mentioned in our paper:


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
