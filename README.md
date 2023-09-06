# Visual-Speech-Recognition-for-Low-Resource-Languages
Visual Speech Recognition For Low-Resource Languages with Automatic Labels


## Models

In the following table, we provide all end-to-end trained models mentioned in our paper:

<details open>

<summary>mTEDx It</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| best_ckpt.pt |       mTEDx        |        46           |    96.6    | It  |
| best_ckpt.pt |        mTEDx + VoxCeleb2            |        84          |    36.7    | It  |
| best_ckpt.pt |        mTEDx + VoxCeleb2 + AVSpeech       |        152         |    25.0    | It  |
