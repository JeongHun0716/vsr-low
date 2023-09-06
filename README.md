# Visual-Speech-Recognition-for-Low-Resource-Languages
Visual Speech Recognition For Low-Resource Languages with Automatic Labels


## Models

In the following table, we provide all end-to-end trained models mentioned in our paper:

<details open>

<summary>mTEDx It</summary>

| Model         | Training Datasets  | Training data (h)  |  WER [%]   |    Target Languages     |
|--------------|:----------|:------------------:|:----------:|:------------------------:|
| best_ckpt.pt |       MT<sub>It        |        46           |    96.6    | It  |
| best_ckpt.pt |        MT<sub>It<sub> + VC<sub>It           |        84          |    36.7    | It  |
| best_ckpt.pt |        MT<sub>It + VC<sub>It + AV<sub>It       |        152         |    25.0    | It  |
