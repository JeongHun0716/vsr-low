## Statistics of Dataset for Portuguese VSR

| Dataset        | split  | Duration (h)  |  Human-labeled # of Video      |   Human-labeled # of Video      |
|--------------|:----------|:------------------:|:-----------------:|:-----------------:|
| mTEDx |       train        |        82           |    52,058    | -    |
| mTEDx |       valid        |         0.7         |    532    |   -    | 
| mTEDx |       test        |          0.56       |    401    |  -    | 
| VoxCeleb2 |       train        |        9           |    -    |   4,843    | 
| AVSpeech |       train        |        329           |    -    |  176,601    | 


## Structure of txt files of Multilingual TEDx (mTEDx) dataset
The original videos downloaded from the mTEDx website are provided as a form of not-trimmed things. Therefore, you should trim the videos to contain only the speaker. For this purpose, Pingchuan Ma manually cleaned the French corpus and provided the file of video lists. Additionally, we use only under 20 seconds of videos in these provided files to train the VSR model.

The format of *.txt file is as follows:

line i : Video name &emsp; start_sec &emsp;  enc_sec &emsp;  transcription

(e.g., line 2 : m_7lX1_mg4Y_0100.mp4&emsp;	470.26&emsp;	473.77&emsp;	eu acredito muito em palavras de empoderamento)


## Structure of txt files of VoxCeleb2 dataset
The original videos downloaded from the VoxCeleb2 are provided as a form of already trimmed. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name &emsp; transcription

(e.g., line 8 : test/mp4/id03127/4lq9ubDWxEw/00034.mp4&emsp;	ter tido um anjinho eu acho que dá uma tipo assim)


## Structure of txt files of AVSpeech dataset
You can download the trimmed video from the AVSpeech CSV files provided by the official website. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name_start sec_end sec &emsp;  transcription

(e.g., line 3 : jo6klF-Ptu4_210.009800_214.948067.mp4&emsp;	uma caçada digamos uma caçada aos terroristas)
