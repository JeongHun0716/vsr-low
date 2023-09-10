## Statistics of Dataset for Italian VSR

| Dataset        | split  | Duration (h)  |  Human-labeled # of Video      |   Human-labeled # of Video      |
|--------------|:----------|:------------------:|:-----------------:|:-----------------:|
| mTEDx |       train        |        46           |    26,108    | -    |
| mTEDx |       valid        |         0.35         |    252    |   -    | 
| mTEDx |       test        |          0.45       |    309    |  -    | 
| VoxCeleb2 |       train        |        38           |    -    |   19,261    | 
| AVSpeech |       train        |        68           |    -    |  38,227    | 


## Structure of txt files of Multilingual TEDx (mTEDx) dataset
The original videos downloaded from the mTEDx website are provided as a form of not-trimmed things. Therefore, you should trim the videos to contain only the speaker. For this purpose, Pingchuan Ma manually cleaned the French corpus and provided the file of video lists. Additionally, we use only under 20 seconds of videos in these provided files to train the VSR model.

The format of *.txt file is as follows:

line i : Video name &emsp; start_sec &emsp;  enc_sec &emsp;  transcription

(e.g., line 1 : 0u7tTptBo9I_0004&emsp;	42.91&emsp;	45.26&emsp;	et pourtant on vient tous de locéan)


## Structure of txt files of VoxCeleb2 dataset
The original videos downloaded from the VoxCeleb2 are provided as a form of already trimmed. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name &emsp; transcription

(e.g., line 4 : test/mp4/id03030/kkzpqpAxGJ4/00228.mp4&emsp;	il faut passer à autre chose donc les études sont là pour nous assurer d'un)


## Structure of txt files of Multilingual TEDx (mTEDx) dataset
You can download the trimmed video from the AVSpeech CSV files provided by the official website. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name_start sec_end sec &emsp;  transcription

(e.g., line 2 : enSzPLXxw8I_115.715600_119.986533.mp4 &emsp;	le rat nommera les tortues raphaël donatello leonardo et michelangelo)
