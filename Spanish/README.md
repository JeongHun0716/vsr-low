## Statistics of Dataset for Spanish VSR

| Dataset        | split  | Duration (h)  |  Human-labeled # of Video      |   Human-labeled # of Video      |
|--------------|:----------|:------------------:|:-----------------:|:-----------------:|
| mTEDx |       train        |        72           |    44,532    | -    |
| mTEDx |       valid        |         0.65         |    403    |   -    | 
| mTEDx |       test        |          0.45       |    302    |  -    | 
| VoxCeleb2 |       train        |        42           |    -    |   22,682    | 
| AVSpeech |       train        |        270           |    -    |  151,173    | 


## Structure of txt files of Multilingual TEDx (mTEDx) dataset
The original videos downloaded from the mTEDx website are provided as a form of not-trimmed things. Therefore, you should trim the videos to contain only the speaker. For this purpose, Pingchuan Ma manually cleaned the French corpus and provided the file of video lists. Additionally, we use only under 20 seconds of videos in these provided files to train the VSR model.

The format of *.txt file is as follows:

line i : Video name &emsp; start_sec &emsp;  enc_sec &emsp;  transcription

(e.g., line 2 : ninrJokK-oA_0247.mp4&emsp;	685.96&emsp;	687.01&emsp;	dejad que os lo muestre


## Structure of txt files of VoxCeleb2 dataset
The original videos downloaded from the VoxCeleb2 are provided as a form of already trimmed. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name &emsp; transcription

(e.g., line 1 : test/mp4/id03127/TtFfWu9vKp8/00200.mp4&emsp;	muy concentrada muy conpenetrada en mi trabajo)


## Structure of txt files of AVSpeech dataset
You can download the trimmed video from the AVSpeech CSV files provided by the official website. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name_start sec_end sec &emsp;  transcription

(e.g., line 2 : 5kFce3iHPes_196.432567_199.432567.mp4&emsp;	use live es una herramienta que ya habíamos visto en otros teléfonos)
