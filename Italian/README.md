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

(e.g., line 3 : htIYaFDjIJI_0024.mp4&emsp;	201.16&emsp;	202.99&emsp;	il pattern è sempre lo stesso)


## Structure of txt files of VoxCeleb2 dataset
The original videos downloaded from the VoxCeleb2 are provided as a form of already trimmed. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name &emsp; transcription

(e.g., line 3 : dev/mp4/id08977/3ap0QSyN6Vc/00027.mp4&emsp;	gli era detto nel momento in cui gli annunciava che sarebbe andato a morire)


## Structure of txt files of Multilingual TEDx (mTEDx) dataset
You can download the trimmed video from the AVSpeech CSV files provided by the official website. So, you can use the trimmed videos to train the VSR model using only the provided automatic labels. 

The format of *.txt file is as follows:

line i : Video name_start sec_end sec &emsp;  transcription

(e.g., line 7 : mziTtu17RK8_206.200000_210.000000.mp4&emsp;	soprattutto in un ambito di pilotaggio di un aeromobile)
