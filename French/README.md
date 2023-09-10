## Statistics of Dataset for French VSR

| Dataset        | split  | Duration (h)  |  Human-labeled # of Video      |   Human-labeled # of Video      |
|--------------|:----------|:------------------:|:-----------------:|:-----------------:|
| mTEDx |       train        |        85           |    58,426    | -    |
| mTEDx |       valid        |         0.4         |    235    |   -    | 
| mTEDx |       test        |          0.3       |    333    |  -    | 
| VoxCeleb2 |       train        |        124           |    -    |   66,943    | 
| AVSpeech |       train        |        122           |    -    |  69,020    | 


## Structure of txt files of Multilingual TEDx (mTEDx) dataset
The original videos downloaded from the mTEDx website are provided as a form of not-trimmed things. Therefore, you should trim the videos to contain only the speaker. For this purpose, Pingchuan Ma manually cleaned the French corpus and provided the file of video lists. Additionally, we use only under 20 seconds of videos in these provided files to train the VSR model.
