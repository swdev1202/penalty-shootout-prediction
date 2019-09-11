# Penalty Shootout Prediction

## Introduction
This is a CNN classifier to predict where the ball will go before the kicker's contact with the ball with a single image. It was hard to find high resolution images of real players. Therefore, I decided to collect and create my own dataset using FIFA 18.

## Quick Start
1. Download the custom dataset [HERE](https://drive.google.com/file/d/1l0aXfoB5dWNBcomUI-EitTZeS9tWMh-2/view?usp=sharing)
2. Place the dataset in `./data/`
3. Download the pre-trained model [HERE](https://drive.google.com/file/d/1XYG2KSVlcjiqz1O4kkVm36OTT_245HB0/view?usp=sharing)
4. Place the `.h5` model in `./model/`
5. Run `python inference.py` once you place images you want to test in `./data/test/`

For `video_inference`, place your .mp4 file in `./data/test/video/` and run `python video_inference.py`. Note the video must be in 1280x720 (Width x Height). 

Watch a working demo [here](https://www.youtube.com/watch?v=AHos7RDP9cw)

## Additional Documents
[It](https://drive.google.com/file/d/1CiOOZzIrlC_yKVJcFOp7ROhR9vwisC0T/view?usp=sharing) has more details on the project.

This was my personal project for CMPE257 (Machine Learning) @ San Jose State University in Spring 2019.
