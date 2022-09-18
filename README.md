# Emotion-Recognition
This repository contains a pytorch implementation of an emotion classification algorithm, which is basically an image classification task. Following is the result of training efficient-net-b0 on Microsoft FER(Facial Expression Recognition) Dataset.

<p align="center">
    <img src="/output/result.gif" width="300px" height="300px"/>
</p>

## Requirements
download requirements using conda

## Usage
### Dataset
Download this dataset if you want to train your model from scratch
- [FER](https://www.kaggle.com/datasets/msambare/fer2013) 

## Try it yourself
make sure you have git and conda installed on your system, then follow this steps
```
git clone https://github.com/yMayanand/Emotion-Recognition.git
cd Emotion-Recognition
conda env create -f envs/test_env.yml
conda activate test-emotion
python video_cam.py
```
you can also try my project directly from [here](https://huggingface.co/spaces/Mayanand/emotion-recognition) which is live on huggingface spaces.
