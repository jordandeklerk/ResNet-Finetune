# Fine-Tuning ResNet50 

<hr>

## Contents

1. [Highlights](#Highlights)
2. [Requirements](#Requirements)
3. [Usage](#Usage)
4. [Results](#Results)


<hr>

## Highlights
This project is an implementation of fine-tuning ResNet50 for the Stanford cars dataset. We follow the traditional method of fine-tuning foundation models: we first freeze the layers of the pre-trained ResNet and train only the classification head of the network. Then we un-freeze the pre-trained layers and fine-tune the full model. 

<hr>

## Requirements
```shell
pip install -r requirements.txt
```

<hr>

## Usage
To replicate the reported results, we have included the Jupyter Notebook along with the main Python files. It is best to open this notebook on Kaggle and import the Stanford cars dataset from there as the torchvision import does not work anymore. 

<hr>

## Results
We tested our fine-tuned ResNet50 model on the Stanford cars dataset
  * Stanford Cars
    * ```ResNet50-Finetune``` - 84.8% accuracy on the test data

Model Summary:
```
=============================================================================================
Layer (type:depth-idx)                        Kernel Shape     Output Shape     Param #
=============================================================================================
StanfordCarsNet                               --               [1, 196]         --
├─Sequential: 1-1                             --               [1, 2048, 1, 1]  --
│    └─Conv2d: 2-1                            [7, 7]           [1, 64, 112, 112] 9,408
│    └─BatchNorm2d: 2-2                       --               [1, 64, 112, 112] 128
│    └─ReLU: 2-3                              --               [1, 64, 112, 112] --
│    └─MaxPool2d: 2-4                         3                [1, 64, 56, 56]  --
│    └─Sequential: 2-5                        --               [1, 256, 56, 56] --
│    │    └─Bottleneck: 3-1                   --               [1, 256, 56, 56] 75,008
│    │    └─Bottleneck: 3-2                   --               [1, 256, 56, 56] 70,400
│    │    └─Bottleneck: 3-3                   --               [1, 256, 56, 56] 70,400
│    └─Sequential: 2-6                        --               [1, 512, 28, 28] --
│    │    └─Bottleneck: 3-4                   --               [1, 512, 28, 28] 379,392
│    │    └─Bottleneck: 3-5                   --               [1, 512, 28, 28] 280,064
│    │    └─Bottleneck: 3-6                   --               [1, 512, 28, 28] 280,064
│    │    └─Bottleneck: 3-7                   --               [1, 512, 28, 28] 280,064
│    └─Sequential: 2-7                        --               [1, 1024, 14, 14] --
│    │    └─Bottleneck: 3-8                   --               [1, 1024, 14, 14] 1,512,448
│    │    └─Bottleneck: 3-9                   --               [1, 1024, 14, 14] 1,117,184
│    │    └─Bottleneck: 3-10                  --               [1, 1024, 14, 14] 1,117,184
│    │    └─Bottleneck: 3-11                  --               [1, 1024, 14, 14] 1,117,184
│    │    └─Bottleneck: 3-12                  --               [1, 1024, 14, 14] 1,117,184
│    │    └─Bottleneck: 3-13                  --               [1, 1024, 14, 14] 1,117,184
...
Input size (MB): 0.60
Forward/backward pass size (MB): 177.83
Params size (MB): 95.64
Estimated Total Size (MB): 274.07
=============================================================================================
```
