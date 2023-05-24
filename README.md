# Source Code 01


## Pre-requisites
1. Python >= 3.8
1. Install espeak
1. Clone this repository 
1. Install python requirements. Please refer [requirements.txt](requirements.txt)
1. Download the [pretrained model and codebooks of the test speakers](https://drive.google.com/drive/folders/1b9GeuJmVIdW9m-BzwnRqFWdt_YjJ0Ukx?usp=share_link)
<br><br>

## Inference Example
```
python inference.py -c configs/base.json -cp pretrained/G_00500000.pth -cb codebooks/VCTK/p245.npy -t "This work is awesome." -o synthesized_speech.wav 
```
