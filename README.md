## Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models (DG) (under review) <br><sub>Official PyTorch implementation of the Discriminator Guidance </sub>
**[Dongjun Kim](https://sites.google.com/view/dongjun-kim) \*, [Yeongmin Kim](https://sites.google.com/view/yeongmin-space/%ED%99%88) \*, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon**   
<sup> * Equal contribution </sup> <br>

| [paper](https://arxiv.org/abs/2211.17091) |  <br>
**Camera-ready final version will be released within this month. Stay tuned!** <br>
**See [https://github.com/alsdudrla10/DG_imagenet](https://github.com/alsdudrla10/DG_imagenet) for the ImageNet256 code release.** <br>

## Overview
![Teaser image](./figures/Figure1_v2.PNG)

## Step-by-Step running of Discriminator Guidance

### 1) Prepare pre-trained score network
  - Download at [EDM](https://github.com/NVlabs/edm).
  - Save
  


  ${project_page}/DG/
  ├── checkpoints
  │   ├── pretrained_score/edm-cifar10-32x32-uncond-vp.pkl
  │   ├── discriminator
  │   ├── ADM_classifier
  ├── ...

  - save_directory: DG/checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl

### 2) Fake sample generation
  - command: python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl --outdir=samples/cifar_uncond_vanilla --dg_weight_1st_order=0

### 3) Prepare real data
  - download [here](https://drive.google.com/drive/folders/1lOwHMS1GRuIfJ9ix9A6vtOm7vX8EN87Y)
  - save_directory: DG/data/true_data.npz

### 4) Prepare pre-trained classifier
  - download [here](https://drive.google.com/drive/folders/1lOwHMS1GRuIfJ9ix9A6vtOm7vX8EN87Y)
  - save_directory: DG/checkpoints/ADM_classifier/32x32_classifier.pt
  - We train 32 resolution classifier from [here](https://github.com/openai/guided-diffusion)

### 5) Discriminator training
  - command: python3 train.py
  - downalod checkpoint [here](https://drive.google.com/drive/folders/1lOwHMS1GRuIfJ9ix9A6vtOm7vX8EN87Y)

### 6) Generation with Discriminator Guidance
  - command: python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl --outdir=samples/cifar_uncond
  
### 7) FID evaluation
  - command: python3 fid_npzs.py --ref=/stats/cifar10-32x32.npz --num_samples=50000 --images=/samples/cifar_uncond/
  - download stat files [here](https://drive.google.com/drive/folders/1lOwHMS1GRuIfJ9ix9A6vtOm7vX8EN87Y)

## Results on data diffusion
|FID-50k |Cifar-10|FFHQ64|CelebA64|
|------------|------------|------------|------------|
|Privious SOTA|2.03|2.39|1.90|4.59|
|+ DG|1.77|1.98|1.34|3.17|

## Results on latent diffusion
|FID-50k |Cifar-10|
|------------|------------|
|Privious SOTA|2.10|
|+ DG|1.94|


## Samples from Cifar-10
![Teaser image](./figures/Figure3.PNG)



