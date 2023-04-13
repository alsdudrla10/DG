## Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models (DG) (under review) <br><sub>Official PyTorch implementation of the Discriminator Guidance </sub>
**[Dongjun Kim](https://sites.google.com/view/dongjun-kim) \*, [Yeongmin Kim](https://sites.google.com/view/yeongmin-space/%ED%99%88) \*, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon**   
<sup> * Equal contribution </sup> <br>

| [paper](https://arxiv.org/abs/2211.17091) |  <br>
**Camera-ready final version will be released within this month. Stay tuned!** <br>
**See [https://github.com/alsdudrla10/DG_imagenet](https://github.com/alsdudrla10/DG_imagenet) for the ImageNet256 code release.** <br>

## Overview
![Teaser image](./figures/Figure1_v2.PNG)

## Step-by-Step Running of Discriminator Guidance

### 1) Prepare pre-trained score network
  - Download **edm-cifar10-32x32-uncond-vp.pkl** at [EDM](https://github.com/NVlabs/edm).
  - Place **edm-cifar10-32x32-uncond-vp.pkl** at the directory specified.  
 
  ```
  ${project_page}/DG/
  ├── checkpoints
  │   ├── pretrained_score/edm-cifar10-32x32-uncond-vp.pkl
  ├── ...
  ```

### 2) Fake sample generation
  - Run: 
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl --outdir=samples/cifar_uncond_vanilla --dg_weight_1st_order=0
   ```

### 3) Prepare real data
  - Download [DG/data/true_data.npz](https://drive.google.com/drive/folders/18qh5QGP2gLgVjr0dh2g8dfBYZoGC0uVT)
  - Place **true_data.npz** at the directory specified.
  ```
  ${project_page}/DG/
  ├── data
  │   ├── true_data.npz
  ├── ...
  ```

### 4) Prepare pre-trained classifier
  - Download [DG/checkpoints/ADM_classifier/32x32_classifier.pt](https://drive.google.com/drive/folders/1gb68C13-QOt8yA6ZnnS6G5pVIlPO7j_y)
  - We train 32 resolution classifier from [ADM](https://github.com/openai/guided-diffusion).
  - Place **32x32_classifier.pt** at the directory specified.
  ```
  ${project_page}/DG/
  ├── checkpoints
  │   ├── ADM_classifier/32x32_classifier.pt
  ├── ...
  ```

### 5) Discriminator training
  - Download pre-trained checkpoint [DG/checkpoints/discriminator/cifar_uncond/discriminator_60.pt](https://drive.google.com/drive/folders/1Mf3F1yGfWT8bO0_iOBX-PWG3O-OLROE2) for the test.
  - Place **discriminator_60.pt** at the directory specified.
  ```
  ${project_page}/DG/
  ├── checkpoints
  │   ├── discriminator/cifar_uncond/discriminator_60.pt
  ├── ...
  ```
  - To train the discriminator from scratch, run:
   ```
   python3 train.py
   ```

### 6) Generation with Discriminator Guidance
  - Run: 
  ```
  python3 generate.py --network checkpoints/pretrained_score/edm-cifar10-32x32-uncond-vp.pkl --outdir=samples/cifar_uncond
   ```
  
### 7) FID evaluation
  - Download stat files at [DG/stats/cifar10-32x32.npz](https://drive.google.com/drive/folders/1xTdHz2fe71yvO2YpVfsY3sgH5Df7_b6y)
  - Place **cifar10-32x32.npz** at the directory specified.
  ```
  ${project_page}/DG/
  ├── stats
  │   ├── cifar10-32x32.npz
  ├── ...
  ```
  - Run: 
  ```
  python3 fid_npzs.py --ref=/stats/cifar10-32x32.npz --num_samples=50000 --images=/samples/cifar_uncond/
   ```

## Experimental Results
### EDM-G++
|FID-50k |Cifar-10|Cifar-10(conditional)|FFHQ64|
|------------|------------|------------|------------|
|EDM|2.03|1.82|2.39|
|EDM-G++|1.77|1.64|1.98|

### Other backbones
|FID-50k  |Cifar-10|CelebA64|
|------------|------------|------------|
|Backbone model|LSGM|ST|
|Backbone|2.10|1.90|
|G++|1.94|1.34|


### Samples from Unconditional Cifar-10
![Teaser image](./figures/Figure3.PNG)

### Samples from Conditional Cifar-10
![Teaser image](./figures/Figure3.PNG)



