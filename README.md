This is an official PyTorch implementation of NeurIPS 2025 paper "An Efficient Text-guided Cross-Modal Alignment Fusion Network for Referring Image Segmentation."
This is a raw version at the moment; a tweaked version will be released online after the paper is accepted!

#Specification of dependencies

## Environment
```bash
conda create -n TCAFN python=3.9.18 -y
conda activate TCAFN
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirement.txt
```

## Datasets
The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)

Both the datasets and pretrained weights are publicly available and do not contain any information that could reveal the authors' identities.

## Pretrained weights
Download the pretrained weights of DiNOv2-B, DiNOv2-L and ViT-B to pretrain
```bash
mkdir pretrain && cd pretrain
## DiNOv2-B
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
## DiNOv2-L
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
## ViT-B
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

# Quick Start

To train TCAFN, modify the script according to your requirements and run it:

```
bash run_scripts/train.sh
```

To evaluate TCAFN, specify the model file path in test.sh according to your requirements and run the script:

```
bash run_scripts/test.sh
```


## Weights

After acceptance of our paper, we will open-source our model weights.


## Results
The mIoU result is as follows:
| Method                       | RefCOCO (val) | RefCOCO (testA) | RefCOCO (testB) | RefCOCO+ (val) | RefCOCO+ (testA) | RefCOCO+ (testB) | G-Ref (val(u)) | G-Ref (test(u)) | G-Ref (val(g)) |
|------------------------------|---------------|------------------|-----------------|----------------|-------------------|------------------|----------------|------------------|----------------|
| TCAFN-B (Ours)             | 76.11          | 78.31            | 73.04           | 68.36           | 73.56             | 60.91            | 67.91          | 68.12            | 66.64           |
| TCAFN-L (Ours)             | 77.42      | 79.08        | 74.63       | 69.11       | 74.23         | 63.96        | 69.17      | 70.13        | 67.9       | 68.51 |



The oIoU result is as follows:
| Method           | RefCOCO (val) | RefCOCO (testA) | RefCOCO (testB) | RefCOCO+ (val) | RefCOCO+ (testA) | RefCOCO+ (testB) | G-Ref (val(u)) | G-Ref (test(u)) | G-Ref (val(g)) |
|------------------|---------------|-----------------|-----------------|----------------|------------------|------------------|----------------|-----------------|----------------|
| TCAFN-B (Ours)  | 74.54          | 77.53            | 71.28            | 66.03           | 71.92             | 57.25             | 65.94           | 66.15            | 64.03           | 
| TCAFN-L (Ours)  | 75.76          | 78.41            | 72.94            | 67.84           | 73.13             | 60.42             | 67.35           | 68.34            | 66.41           |
