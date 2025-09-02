# IDD Semantic Segmentation with SegFormer

## Overview
This repository contains code for training and inference of semantic segmentation on the IDD (India Driving Dataset) using SegFormer.

## Setup

1. Clone this repository:
```bash
git clone <repo_url>
cd <repo_name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

1. Download the IDD dataset: [IDD Dataset](https://idd.insaan.iiit.ac.in/)
2. Organize the dataset as:

```
data/
├── leftImg8bit/
│   ├── train/
│   └── val/
├── gtFine_masks/
│   ├── train/
│   └── val/
```

3. You can also use the included `sample_images` and `sample_masks` for quick testing.

## Training

```bash
python src/train.py \
    --data_root data/ \
    --epochs 12 \
    --batch_size 4 \
    --save_dir checkpoints/
```

Training uses SegFormer with a pretrained ADE20K backbone. Checkpoints will be saved in `checkpoints/`.

## Inference

```bash
python src/inference.py \
    --image_path data/sample_images/city1_sample.png \
    --model_path checkpoints/best_miou_epochX.pt
```

Generates segmentation overlay on the input image. Supports saving output masks and overlay images.

## Utilities

`src/utils.py` contains:
- Mask remapping from original IDD IDs to contiguous labels
- mIoU calculation
- Visualization helper functions