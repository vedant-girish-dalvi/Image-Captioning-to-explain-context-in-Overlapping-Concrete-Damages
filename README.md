# Image Captioning to Explain Context in Overlapping Concrete Damages

A deep learning project that generates **context-aware natural language descriptions** for images showing **overlapping concrete damages** (e.g., cracks, spalls, surface defects). This repository contains code for training, inference, and utilities to build an image captioning model tailored for civil infrastructure damage assessment.

> This project uses computer vision and natural language generation to provide meaningful captions that describe *damage types*, *locations*, and *contextual overlapping relationships* among concrete damage regions in inspection images.

---

## Motivation

Concrete infrastructure inspection often requires experienced engineers to manually analyze images for damage.  
Automating descriptive caption generation for complex damage scenarios can:

- Speed up **visual inspection workflows**
- Aid **documentation and reporting**
- Support **AI-assisted structural assessment**
- Help non-experts *understand* damage context

The key challenge addressed here is **overlapping damage contexts** — situations where multiple types of damage coexist and interact visually.

---

## Features

- Train an custom encoder–decoder based image captioning model from Segmentation Models Library in PyTorch   
- Handle overlapping damage cues in a single image  
- Inference script to generate captions for new images  
- Utility modules for preprocessing and visualization

---

## Repository Structure
```text
.
├── dataset/ # Concrete damage images and annotations
├── captions/ # Caption files for training
├── app.py # Application / demo interface
├── train.py # Training pipeline
├── inference.py # Caption generation script
├── utils.py # Helper functions
├── requirements.txt # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/vedant-girish-dalvi/Image-Captioning-to-explain-context-in-Overlapping-Concrete-Damages.git
cd Image-Captioning-to-explain-context-in-Overlapping-Concrete-Damages
```

### 2. Create and activate a Python (or Conda) environment 

```bash
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```
### 3. Install dependencies

```bash
pip install -r requirements.txt

### 4. Dataset

Place your dataset of concrete damage image + caption pairs under dataset/.
Your dataset should include:

Images (.jpg, .png, etc.)

A captions/annotations file (CSV/JSON) with paired descriptions

If you’re using a custom dataset, update the data loader paths accordingly.

### 5. Training

To train the image captioning model:

```bash
python train.py \
  --data_dir dataset/ \
  --epochs 30 \
  --batch_size 32 \
  --model_out best_model.pth

### 6. Inference

Generate captions for new images:

```bash
python inference.py \
  --image_path path/to/image.jpg \
  --model_path best_model.pth

### 7. Utilities

utils.py: helpers for preprocessing, tokenization, image transforms

Add visualization functions (optional) to overlay captions on images

### 8. Citation

If you use this project or base your research on this work, please cite:

```text
@inproceedings{Dalvi2025ImageCaptioning,
title = {Image Captioning for Building Damage Using Deep Learning: Explaining Context in Overlapping Structural Defects},
author = {Dalvi, Vedant Girish and Martin, Jakob and Weilbach-Eyüboglu, Timur},
booktitle = {Proceedings of the 36th Forum Bauinformatik (FBI 2025)},
pages = {349--357},
year = {2025},
publisher = {RWTH Aachen University},
doi = {10.18154/RWTH-CONV-254876},
url = {https://publications.rwth-aachen.de/record/1021219}

}



