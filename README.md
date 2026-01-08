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

✔ Train an custom encoder–decoder based image captioning model from Segmentation Models Library in PyTorch   
✔ Handle overlapping damage cues in a single image  
✔ Inference script to generate captions for new images  
✔ Utility modules for preprocessing and visualization

---

## 📦 Repository Structure

