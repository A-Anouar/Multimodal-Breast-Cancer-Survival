# Multimodal AI for Breast Cancer Survival Prediction 🧬🔬

This repository contains the source code for a Multimodal Artificial Intelligence framework designed to predict breast cancer survival. By utilizing a **Decision-Level Late Fusion** strategy, this project bridges the data silos between digital pathology and high-throughput molecular biology.

## 📌 Project Architecture

Given the real-world clinical constraints of unaligned datasets (e.g., missing paired modalities in TCGA-BRCA cohorts), this framework relies on two independent predictive experts:

1. **The Histopathological Vision Pillar:** 
   - Processes 224x224 pixel H&E stained Whole-Slide Image (WSI) patches.
   - Utilizes Transfer Learning via a fine-tuned **ResNet50** (ImageNet backbone) to extract spatial tissue morphologies (e.g., nuclear atypia).
   
2. **The Genomic Pillar:**
   - Processes high-dimensional transcriptomic profiles (1,937 pure gene expressions).
   - Utilizes an unsupervised **Deep Auto-Encoder** for dimensionality reduction (compressing genes into a 128-feature latent space).
   - A **Multi-Layer Perceptron (MLP)** classifies the molecular signatures, stabilized by a dynamic Early Stopping mechanism at epoch 5 to prevent clinical overfitting.

3. **Late Fusion:**
   - The visual and molecular risk probabilities are mathematically synthesized using a weighted average to correct unimodal blind spots and provide a robust clinical consensus.

## 🚀 Local Deployment (Clinical Simulator)

The predictive weights are deployed locally into an interactive diagnostic simulator using Streamlit.

### Installation
```bash
pip install -r requirements.txt
