# VAE-GAN-AnomalyDetection
A comparative study of GANs and VAEs for anomaly detection in image datasets (MNIST &amp; Fashion MNIST) with real-world extension using VAE on finance, healthcare, and cybersecurity data.

# 🎯 Generative AI-Based Anomaly Detection in Images and Signals

This project implements and compares deep generative models — **GANs**, **Autoencoders**, and **VAEs** — for anomaly detection on image datasets (MNIST Digits & Fashion MNIST), with real-world extension using VAEs on a financially significant anomaly detection problem.

---

## 📌 Objectives

- Build **Generative Adversarial Networks (GANs)** to generate and detect anomalies.
- Train **Autoencoders** & **Variational Autoencoders (VAEs)** to reconstruct and identify outliers.
- Compare generative models in terms of image quality, training stability, and latent space.
- Apply VAE to **real-world anomaly detection** for domains like finance, cybersecurity, and healthcare.

---

## 📂 Datasets

| Dataset         | Description |
|----------------|-------------|
| **MNIST Digits** | 70,000 grayscale 28×28 handwritten digit images |
| **Fashion MNIST** | Zalando’s 70,000 grayscale 28×28 fashion item images |
| **Real-World Dataset** | Domain-specific dataset for anomaly detection (e.g., financial fraud, network intrusion, medical data) |

---

## 🔍 Part 1: Exploratory Data Analysis (EDA)

- Load and preview the MNIST & Fashion MNIST datasets
- Show 5 random images per dataset
- Display number of samples, class labels, and class distributions

🤖 Part 2: Generative Adversarial Networks (GANs)
🏗️ Architecture:
Generator: Transforms random noise to image

Discriminator: Classifies real vs. generated images

Loss Functions:


