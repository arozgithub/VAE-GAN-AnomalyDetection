# VAE-GAN-AnomalyDetection
A comparative study of GANs and VAEs for anomaly detection in image datasets (MNIST & Fashion MNIST) with real-world extension using VAE on finance, healthcare, and cybersecurity data.

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

## 🤖 Part 2: Generative Adversarial Networks (GANs)

### 🏗️ Architecture:
- Generator: Transforms random noise to image
- Discriminator: Classifies real vs. generated images

**Loss Functions:**
```python
d_loss = -torch.mean(torch.log(D(real)) + torch.log(1 - D(fake)))
g_loss = -torch.mean(torch.log(D(G(z))))
```

### 📈 Training:
- Train on MNIST Digits
- Train on Fashion MNIST (e.g., class "Shoe")

### 🎨 Image Generation:
- Generate 10 random digit images
- Generate 5 images of digit "3" (roll number: L238023)
- Generate fashion images of shoes

---

## 🔁 Part 3: Variational Autoencoders (VAE)

### 🧬 Architecture:
- Encoder → Latent vector z (mu, logvar)
- Reparameterization Trick
- Decoder reconstructs from z

**Loss = BCE + KL Divergence**
```python
kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

### ✅ Tasks:
- Train VAE on MNIST Digits and Fashion MNIST (shoe)
- Visualize latent space using t-SNE or PCA
- Generate new digits, including 5 images of digit "2" (roll number: L238023)

### 🧪 Run:
```bash
python vae/train_vae.py --dataset mnist --digit 2
python vae/train_vae.py --dataset fashion --class shoe
```

---

## 🔍 Part 4: GAN vs. VAE Comparison

| Aspect                | GAN                          | VAE                          |
|-----------------------|------------------------------|------------------------------|
| Image Quality         | High fidelity                | Blurry but diverse           |
| Training Stability    | Sensitive to tuning          | Stable                       |
| Latent Representation | Implicit, hard to interpret  | Explicit and structured      |
| Sampling Control      | Random noise to image        | Vector manipulation possible |

---

## 🌍 Part 5: Real-World Anomaly Detection with VAE

### ✅ Problem Chosen:
**Example: Financial Fraud Detection**

> Fraud costs banks $42B annually.  
> VAE detects suspicious transactions based on reconstruction loss.

### 🧬 Dataset:
- Real-world dataset with labeled normal & anomalous instances

### 🧠 Model:
- VAE trained on only normal data
- Anomalies flagged by reconstruction error

### 📊 Evaluation:
- **Metrics**: Precision, Recall, F1-Score
- **Plots**: Reconstruction Error Histogram, ROC Curve

---

## 🚀 Clone the Repo and Install Dependencies

```bash
git clone https://github.com/arozgithub/VAE-GAN-AnomalyDetection.git
cd generative-anomaly-detection
pip install -r requirements.txt
```
