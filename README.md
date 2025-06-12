# 🧠 Conditional GAN (WGAN-GP) for MNIST Digit-Pair Image Generation

This project implements a **Conditional Generative Adversarial Network (GAN)** based on the **Wasserstein GAN with Gradient Penalty (WGAN-GP)** framework. The model is trained to generate horizontal combinations of two MNIST digits, conditioned on whether the combined number is less than 50 or not.

---

## 📦 Dataset Preparation

- **Base dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Customization**:
  - Combines two grayscale digit images side-by-side to form a single `28x56` image
  - The target label is calculated as `10 × digit1 + digit2` (e.g., 3 and 7 → 37)
  - A binary condition is set:
    - `1` if the combined label is ≤ 49
    - `0` otherwise

---

## ⚙️ Model Architecture

### Generator
- Takes random noise and a condition (0 or 1) as input
- Fully connected network with ReLU activations
- Output reshaped to `28x56` image
- Final activation: `tanh`

### Critic (Discriminator)
- Receives a flattened image and the associated condition
- Outputs a **Wasserstein score** (no sigmoid activation)


## 📐 Training Details

- **WGAN-GP** is used to stabilize GAN training via gradient penalty
- **Loss functions**:
  - Critic loss = Wasserstein loss + Gradient penalty
  - Generator loss = Negative of critic's prediction
- **Optimizer**: Adam with `lr=0.0001`, `betas=(0.5, 0.9)`
- **Batch size**: 64
- **Epochs**: 30
- **Noise dimension**: 100



## 🧪 Image Generation

- The trained generator can sample new images by:
  - Generating a latent vector `z` from a standard normal distribution
  - Feeding it along with a **binary condition** (0 or 1)
  - Producing new `28x56` digit-pair images
    
## 🧭 Key Learnings

-Implementing WGAN-GP for stable GAN training
-Creating a custom PyTorch dataset for multi-digit image generation
-Training conditional generative models
-Visualizing training stability and sample diversity

## 🔮 Future Work

-Add support for class-conditional generation using digit class embeddings
-Compare performance with Conditional VAEs and StyleGAN variants
-Explore multi-label conditioning using auxiliary classifiers
-Deploy model with Streamlit for interactive demo


## 📬 Contact

**Madhav Dahal**  
📧 madhavdahal16@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/madhav-dahal-ms-9a1147b0)  
🔗 [GitHub](https://github.com/Madhav4487)
