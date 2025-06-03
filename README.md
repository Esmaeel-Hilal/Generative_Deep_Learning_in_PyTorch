# Generative_Deep_Learning_in_PyTorch

This repository contains my PyTorch implementations of the models and experiments from the book  
**_Generative Deep Learning_ by David Foster (2nd Edition, 2023)**.

The purpose of this project is twofold:

1. **Faithfully reproduce the original code and models from the book** to understand core generative modeling concepts.
2. **Extend and experiment** beyond the book to deepen my intuition, explore new ideas, and build a reusable PyTorch-based codebase.

---

## 📁 Repository Structure
├── LICENSE
├── README.md
├── book_implementations_notebooks/
├── book_implementations_modular/
└── experiments/


### 🔹 `book_implementations_notebooks/`
Contains Jupyter notebooks that closely follow the style and content of the book.  
Each notebook is organized according to chapter and section (e.g., `03_vae/01_autoencoder`) for clarity and traceability.  
The purpose is to recreate and validate the book’s original TensorFlow/Keras implementations in PyTorch, step-by-step.

### 🔹 `book_implementations_modular/`
Contains modular, reusable `.py` implementations of the same models.  
This version separates functionality into clean components (`model.py`, `train.py`, `utils.py`, etc.) and is suitable for:
- Running from scripts
- Plugging into other pipelines
- Building larger projects

Ideal for engineering best practices and code reuse.

### 🔹 `experiments/`
My own experiments and extensions that go beyond the original book code.  
Examples include:
- Denoising autoencoders
- Latent space visualizations
- Architectural variations
- Additional loss functions

This folder reflects my active exploration and creative work with generative models.

---

## 🚧 Status

This project is under active development. Current focus:
- ✅ Autoencoders
- 🔄 Variational Autoencoders
- ⏳ GANs and Diffusion Models (coming soon)

---

## 📚 Acknowledgments

This repository is inspired by the structure and content of *Generative Deep Learning (2nd Ed.)* by David Foster, and aims to faithfully reproduce and expand upon its ideas in PyTorch.  
All credit for the original model designs and educational value goes to the author.
