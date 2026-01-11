# 🛡️ Autoencoder Fraud Detection System

A robust, full-stack anomaly detection application built with **PyTorch** and **Streamlit**. This project uses an unsupervised Autoencoder Deep Learning model to detect fraudulent credit card transactions based on reconstruction error, trained on real-world financial data.

## 🚀 Live Demo
*(Optional: Add your Hugging Face Space link here if you deploy it, e.g., [View Live App](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME))*

## 📊 Dataset
The model was trained on the **Credit Card Fraud Detection** dataset.
* **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Content:** Transactions made by credit cards in September 2013 by European cardholders. The dataset is highly unbalanced, with frauds accounting for only 0.172% of all transactions.
* **Privacy:** Features `V1`, `V2`, ... `V28` are the result of a PCA transformation to protect user confidentiality. `Time` and `Amount` are the only original features.

## 🧠 Model Architecture
We use an **Undercomplete Autoencoder** to learn the latent representation of *normal* transactions.
* **Objective:** Minimize reconstruction error (MSE) on legitimate transactions.
* **Anomaly Detection:** Fraudulent transactions differ significantly from the learned pattern, resulting in a high reconstruction error (MSE > Threshold).
* **Explainability:** The system calculates feature-wise error contributions to explain *why* a specific transaction was flagged (e.g., "Amount was too high" or "V14 anomaly").

## ✨ Key Features
* **Real-time Simulation:** Interactive dashboard to simulate normal vs. fraudulent scenarios.
* **Batch Processing:** Upload a CSV file of transactions to process thousands of records instantly.
* **PDF Reporting:** Generates downloadable "Risk Reports" summarizing total money at risk and fraud rates.
* **Explainable AI (XAI):** Visualizes the top features contributing to the fraud score, helping analysts trust the decision.

## 🛠️ Tech Stack
* **Core:** Python 3.10
* **Deep Learning:** PyTorch
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Docker, Hugging Face Spaces

## 📦 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
