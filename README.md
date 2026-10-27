# 🔢 MNIST Digit Classifier Streamlit App

A lightweight, fast handwritten digit recognition web app built with Python, scikit-learn, and Streamlit. Ready for 1-click deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97.1%25-brightgreen)

---

## ✨ Features
- ✍️ **Interactive Canvas**: Draw digits directly with adjustable stroke width.
- 📁 **Image Upload**: Upload `.png`, `.jpg`, or `.jpeg` images of handwritten digits with automatic background inversion.
- 🎲 **Sample Digits**: Instant testing using pre-extracted real MNIST test set samples.
- 📊 **Probability Bar Chart**: Live confidence scores and distribution across all 10 digits (0–9).
- ⚡ **Lightweight & Fast**: Model footprint is only ~2.5 MB (instant cold-start on Streamlit Cloud).

---

## 🚀 Deploy to Streamlit Community Cloud (streamlit.io)

1. **Push this repository to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for MNIST Streamlit App"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Sign in with GitHub.
   - Click **"New app"**.
   - Select your repository and branch (`main`).
   - Set **Main file path** to `app.py`.
   - Click **Deploy**!

---

## 💻 Local Setup & Development

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Re-train the model**:
   ```bash
   python train_model.py
   ```

3. **Run the app locally**:
   ```bash
   streamlit run app.py
   ```
