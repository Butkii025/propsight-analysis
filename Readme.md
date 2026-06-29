<div align="center">

<img src="https://img.shields.io/badge/PropSight-Real%20Estate%20AI-4f46e5?style=for-the-badge&logo=house&logoColor=white" alt="PropSight"/>

# 🏡 PropSight — Real Estate Valuation Platform

**Predict residential property values with machine learning, powered by Streamlit.**

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-propsight.streamlit.app-ff4b4b?style=for-the-badge)](https://propsight.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## ✨ What is PropSight?

PropSight is an interactive real estate valuation web app that uses ensemble machine learning models to estimate house prices based on property features. Simply enter a home's specs — size, quality, neighborhood, year built — and PropSight returns an instant price prediction backed by three trained models: **Ridge Regression**, **Lasso Regression**, and a **Gradient Boosting Regressor**.

> 📊 **Held-out test accuracy: R² = 93.68% · MAE = $12,695**

---

## 🎯 Features

- 🔮 **Instant price prediction** — enter property details and get a valuation in real time
- 🤖 **Three ML models** — Ridge, Lasso, and Gradient Boosting with live comparison
- 📈 **Key price drivers** — feature importance chart pulled live from the trained GBR model
- 🏘️ **Neighborhood-aware** — one-hot encoded neighborhood effects trained into the model
- 📄 **Technical documentation** — downloadable PDF with full methodology
- 🎛️ **Sensible defaults** — non-user features default to dataset medians (no phantom $184M predictions)

---

## 🖥️ Try It Live

**No installation needed** — the app is deployed and ready at:

> **[https://propsight.streamlit.app](https://propsight.streamlit.app)**


DataSet Available at Kaggle:

> **[Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submissions#)**

---

## 🗂️ Project Structure

```
propsight-analysis/
├── app.py                      # 🚀 Streamlit app (entry point)
├── requirements.txt            # 📦 Python dependencies
├── notebook.ipynb              # 📓 Exploratory data analysis
├── Technical_Documentation.pdf # 📄 Full methodology writeup
│
├── data/
│   └── train.csv               # 🏠 Ames Housing dataset
│
└── models/
    ├── ridge_model.pkl         # Ridge regression
    ├── lasso_model.pkl         # Lasso regression
    ├── gbr_model.pkl           # Gradient Boosting (primary)
    ├── model_columns.pkl       # Feature schema
    ├── feature_medians.pkl     # Median defaults for non-user inputs
    ├── feature_importances.pkl # Live driver chart data
    └── metrics.pkl             # Honest held-out test metrics
```

---

## ⚡ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Butkii025/propsight-analysis.git
cd propsight-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧠 Models & Performance

| Model | Description | Role |
|---|---|---|
| **Gradient Boosting** | 500 trees, max depth 4 | Primary predictor |
| **Ridge Regression** | L2-regularised linear | Comparison baseline |
| **Lasso Regression** | L1-regularised linear | Comparison baseline |

All models are trained on the **Ames Housing dataset** and evaluated on a held-out test split.

```
Test R²  →  93.68%
Test MAE →  $12,695
```

---

## 🔧 Retraining the Model

If you update `data/train.csv`, regenerate all model artifacts with:

```bash
python train_model.py
```

Then push the updated `models/` folder — Streamlit Cloud auto-redeploys on every push.

```bash
git add models/
git commit -m "retrain: updated model artifacts"
git push
```

---

## 🐛 Bugs Fixed vs. Original

| Bug | File | Fix |
|---|---|---|
| Default inputs predicted **$184M** | `app.py` | Non-user features now default to dataset **medians** |
| Neighborhood had **zero effect** on price | `train_model.py` | Neighborhood one-hot columns now trained and saved |
| R²/MAE computed on **training data** | `train_model.py` | Reports honest **held-out test** metrics |
| "Key Drivers" chart was **hardcoded** | `app.py` | Pulled live from `gbr_model.feature_importances_` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML Models | scikit-learn (Ridge, Lasso, GBR) |
| Data Wrangling | pandas, numpy |
| Visualisation | Plotly |
| Model Serialisation | joblib |
| Deployment | Streamlit Community Cloud |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

</div>
