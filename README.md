
# Fraud Detection Project

## 📌 Overview
This project demonstrates a **machine learning pipeline** for fraud detection using multiple models (RandomForest, LightGBM, XGBoost).  
It includes **data preprocessing**, **model training with hyperparameter tuning**, **evaluation**, and an **interactive dashboard** built with Streamlit.

The repository is structured to follow **professional ML project standards** with a clear folder organization, documentation, and reproducible setup.

---

## 📂 Project Structure
```
fraud_detection_project/
│
├── data/
│   ├── raw/               # raw input data
│   ├── processed/         # cleaned data
│
├── notebooks/             # Jupyter notebooks for exploration & training
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│
├── src/                   # project source code
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── compare_models.py
│   ├── predict.py
│
├── models/                # saved trained models (.pkl)
├── reports/               # graphs, metrics, PDF reports
│   ├── figures/           # visualizations
│
├── app/                   # Streamlit app
│
├── requirements.txt       # dependencies
├── README.md              
├── LICENSE                
```

---

## 📊 Models Implemented
- **RandomForestClassifier**
- **LightGBMClassifier**
- **XGBoostClassifier**

Models are trained via **Pipeline** in scikit-learn with:
- Data preprocessing (encoding categorical features, scaling numeric features)
- Cross-validation (StratifiedKFold)
- Hyperparameter tuning (GridSearchCV, Optuna)

---

## 🚀 Quickstart

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/fraud_detection_project.git
cd fraud_detection_project
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run training with demo data
The repository includes a small **synthetic dataset** (`data/raw/sample.csv`) so you can run the project instantly:
```bash
python src/train_models.py
```

Note: Model training with GridSearchCV may take several minutes

Trained models will be saved in the `models/` folder.

### 4️⃣ Launch the dashboard
```bash
streamlit run app/app.py
```

---

## 📂 Data Sources

### Demo Data (included)
- **`data/raw/sample.csv`** — 200 synthetic transactions with realistic distributions.

### Real Fraud Dataset (optional)
You can retrain the models on the **Credit Card Fraud Detection** dataset from Kaggle:  
🔗 [https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Steps:
1. Download the dataset from Kaggle.
2. Place the CSV in `data/raw/` and update the data path in `train_models.py`.
3. Retrain using:
```bash
python src/train_models.py
```

---

## 📈 Reports & Visuals
- **ROC curves** comparing models
- **Pipeline diagram** (`reports/figures/pipeline_schema.png`)
- **Performance metrics table**

Example ROC Curve:  
![ROC Curve](reports/figures/roc_curve_example.png)

Pipeline Diagram:  
![Pipeline](reports/figures/pipeline_schema.png)

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ✨ Author
Developed by **Your Name** — ML Engineer  
GitHub: [yourusername](https://github.com/yourusername)
