# 🛡️ Fraud Shield – Credit Card Fraud Detection System

## 📌 Overview
Fraud Shield is an end-to-end machine learning system designed to detect fraudulent credit card transactions in highly imbalanced datasets.  
The solution prioritizes recall optimization to minimize missed fraud cases and supports real-time decision-making through an interactive interface.

---

## 🚀 Key Features

- 🔄 End-to-end pipeline: preprocessing, training, evaluation, and deployment  
- ⚖️ Effective handling of class imbalance using SMOTE  
- 🤖 Optimized LightGBM model for high fraud detection performance  
- 🎯 Threshold tuning to enhance sensitivity and reduce missed fraud  
- 🌐 Streamlit dashboard for real-time transaction prediction  
- 🔌 FastAPI backend for model serving  

---

## 🛠️ Tech Stack

### Languages & Tools
- Python  
- Streamlit  
- FastAPI  
- Jupyter Notebook  

### Libraries
- NumPy  
- Pandas  
- Scikit-learn  
- LightGBM  
- Imbalanced-learn  
- Plotly  
- Matplotlib  

---

## ⚙️ How It Works

1. 📥 Transaction data is input, cleaned, and scaled  
2. ⚖️ Class imbalance is handled using SMOTE  
3. 🤖 LightGBM model is trained and optimized  
4. 📊 Fraud probability is generated and threshold applied for final decision  

---

## 📊 Model Performance

- 🚨 Focus on Recall (primary metric) to reduce false negatives  
- 📈 Evaluated using PR-AUC and ROC-AUC  
- 🎯 Threshold optimization improves fraud detection capability  

---

## ▶️ Run Instructions

### Install dependencies
```bash
pip install -r requirements.txt

Run FastAPI (Terminal 1)
uvicorn api.app:app --reload

API will be available locally on port 8000

Run Streamlit UI (Terminal 2)
streamlit run ui/streamlit_app.py

UI will be available locally on port 8501

📁 Project Structure
fraud-detection-system/
├── api/
│   ├── app.py
│   └── predict.py
├── ui/
│   └── streamlit_app.py
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model_train.py
│   ├── model_eval.py
│   ├── inference.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebook/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md

🏁 Conclusion

Fraud Shield presents a practical and scalable approach to fraud detection by focusing on high recall and real-world usability.
The system effectively identifies fraudulent transactions while remaining reliable and ready for deployment in real-world financial environments.