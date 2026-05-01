🛡️ Fraud Shield – Credit Card Fraud Detection System

📌 Overview
Fraud Shield is an end-to-end machine learning system designed to detect fraudulent credit card transactions in highly imbalanced datasets.
The solution prioritizes recall optimization to minimize missed fraud cases and supports real-time decision-making through an interactive interface.

🚀 Key Features
🔄 End-to-end pipeline: preprocessing, training, evaluation, and deployment
⚖️ Effective handling of class imbalance using SMOTE
🤖 Optimized LightGBM model for high fraud detection performance
🎯 Threshold tuning to enhance sensitivity and reduce missed fraud
🌐 Streamlit dashboard for real-time transaction prediction
🔌 FastAPI backend for model serving

🛠️ Tech Stack
Languages & Tools
- Python
- Streamlit
- FastAPI
- Jupyter Notebook

Libraries
- NumPy
- Pandas
- Scikit-learn
- LightGBM
- Imbalanced-learn
- Plotly
- Matplotlib

⚙️ How It Works
📥 Transaction data is input, cleaned, and scaled
⚖️ Class imbalance is handled using SMOTE
🤖 LightGBM model is trained and optimized
📊 Fraud probability is generated and threshold applied for final decision

📊 Model Performance
🚨 Focus on Recall (primary metric) to reduce false negatives
📈 Evaluated using PR-AUC and ROC-AUC for reliable performance
🎯 Threshold optimization significantly improves fraud detection capability

▶️ Run Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run FastAPI (Terminal 1):
```bash
uvicorn api.app:app --reload
```
API will be available at: http://localhost:8000

3. Run Streamlit UI (Terminal 2):
```bash
streamlit run ui/streamlit_app.py
```
UI will be available at: http://localhost:8501

📁 Project Structure
```
fraud-detection-system/
├── api/
│   ├── app.py          # FastAPI application
│   └── predict.py      # Prediction logic
├── ui/
│   └── streamlit_app.py # Streamlit dashboard
├── src/
│   ├── data_loader.py  # Data loading utilities
│   ├── preprocess.py   # Data preprocessing
│   ├── model_train.py  # Model training
│   ├── model_eval.py   # Model evaluation
│   ├── inference.py    # Inference pipeline
│   └── utils.py        # Utility functions
├── data/
│   ├── raw/            # Raw dataset (gitignored)
│   └── processed/      # Processed data (gitignored)
├── models/             # Trained models (gitignored)
├── notebook/           # Jupyter notebooks
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

🏁 Conclusion
Fraud Shield presents a practical and scalable approach to fraud detection by focusing on high recall and real-world usability.
The system effectively identifies fraudulent transactions while remaining reliable and ready for deployment in real-world financial environments.