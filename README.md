# 💊 InteracTox : Drug-Drug Interaction Predictor

An advanced machine learning application to *predict drug-drug interactions (DDIs)* and their *severity levels*.  
This project has two components:
1. **Streamlit Web App (app1.py)** – A user-friendly interface for training, prediction, and visualization.  
2. **CLI Script (new.py)** – A command-line interface for training and testing the model interactively.  

---

## 📂 Project Structure


├── app1.py        # Streamlit web application
├── new\.py         # CLI-based training and prediction script
├── DDI\_1\_\_1\_.csv  # Example dataset (replace with your own)
├── requirements.txt
└── README.md

`

---

## ⚙ Installation

1. Clone this repository:
   
   git clone https://github.com/your-username/drug-interaction-predictor.git
   cd drug-interaction-predictor


2. Create a virtual environment (recommended):

   
   python -m venv myenv
   source myenv/bin/activate   # On Windows: myenv\Scripts\activate
   

3. Install dependencies:

   bash
   pip install -r requirements.txt
   

---

## 📦 Requirements

Your requirements.txt should include:


streamlit
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
plotly
joblib


---

## 🚀 Usage

### 1️⃣ Run Streamlit Web App

bash
streamlit run app1.py


* *Home Page*: Introduction and features
* *Model Training*: Upload CSV and train the model
* *Drug Prediction*: Enter two drugs to predict interactions
* *Available Drugs*: Browse searchable list of known drugs
* *About*: Technical details

---

### 2️⃣ Run CLI Script

bash
python new.py


* Trains the model using the dataset (DDI_1__1_.csv by default).
* Allows interactive drug-drug interaction predictions directly in terminal.
* Type quit to exit.

Example:


Enter first drug name: aspirin
Enter second drug name: warfarin

Prediction Results:
Drug 1: aspirin
Drug 2: warfarin
Interaction Type: Synergistic
Severity: High


---

## 📊 Dataset Format

CSV file should include columns:

* Drug1Name
* Drug2Name
* InteractionType
* Severity

---

## 🧬 Model Details

* *Algorithms*: Random Forest, XGBoost, Logistic Regression (stacking ensemble)
* *Preprocessing*: Label encoding, StandardScaler, ADASYN (for class imbalance)
* *Features*: Encoded drug names, sum/product/difference features
* *Outputs*:

  * Predicted *interaction type*
  * Predicted *severity*
  * Confidence scores 

---

## ⚠ Disclaimer

This project is for *research and educational purposes only*.
It is *not a substitute for professional medical advice*. Always consult healthcare professionals before making medical decisions.
