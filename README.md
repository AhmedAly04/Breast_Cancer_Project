# 🎗️ Breast Cancer Diagnostic Web App (ML + Streamlit)
This web application allows users to predict whether a breast mass is Malignant or Benign based on clinical image features.
It combines machine learning (RandomForestClassifier) with a Streamlit web interface for real-time diagnostic assistance.

---

## 📚 Table of Contents

- [🎥 Demo](#-demo)
- [✅ What the App Does](#-what-the-app-does)
- [📊 Features Used in the Model](#-features-used-in-the-model)
- [📁 Project Structure](#-project-structure)
- [🧼 Data Exploration & Preprocessing](#-data-exploration--preprocessing)
- [📈 Model Training](#-model-training)
- [🌐 Web App with Django](#-web-app-with-django)
- [📥 Downloads](#-downloads)

---

## 🎥 Demo

[Click to Watch Demo](https://github.com/user-attachments/assets/b3ad520f-59ea-423b-bb89-134cf59612de)

---

## ✅ What the App Does

1. 🎯 Classifies Tumor Type Predicts diagnosis (Malignant/Benign) based on 10 key cell nuclei features.
  
3. ⚖️ Handles Class Imbalance Uses class_weight='balanced' to ensure the model performs accurately even if the dataset has fewer malignant cases.
   
4. 🧠 **Uses an Optimized ML Model**
   A RandomForestClassifier selected after comparing multiple algorithms (Logistic Regression, KNN, SVM, Naive Bayes, and Decision Trees).
   
5. 🌐 **Interactive Web Interface**  
   Users can input ride details and get instant fare prediction via a user-friendly Streamlit app.

---

## 📊 Features Used in the Model.

The model utilizes the following clinical features:

- `radius_mean`.
- `texture_worst`.
- `perimeter_se`.
- `area_se`.
- `area_worst`.
- `compactness_se`.
- `concavity_worst`.
- `concave points_se`.
- `smoothness_worst`.
- `symmetry_worst`.
- `diagnosis` (Target: Benign/Malignant).

---

## 📁 Project Structure

<img width="390" height="220" alt="Untitled" src="https://github.com/user-attachments/assets/94d9c5a3-38e0-4605-9d67-e9d3f295f4c7" />

---

## 🧼 Data Exploration & Preprocessing.

- **Exploratory Data Analysis (EDA)**: Performed using `pandas`, `matplotlib`, and `seaborn` to visualize feature correlations.
- **Feature Scaling**: Implemented a `StandardScaler` to normalize features, ensuring distance-based models (like KNN/SVM) perform correctly.
- **Handling Imbalance**: Applied `Balanced Class Weight` to account for the distribution of diagnosis classes.

---

## 📈 Model Training

- Primary Algorithm: `RandomForestClassifier`.
 Model Comparison: Benchmarked against Logistic Regression, KNN, Decision Trees, SVM, and Naive Bayes.
- Framework: `scikit-learn`.
- Steps:
  - Train/Test split (80/20).
  - Hyperparameter tuning for the Random Forest.
  - Evaluation Metrics: Accuracy, Precision, Recall, and F1-Score (prioritizing high recall for medical safety).
- **Serialization** : Final model and scaler saved as `.pkl` files.

---

## 🌐 Web App with Streamlit.

- Streamlit form takes user inputs for ride parameters.
- Backend loads the trained model and processes features.
- Prediction Clearly displays the diagnosis and the probability of the result.
