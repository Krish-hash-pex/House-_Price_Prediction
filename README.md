# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project aims to predict house prices based on various features such as size, location, and other housing attributes using **Machine Learning regression models**. The goal is to build an end-to-end data science pipeline including data preprocessing, exploratory data analysis, model training, evaluation, and interpretation.

This project is suitable for **beginners/freshers** and is fully **CV and GitHub ready**.

---

## 🎯 Problem Statement

Accurately estimating house prices is important for buyers, sellers, and real estate companies. The objective of this project is to develop a machine learning model that predicts house prices using historical housing data.

---

## 📂 Dataset

* Source: CSV-based housing price dataset
* File used: `data.csv`
* Target variable: `price`

The dataset contains numerical and categorical features related to houses.

---

## 🛠️ Tools & Technologies

* **Programming Language:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Matplotlib
  * Seaborn
  * Scikit-learn
* **IDE:** Visual Studio Code
* **Version Control:** Git & GitHub

---

## 🔍 Exploratory Data Analysis (EDA)

* Checked for missing values
* Analyzed price distribution using histograms
* Visualized important features affecting house prices

📊 Generated Visuals:

* `price_distribution.png`
* `feature_importance.png`

---

## ⚙️ Data Preprocessing

* Handled missing numerical values using mean imputation
* Converted categorical variables into numerical form using one-hot encoding
* Split the dataset into training and testing sets (80/20 split)

---

## 🤖 Machine Learning Models Used

### 1️⃣ Linear Regression

* Baseline regression model
* Used to understand linear relationships between features and price

### 2️⃣ Random Forest Regressor

* Ensemble learning model
* Provided better performance compared to Linear Regression

---

## 📈 Model Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **R² Score**

Random Forest achieved a higher R² score, indicating better predictive performance.

---

## 📊 Feature Importance

Top features influencing house prices were extracted using the Random Forest model and visualized using a bar chart.

---

## 📁 Project Structure

```
HPP/
│── house_price.py
│── data.csv
│── price_distribution.png
│── feature_importance.png
│── README.md
│── venv/
```

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone <repository-url>
```

2. Navigate to the project directory

```bash
cd HPP
```

3. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

4. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

5. Run the script

```bash
python house_price.py
```

---

## 📌 Results

* Successfully built an end-to-end regression pipeline
* Random Forest model outperformed Linear Regression
* Visual insights generated for better interpretability

---

## 📝 Conclusion

This project demonstrates the complete workflow of a Data Science project—from data preprocessing and visualization to machine learning model building and evaluation. It serves as a strong foundation for more advanced predictive modeling projects.

---

## 👤 Author

**Krishnadwaipayan Ghosh**
Aspiring Data Scientist

---

## ⭐ Acknowledgements

* Scikit-learn Documentation
* Kaggle Datasets
* Python Community

---

> ⭐ If you found this project useful, feel free to star the repository!
