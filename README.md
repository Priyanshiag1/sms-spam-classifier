# 📩 SMS Spam Classifier

A Machine Learning project that classifies SMS messages as **Spam** or **Ham** (Not Spam). Built using Python, trained and evaluated with multiple algorithms in **Jupyter Notebook**, and deployed as a web app using **Streamlit**.

## 🔍 Project Overview

This project uses **Machine Learning** to classify SMS text messages into two categories:
- **Spam** (unsolicited messages)
- **Ham** (legitimate messages)

The final model selected is **Multinomial Naive Bayes**, which performed the best in terms of accuracy and F1-score among all tested algorithms.

The classifier has been deployed as an interactive **Streamlit web app** where users can enter a message and instantly get predictions.

## 🛠 Tech Stack

- **Python**
- **Jupyter Notebook**
- **Pandas, NumPy, Matplotlib, Seaborn** (for data analysis & visualization)
- **Scikit-learn** (for machine learning models)
- **NLTK** (for text preprocessing)
- **Streamlit** (for web app)

## 📊 Modeling Approach

### 1. **Data Cleaning and Preprocessing**
- Removed punctuation and special characters
- Converted to lowercase
- Removed stopwords
- Applied stemming using **PorterStemmer**
- Transformed using **TF-IDF Vectorizer**

### 2. **Exploratory Data Analysis (EDA)**
- Message length distribution
- Word frequency analysis
- Spam vs. Ham comparison

### 3. **Model Training**
Tested several models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- **Multinomial Naive Bayes (final model)**

### 4. **Evaluation Metrics**
- Accuracy
- Precision

## ⚙️ How It Works

1. User enters an SMS message in the web interface.
2. The text is preprocessed and vectorized.
3. The trained Naive Bayes model predicts the category.
4. Result is displayed: **Spam** or **Ham**
