# 📊 Sentiment Analysis of Movie Tweets Using Machine Learning

---

## (1) Problem Statement
With the increasing use of social media platforms, analyzing user opinions has become important. This project focuses on performing sentiment analysis on tweets related to a movie. The goal is to classify tweets into Positive, Negative, and Neutral categories using machine learning techniques.

---

## (2) Objective
- To preprocess and clean tweet data  
- To convert text into numerical form using TF-IDF  
- To implement multiple machine learning models  
- To evaluate model performance using standard metrics  
- To identify the best performing model  

---

## (3) Dataset
**Source:** Manually collected tweets  

**Features:**
- ID  
- Tweet Text  
- Sentiment (Positive / Negative / Neutral)  

**Size:**
- Total Tweets: 100  
- Positive: ~35%  
- Negative: ~35%  
- Neutral: ~30%  

---

## (4) Methodology

### Data Preprocessing
- Converted text to lowercase  
- Removed URLs and special characters  
- Cleaned text for better analysis  

### EDA
- Checked distribution of sentiments  
- Observed dataset balance  

### Model Building
- TF-IDF Vectorization  
- Models used:
  - Naive Bayes  
  - Logistic Regression  
  - Support Vector Machine (SVM)  

### Evaluation
- Accuracy  
- Precision (Macro)  
- Recall (Macro)  
- F1 Score (Macro)  
- Cross Validation  

---

## (5) Results
- SVM performed best overall  
- Naive Bayes showed strong precision  
- Logistic Regression performed comparatively lower  

**Insights:**
- Most confusion between Positive and Neutral tweets  
- Negative tweets were easier to classify  

---

## (6) How to Run

```bash
pip install -r requirements.txt
python main.py

## (7) Conclusion
This project demonstrates that machine learning models can effectively classify tweet sentiments. Among all models, SVM performed the best due to its ability to handle high-dimensional text data efficiently.

## (8) Student's details
- Name: Disha Gawade
- Roll No: 19
- UIN: 231A08
- YEAR: TE-AIDS
