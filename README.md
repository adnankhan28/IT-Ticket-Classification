# 🧠 IT Ticket Classification Using Machine Learning

This project uses a Machine Learning model to automatically categorize IT support tickets based on their description.

## ✅ Problem Statement
Support engineers spend time manually classifying IT service requests. This project solves that problem using machine learning.

## 📁 Dataset
- File: `IT_Service_Tickets.csv`
- Fields: `Document` (description), `Topic_group` (category)
- Goal: Predict `Topic_group` based on the ticket description

## 🔧 Technologies Used
- Python, Scikit-learn, Pandas, Matplotlib
- TF-IDF Vectorizer
- Naive Bayes and Logistic Regression models
- Streamlit for interactive app

## 🚀 How It Works
1. **Preprocess**: Clean text and convert it into TF-IDF vectors
2. **Modeling**: Train model to classify support categories
3. **Web App**: Streamlit lets users enter ticket descriptions and get predicted categories

## 📊 Results
Achieved high accuracy using Logistic Regression.  
Use `streamlit_app.py` to test it interactively.

## 📷 Screenshot

The distribution of ticket categories in the dataset:

![Ticket Category Distribution](screenshots/myplot.png)

## 👨‍💻 Author

**Adnan Khan**  
Desktop Support Engineer  
Aspiring Data Scientist  
📍 Bhiwandi, Maharashtra  
📧 adnan.khan282001@gmail.com  
📘 [LinkedIn](https://www.linkedin.com/in/adnankhan282001)


## 🧪 How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py

