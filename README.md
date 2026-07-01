# Customer Segmentation & Churn Pattern Analytics in European Banking

## Project Overview
This project analyzes customer churn patterns in a 
European bank using Machine Learning and an interactive 
Streamlit dashboard. It identifies customers likely to 
leave the bank and provides actionable business insights 
for improving customer retention across France, Germany, 
and Spain.

## Live Dashboard
🌐 [Click here to view Dashboard](#) 
← Add your Streamlit URL after deployment

## Features
- 📊 Overall Churn KPI Dashboard
- 🗺️ Geographic Churn Analysis
- 👥 Customer Segmentation Analysis
- 🤖 Real-time Churn Predictor
- 🔍 Interactive Filters by Country & Gender
- 💡 Business Recommendations

## Key Findings
- Overall Churn Rate: 20.37%
- Age 46-60 has highest churn: 51.12% (Critical!)
- Germany has highest regional churn: 32.44%
- High balance customers churn at: 25.23%
- Inactive members churn at: 26.85%

## Machine Learning Model
- Algorithm: Gradient Boosting Classifier
- Accuracy: 87.00%
- Best Feature: Age (strongest churn predictor)
- Models Compared: Logistic Regression, Decision Tree,
  Random Forest, Gradient Boosting

## Technologies Used
- Python 3.11
- Pandas - Data Analysis
- NumPy - Numerical Computing
- Matplotlib & Seaborn - Visualization
- Plotly - Interactive Charts
- Streamlit - Web Dashboard
- Scikit-learn - Machine Learning
- Pickle - Model Saving

## Project Structure
European_Bank_Project/
├── app.py               # Streamlit Dashboard
├── train_model.py       # Model Training Script
├── customer_churn.csv   # Dataset
├── churn_model.pkl      # Trained ML Model
├── scaler.pkl           # Feature Scaler
├── requirements.txt     # Dependencies
└── README.md            # Project Documentation

## Dataset
- Source: European Bank Customer Dataset
- Records: 10,000 customers
- Features: 11 analytical features
- Target: Exited (Churn Indicator)
- Countries: France, Germany, Spain

## Installation & Setup
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Train the model:
   python train_model.py
4. Run the dashboard:
   python -m streamlit run app.py

## Author
**Gouranga Ghorai**
B.Tech CSE (AI & ML)
Machine Learning Intern - Unified Mentor
Finance Analytics Domain