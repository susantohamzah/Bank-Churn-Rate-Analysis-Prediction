# **Bank Churn Rate Analysis Prediction**

## **Objective**
A bank company currently facing some issues regarding churning customers. To face such issue, a machine learning model is created in order to determine the variables that are responsible in making customers churn. To build said machine learning model, there are numerous general information of customers, compiled into a dataset which will be the base of this machine learning model creation and analysis. The final objective is to create a model that is highly capable in predicting whether a customer will churn from the bank or not.

## **Problem Solved**
This project aims to address the issue of high customer churn rate in a bank company. By developing a machine learning model, the project seeks to identify the key factors contributing to customer churn and predict whether a customer is likely to churn or not. Ultimately, the goal is to help the bank company reduce customer churn and improve customer retention.

## **Background**
High customer churn rates can have detrimental effects on a bank company, including loss of revenue, decreased customer satisfaction, and increased marketing costs to acquire new customers. Understanding the factors influencing customer churn and implementing effective retention strategies are crucial for maintaining a stable customer base and fostering long-term relationships with customers.

## **Tech Stack**
- **Programming Languages:** Pandas, scikit-learn
- **Data Analysis Tools:** Matplotlib, seaborn
- **Machine Learning Frameworks:** Hugging Face
- **Visualization Tools:** Looker, BigQuery
- **Deployment Tools:** Streamlit

## **Prediction Web**
[Churn Prediction Web](https://huggingface.co/spaces/hammammahdy/bank_customers_churn_prediction)

## **Google Looker Analytics Dashboard**
[Churn Analytics Dashboard](https://lookerstudio.google.com/reporting/fda9f4ee-7b07-4d62-a910-de031a2211e6)

## **Analysis Slide**
[Download Slide Here](https://docs.google.com/presentation/d/16IsHyTlZeHITvP_m_WnhH-w6HvIiv-zH1C1KbYPu_0o/edit?usp=sharing)

### **File Explanation**
- `/Deployment/app.py`: Script for deployment using Streamlit.
- `/Deployment/image.webp`: Image file for the deployment.
- `/Deployment/model_rf.joblib`: Serialized machine learning model file.
- `/Deployment/model_rf.pkl`: Serialized machine learning model file.
- `/Deployment/prediction.py`: Script for prediction in the deployment.
- `/Deployment/requirements.txt`: List of dependencies required for deployment.
- `Machine Learning.ipynb`: Jupyter notebook containing machine learning model development.
- `README.md`: Project overview and instructions.
- `url.txt`: File containing URLs for model deployment and other resources.

## **Database and Automation**
This project utilizes Google BigQuery as the database and implements data automation using BigQuery queues to send data for visualization in Google Data Studio.
