import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle
import json
import joblib
from PIL import Image
import matplotlib.pyplot as plt

def run():
    # membagi layout menjadi 3 agar dapat diletakkan di tengah
    col_left, col_mid, col_right = st.columns([2, 2, 2])
    # menambahkan gambar
    image = Image.open('image.webp')
    with col_mid :
        st.image(image)
        # membuat title
        st.title("ChurnSight")

    col_left, col_mid, col_right = st.columns([1.5, 3, 1.5])
    with col_mid :
        st.write("#### Bank Customer Churn Prediction")


    tab1, tab2, tab3 = st.tabs(["Prediction Form","Bulk Prediction", "EDA"])
    #tab1 form input prediction
    with tab1:
        # Create Form
        st.markdown('-----------------')
        st.write('Identity Information')

        surname = st.text_input('Name', value='', help='Customer Name')
        cust_id = st.number_input('Customer ID', min_value=0, max_value=10000)
        geography = st.selectbox('Country', ('France', 'Spain', 'Germany'))

        col_left, col_mid, col_right = st.columns([2, 2, 2])

        gender =  col_left.selectbox('gender', ('Male', 'Female'), index=0)
        with col_mid:
            age = st.number_input('Age', min_value=18, max_value=95) 
        with col_right:
            tenure = st.number_input('Tenure (Year)', min_value=1, max_value=10, step=1)

        st.markdown('-----------------')
        st.write('Financial Information')

        col_left, col_mid, col_right = st.columns([2, 2, 2])
        with col_left:
            creditscore = st.number_input('Credit Score', min_value=300 , max_value=900, help='How likely to pay a loan back on time, based on information from credit report')
        with col_mid:
            balance = st.number_input('Balance', min_value=0, max_value=350000, help='Amount of Balance')
        with col_right:
            estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=350000, help='Estimated customer salary')

        st.markdown('-----------------')
        st.write('Services and Membership')

        num_of_products = st.number_input('Num of Products', min_value=1 , max_value=4, help='Amount of product or services used (max=4)')

        col1, col2 = st.columns([1, 1])
        has_credit_card = col1.radio(label='Has Credit Card?', options=['no', 'yes'])
        active_member = col2.radio(label='Is Active Member?', options=['no', 'yes'])


        # submit button
        submitted = st.button('Predict')

        # load files
        # with open('model_rf.pkl', 'rb') as file_2:
        #     model = pickle.load(file_2)
        with open('list_cat_cols.txt', 'rb') as file_3:
            cat_cols = json.load(file_3)
        with open('list_num_cols.txt', 'rb') as file_4:
            num_cols = json.load(file_4)
        model = joblib.load('model_rf.joblib')

        
        data_inf = {
            'CustomerId' : cust_id,
            'Surname' : surname,
            'geography' : geography,
            'age' : age,
            'gender' : gender,
            'tenure' : tenure,
            'credit_score' : creditscore,
            'balance' : balance,
            'estimated_salary' : estimated_salary,
            'num_of_products' : num_of_products,
            'has_credit_card' : has_credit_card,
            'active_member' : active_member
        }

        # memasukkan data inference ke dataframe
        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf)

        # logic ketika predict button ditekan
        if submitted:
            data_inf_drop = data_inf.drop(['CustomerId', 'Surname'], axis=1)

        # predict
            y_pred_inf = model.predict(data_inf_drop)

            # conditional if 
            if y_pred_inf == 0:
                st.write('### Prediction : Not Churn')
            else :
                st.write('### Prediction : Churn')
                
        with tab2:
            st.write('## Customer Data')
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

            #Condition upload data
            if uploaded_file is not None:
                df_uploaded = pd.read_csv(uploaded_file)

                
                # Nama kolom dataset
                data_inf = {
                    'customer_id',
                    'Surname',
                    'geography',
                    'age',
                    'gender',
                    'tenure',
                    'credit_score',
                    'balance',
                    'estimated_salary',
                    'num_of_products',
                    'has_credit_card',
                    'active_member'
                }

                #Menjadikan dataframe
                st.dataframe(df_uploaded)
                
                df_uploaded_drop = df_uploaded.drop(['customer_id', 'Surname'], axis=1)


                # Predict menggunakan logistic regression model
                y_pred_inf = model.predict(df_uploaded_drop)

                #Menambahkan hasil predict ke dataframe
                df_uploaded['predicted_churn'] = np.where(y_pred_inf == 1, 'Churn', 'Not Churn')
                
                # # #Membuat bar plot
                # # menghitung total masing-masing uniq value pada kolom 'default_payment_next_month'
                default_counts = df_uploaded['predicted_churn'].value_counts()
                # judul grafik
                st.title('Distribution of Customer Churn Prediction')
                #Membuat grafik
                fig, ax = plt.subplots(figsize=(4, 4))
                colors = ['lightgreen', 'lightcoral']
                ax.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')

                # Menampilkan grafik
                st.pyplot(fig)
                
                #Membuat tab untuk menampilkan tabel sesuai hasil prediksi
                tab5, tab6 = st.tabs(["Customer Churn", "Customer Not Churn"])

                #Membuat tab untuk menampilkan tabel dengan prediksi insomnia
                with tab5:
                    #Header
                    st.write('## **Prediction Results for Churn Customer**')
                    
                    #Menampilkan tabel dengan hasil prediksi insomnia
                    st.dataframe(df_uploaded[(df_uploaded['predicted_churn'] == 'Churn')])
                    
                with tab6:
                #Header
                    st.write('## **Prediction Results for Non Churn Customer**')
                    
                    #Menampilkan tabel dengan hasil prediksi Sleep Apnea
                    st.dataframe(df_uploaded[(df_uploaded['predicted_churn'] == 'Not Churn')])

        # tab 2 eda
        with tab3:
            iframe_html = """
            <iframe width="800" height="1280" src="https://lookerstudio.google.com/embed/reporting/fda9f4ee-7b07-4d62-a910-de031a2211e6/page/CPSpD" frameborder="0" style="border:0" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>
            """

            st.markdown(iframe_html, unsafe_allow_html=True)
            # looker_dashboard_url = "https://lookerstudio.google.com/s/omKUXbpaEps"
            # st.markdown(f"[Go to Looker Dashboard]({looker_dashboard_url})")

if __name__ == '__main__':
    run()

