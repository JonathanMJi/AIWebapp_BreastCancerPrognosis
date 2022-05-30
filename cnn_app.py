from cv2 import convertMaps
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from streamlit_option_menu import option_menu
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('BreastCancer_DL.h5')

def predict_survival(age_at_diagnosis, overall_survival_months, lymph_nodes_examined_positive, tumor_size, tumor_stage, brca1, brca2, tp53, pten, egfr):
    df = pd.read_csv('METABRIC_RNA_Mutation_Signature_Preprocessed.csv', delimiter=',')
    #Convert Categorical values to Numerical values
    #features_to_drop = df.columns[52:]
    #df = df.drop(features_to_drop, axis=1)
    #all_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    #unwanted_columns = ['patient_id', 'death_from_cancer']
    #all_categorical_columns = [ele for ele in all_categorical_columns if ele not in unwanted_columns]
    #dummies_df = pd.get_dummies(df.drop('patient_id',axis=1),columns = all_categorical_columns,dummy_na=True)
    #dummies_df.dropna(inplace = True)
    X = df.drop(['death_from_cancer', 'overall_survival'], axis = 1)
    TestData = X.iloc[[9],:]
    TestData [['age_at_diagnosis', 'overall_survival_months','lymph_nodes_examined_positive', 'tumor_size', 'tumor_stage', 'brca1', 'brca2', 'tp53', 'pten', 'egfr']] = [age_at_diagnosis, overall_survival_months, lymph_nodes_examined_positive,tumor_size,tumor_stage,brca1,brca2,tp53,pten,egfr]
    TestData = np.asarray(TestData).astype(np.float32)
    prediction = model.predict(TestData)
    pred='{0:.{1}f}'.format(prediction[0][0],2)
    return float(pred)

def draw_plot():
            df = pd.read_csv('METABRIC_RNA_Mutation_Signature_Preprocessed.csv', delimiter=',')
            #features_to_drop = df.columns[52:]
            #df = df.drop(features_to_drop, axis=1)
            #all_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            #unwanted_columns = ['patient_id','death_from_cancer' ]
            #all_categorical_columns = [ele for ele in all_categorical_columns if ele not in unwanted_columns] 
            #dummies_df = pd.get_dummies(df.drop('patient_id',axis=1 ), columns= all_categorical_columns, dummy_na=True)
            #dummies_df.dropna(inplace = True)
            X = df.drop( ['death_from_cancer','overall_survival'], axis=1)
            y = df['overall_survival']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
            y_test_pred = model.predict(X_test)
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_test_pred)
            auc_keras = auc(fpr_keras, tpr_keras)

            #using stratify for y because we need the distribution of the two classes to be equal in train and test sets.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
            y_test_pred = model.predict(X_test)
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_test_pred)
            auc_keras = auc(fpr_keras, tpr_keras)
   
            # ROC curve of testing data
            fig, ax = plt.subplots()
            ax.plot(fpr_keras, tpr_keras)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False positive rate')
            ax.set_ylabel('True positive rate')
            ax.set_title('ROC curve for CNN model')
            
            variable_output = 'CNN (area = {:.3f})'.format(auc_keras)
            
            html_str = f"""  
            <h3 style="text-align: center; color: green;">{variable_output}</h3>
            """
            st.markdown(html_str, unsafe_allow_html=True)
        
            st.pyplot(fig)

def main():
    st.title("AI Breast Cancer Prognosis Tool")

    with st.sidebar:
        selected = option_menu(
            menu_title="",
            options=["Prediction", "ROC Curve", "Contact"],
            default_index=0,
        )

    if selected == "Prediction":
        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">CNN</h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        c5, c6 = st.columns(2)
        c7, c8 = st.columns(2)
        c9, c10 = st.columns(2)
        

        age_at_diagnosis = c1.text_input("Age At Diagnosis", "")
        overall_survival_months = c2.text_input("Overall Survival Months","")
        lymph_nodes_examined_positive = c3.text_input("Positive Lymph Nodes","")
        tumor_size = c4.text_input("Tumor Size", "")
        tumor_stage = c5.text_input("Tumor Stage", "")
        brca1 = c6.text_input("brca1", "")
        brca2 = c7.text_input("brca2", "")
        tp53 = c8.text_input("tp53","")
        pten = c9.text_input("pten","")
        egfr = c10.text_input("egfr", "")

        living_html = """
        <div style="background-color:#F08080;padding:10px">
        <h2 style="color:white;text-align:center;">High Risk</h2>
        </div>
        """

        death_html = """
        <div style="background-color:#F4D03F ;padding:10px">
        <h3 style="color:white;text-align:center;">Low Risk</h3>
        </div>
        """

        s = f"""
        <style>
        div.stButton > button:first-child {{ background-color: #04AA6D;color: white; padding: 12px 20px;border: none;border-radius: 4px;cursor: pointer; }}
        <style>
        """
        st.markdown(s, unsafe_allow_html=True)
        if st.button("Predict"):
            output=predict_survival(age_at_diagnosis, overall_survival_months,lymph_nodes_examined_positive, tumor_size, tumor_stage, brca1, brca2, tp53,pten, egfr)
        
            st.success('The probability of survival is {}'.format(output))

            if output > .5:
                st.markdown(death_html, unsafe_allow_html=True)
            else:
                st.markdown(living_html, unsafe_allow_html=True)

    if selected == "Contact":
        
        contact_form = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Contact</h2>
        </div>
        <form action="https://formsubmit.co/jijon000@mysbisd.org" method = "POST">
            <input type="text" name = "name" placeholder="Your name" required>
            <input type= "email" name = "email" placeholder= "Your email" required>
            <textarea name = "message" placeholder ="Feedback"></textarea>
            <button type= "submit">Send</buttom>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)
    
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")

    if selected == "ROC Curve":

        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">ROC Curve</h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        draw_plot()



if __name__ == '__main__':
    main()
