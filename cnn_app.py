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
            
            variable_output = 'AUC (Area Under Curve) = {:.3f}'.format(auc_keras)
            
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
        <h2 style="color:white;text-align:center;">Deep Learning Model</h2>
        <h5 style="color:white;text-align:center;">This model utilizes CNN (Convolutional Neural Networks) for breast cancer prognosis.</h5>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        c5, c6 = st.columns(2)
        c7, c8 = st.columns(2)
        c9, c10 = st.columns(2)
        

        c1, c2 = st.columns(2)
        age_at_diagnosis = c1.text_input("Age at Diagnosis (Numeric only)","85", key="age")
        overall_survival_months = c2.text_input("Overall Survival Months (Numeric only)", "15", key="month")
        #st.write(age_at_diagnosis, overall_survival_months)

        c3, c4 = st.columns(2)
        lymph_nodes_examined_positive = c3.text_input("Positive Lymph Nodes (Numeric only)", "10", key="lymph")
        tumor_size = c4.text_input("Tumor Size in milimeters (Numeric only)", "22", key="size")
        #st.write(lymph_nodes_examined_positive, tumor_size)

        c5, c6 = st.columns(2)
        tumor_stage = c5.text_input("Tumor Stage (0, 1, 2, 3, 4)","4",key="stage")
        brca1 = c6.text_input("BRCA1 (Postive or negative number)","-0.5", key="brca1")
        #st.write(tumor_stage, brca1)

        c7, c8 = st.columns(2)
        brca2 = c7.text_input("BRCA2 (Postive or negative number)","-0.4", key="brca2")
        tp53 = c8.text_input("TP53 (Postive or negative number)","-0.7", key="tp53")
        #st.write(brca2, tp53)

        c9, c10 = st.columns(2)
        pten = c9.text_input("pten (Postive or negative number)","-0.67", key="pten")
        egfr = c10.text_input("EGFR (Postive or negative number)","-0.6", key="egfr")
        #st.write(pten, egfr)

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
        
        help_html = """
        <div style="background-color:#025246 ;padding:20px">
        <h5 style="color:white;;text-align:center;">What do these numbers mean?</h5>
        <p style="color:white">The value for Positive Lymph Nodes represent the number of lymph nodes that have tested positive for cancer.</p>
        <p style="color:white">The values for BRCA1, BRCA2, TP53, PTEN, and EGFR represent changes in mRNA level of the respective genes in a breast tumor relative to healthy tissue, based on RNA-sequencing results. These values are represented in log 2.</p>
        </div>
        """
        st.markdown(help_html, unsafe_allow_html=True)
    if selected == "Contact":
        
        contact_form = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Contact</h2>
        </div>
        <form action="https://formsubmit.co/jijon000@mysbisd.org" method = "POST">
            <input type="text" name = "name" placeholder="Your name" required>
            <input type= "email" name = "email" placeholder= "Your email" required>
            <textarea name = "message" placeholder ="Feedback"></textarea>
            <button type= "submit">Submit</buttom>
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

        help_html = """
        <div style="background-color:#025246 ;padding:20px">
        <h5 style="color:white;;text-align:center;">What does this mean?</h5>
        <p style="color:white">An ROC curve plots the true positive rate against the false positive rate for a model. Finding the area of an ROC curve, or AUC, is one way we can determine a model's predicting power.</p>
        <p style="color:white">Our model's ROC curve, represented by the blue line, has an area of 0.902, indicating a powerful predicting tool for breast cancer prognosis.</p>
        </div>
        """
        st.markdown(help_html, unsafe_allow_html=True)




if __name__ == '__main__':
    main()
