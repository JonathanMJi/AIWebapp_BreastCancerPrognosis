import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from streamlit_option_menu import option_menu
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import base64

model = tf.keras.models.load_model('BreastCancer_DL.h5')

def predict_survival(age_at_diagnosis, overall_survival_months, lymph_nodes_examined_positive, tumor_size, tumor_stage, brca1, brca2, tp53, pten, egfr):
    df = pd.read_csv('METABRIC_RNA_Mutation_Signature3_Preprocessed.csv', delimiter=',')
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
            df = pd.read_csv('METABRIC_RNA_Mutation_Signature3_Preprocessed.csv', delimiter=',')
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
    st.title("AI Breast Cancer Prognosis App")
    with st.sidebar:
        selected = option_menu(
            menu_title="",
            options=["Prediction",  "ROC Curve", "Dataset", "Contact"],
            default_index=0,
            styles={
                "container":{"padding" : "0!important", "background-color" : "#fafafa"},
                "icon":{"color":"orange","font-size":"15px"},
                "nav-link":{"font-size":"15px","text-align":"left","margin":"0px","--hover-color":"#BADBD2"},
                "nav-link-selected":{"background-color":"#025246"},
            }
        )

    
    if selected == "Prediction":
        
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

        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Deep Learning Model</h2>
        <h5 style="color:white;text-align:center;">This model utilizes CNN (Convolutional Neural Networks) for breast cancer prognosis prediction.</h5>
        </div>
        """

        st.markdown(html_temp, unsafe_allow_html=True)

        st.text("")    
        
        df1=pd.read_csv("PatientSample_1.csv")
        csv1 = df1.to_csv(index=False)
        b64_a = base64.b64encode(csv1.encode()).decode()  # some strings <-> bytes conversions necessary here
        href1 = f'<a href="data:file/csv;base64,{b64_a}" download="PatientSample_1.csv">Download Sample Patient File (High Risk)</a>'
        st.markdown(href1, unsafe_allow_html=True)

        df2=pd.read_csv("PatientSample_2.csv")
        csv2 = df2.to_csv(index=False)
        b64_b = base64.b64encode(csv2.encode()).decode()  # some strings <-> bytes conversions necessary here
        href2 = f'<a href="data:file/csv;base64,{b64_b}" download="PatientSample_2.csv">Download Sample Patient File (Low Risk)</a>'
        st.markdown(href2, unsafe_allow_html=True)
            
        uploaded_file = st.file_uploader("This app only accepts .csv files.", type=["csv"])
            
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)

            s = f"""
            <style>
            div.stButton > button:first-child {{ background-color: #04AA6D;color: white; padding: 12px 20px;border: none;border-radius: 4px;cursor: pointer; }}
            <style>
            """
            st.markdown(s, unsafe_allow_html=True)
            if st.button("Predict"):

                #X=df.drop( ['death_from_cancer', 'overall_survival'], axis=1)
                TestData = np.asarray(df).astype(np.float32)
                prediction = model.predict(TestData)
                pred = '{0:.{1}f}'.format(prediction[0][0],2)
                output = float(pred)            
                st.success('The probability of survival is {}'.format(output))

                if output > .5:
                    st.markdown(death_html, unsafe_allow_html=True)
                else:
                    st.markdown(living_html, unsafe_allow_html=True)    

        help_html = """
        <div style="background-color:#025246 ;padding:20px">
        <h5 style="color:white;;text-align:center;">How do I use this?</h5>
        <p style="color:white">This option takes in a patient's entire clinical and genetic data as a .csv file.</p>
        <p style="color:white">Sample patient data have been provided to serve as examples for dataset formatting and uploading.</p>
        </div>
            """   
        st.markdown(help_html, unsafe_allow_html=True)            


        
    if selected == "Contact":
        
        s = f"""
            <style>
            div.stButton > button:first-child {{ background-color: #04AA6D;color: white; padding: 12px 20px;border: none;border-radius: 4px;cursor: pointer; }}
            <style>
            """
        st.markdown(s, unsafe_allow_html=True)

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

    if selected == "Dataset":

        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Dataset</h2>
        <p style= "color:white;text-align:center">The dataset used to train this CNN model was accessed from <a href="https://www.cbioportal.org/study/summary?id=brca_metabric" target="_blank">cBioportal</a>.</p>

        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        df=pd.read_csv("METABRIC_RNA_Mutation_Signature3.csv")
        st.markdown("", unsafe_allow_html=True)
        st.dataframe(df,1000,200)

        


if __name__ == '__main__':
    main()