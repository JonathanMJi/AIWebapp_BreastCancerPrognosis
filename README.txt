Web App Design 

A Convolutional Neural Networks algorithm (CNN) using Tensorflow and Keras, a novel deep learning approach, is identified and adopted in this study for predicting the survival rate of breast cancer patients. The dataset used for training and testing the machine learning models include both clinical and genomic data of breast cancer sample and were obtained from METABRIC database from cBioPortal. The study reveals that the mRNA levels of these genes correlate well with breast cancer patients’ prognosis and histopathology and have great potential to predict potential survival. The deep learning CNN model produces a 21 gene signature and offers a statistically significant predictive value with an AUC (Area Under Curve) score of ROC.  These 21 genes include MYC, HSD17B11, SPRY2, SYNE1, LAMA2, ABCB1, ABCC1, RPS6, MMP3, MMP7,PDGFRA,KIT, MAPK14, CASP6, CASP8, STAT5A, E2F2, FOLR2, FGF1, RASSF1 and MMP11. 

The CNN model shows convincing result and has high accuracy and is ready to be used as a systematic framework for real time prediction by end users. To make it user friendly, I designed and developed a web application for the predicting breast cancer survival rate using streamlit,  an open-source python framework for building web apps for Machine Learning and Data Science.

1. Create a machine learning model and train it
2. Take the model and dump into h5 (CNN model) 
3. Build Front end web application using streamlit with side menus
4. Take the uploaded patient file from web application, supply to the h5 file, which contains the model, make the prediction, and display the output on the user interface

This web app utilizes a panel including the changes in the mRNA levels (compared with healthy breast tissue) of entire 21 genes and 31 pathological features of breast cancers for the prediction. The user can import an entire panel of patient data and make predictions. The sample panel data can be downloaded from the web app.  

The web app shows the plot of AUC (Area Under Curve) score of ROC = 0.900, dataset with the 21 gene-set signature and a contact whihc allows user to contact me

This web app demonstrates great potential for predicting breast cancer patient survival. The algorithm for this app can also be utilized to model patient survival in other human cancers. 

The link of the web app is: https://jonathanmji-cornellresearchaiwebapp-cnn-app-0i2prn.streamlitapp.com/
