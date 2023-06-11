import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import StandardScaler as sc
data=load_breast_cancer()
from sklearn.decomposition import PCA
td=pd.DataFrame(data.data,columns=data.feature_names)
from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(td,data.target,test_size=0.3,random_state=1)
def sigmoid(x):
    return 1/(1+np.exp(-x))
## using pearson correlation
import seaborn as sns
plt.figure(figsize=(30,30))
cor=xtr.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.CMRmap_r)
## with the following function we can select highly correlated features
## it will remove the first features that are correlated with anything other feature 
def correlation(dataset,threshold):
    col_corr=set()#set of all names of correlated columns
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]>threshold):
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features=correlation(xtr,0.7)
len(set(corr_features))

xtr.drop(corr_features,axis=1,inplace=True)
xte.drop(corr_features,axis=1,inplace=True)

class LogisticRegression():

    def __init__(self, lr=0.001, itr=10000):
        self.lr = lr
        self.itr = itr
        self.weights = None
        self.bias = None
#         print("Learning Rate :",self.lr,"Number of Iterations",self.itr,"Weights :",self.weights)

    def fit(self, X, y):
        n, feat = X.shape
        self.weights = np.zeros(feat)
        self.bias = 0
#         print("Weight Updates :",self.weights)

        for _ in range(self.itr):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n) * np.dot(X.T, (predictions - y))
            db = (1/n) * np.sum(predictions-y)
            
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
#             print("Bias update :",self.bias)
            
#             print("Weight Updates :",self.weights,"Bias Update :",self.bias)
        return self.weights,self.bias
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred


clflogr=LogisticRegression(lr=0.005)
k,m=clflogr.fit(xtr,ytr)
y_pred=clflogr.predict(xte)

def accuracy(y_pred, yte):
    return np.sum(y_pred==yte)/len(yte)
acc = accuracy(y_pred, yte)
print(acc)

########################## saving the trained model
import pickle
filename='trained_model.sav'
pickle.dump(clflogr,open(filename,'wb'))
#wb means write binary

#loading the saved model
loaded_model=pickle.load(open('trained_model.sav','rb'))#this is used to load the model 
# rb means reading the binary file

def load_lottieurl(url:str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()



def main():
    ####giving a title
    st.title('Breast Cancer Prediction Model!')
    lottie_url=load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t3cfpwqj.json")
    st_lottie(
        lottie_url
    )
    ##getting the input data from the user
    ## radius	mean texture	mean perimeter	mean area	mean smoothness	mean compactness
    mean_radius=st.text_input('Enter mean radius')
    mean_texture=st.text_input('Enter mean texture')
    mean_smoothness=st.text_input('Enter mean smoothness')
    mean_compactness=st.text_input('Enter mean compactness')
    mean_symmetry=st.text_input('Enter mean symmetry')
    mean_fractal_dimension=st.text_input('Enter fractal dimension')
    texture_error=st.text_input('Enter texture error')
    smoothness_error=st.text_input('Enter smoothness error')
    symmetry_error=st.text_input('Enter symmetry error')
    worst_symmetry=st.text_input('Enter worst error')

    input=[mean_radius,mean_texture,mean_smoothness,mean_compactness,mean_symmetry,mean_fractal_dimension,texture_error,smoothness_error,symmetry_error,worst_symmetry]

    ######## code for prediction
    diagnosis=' '

    ######## creating a button for prediction

    if st.button('Test Result'):
         input_data_as_numpy_array=np.asarray(input,dtype=float)
         input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
         #input_arrays=sc.transform(input_data_reshaped)
         prediction=loaded_model.predict(input_data_reshaped)
         print(prediction)
         if(prediction[0]==0):
            diagnosis='Malignant'
         else:
            diagnosis='Benign'
    
    st.success(diagnosis)


if __name__=='__main__':
    main()

