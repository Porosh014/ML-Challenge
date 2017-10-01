import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.metrics import accuracy_score

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE


def pca_DimensionReduction(training_set): #A method for PCA dimensionality reduction
    pca = PCA(n_components=None)
    training_set = pca.fit_transform(training_set)
    
    
    explained_variance = pca.explained_variance_ratio_
    
    avg_variance = np.average(explained_variance)
    
    indexes = (explained_variance >= avg_variance)
    
    indexes = list(np.where(indexes == False)) 
    
    return indexes


dataset = pd.read_csv("sample.csv", header = None) #reading csv into Dataframe format

colm2Exclude = [295]

target_variable = dataset.iloc[:,295].values #Target variable initialize

y_labelEncoder = LabelEncoder()
target_variable = y_labelEncoder.fit_transform(target_variable) #Encoding labels


label_binarizer = LabelBinarizer()
target_variable = label_binarizer.fit_transform(target_variable)

dataset = dataset.iloc[:, dataset.columns.difference(dataset.columns[colm2Exclude])] #Excluding target variable
training_set = dataset.values

Train_sc = MinMaxScaler()
training_set = Train_sc.fit_transform(training_set) #Apply Normalization over all the features


ExcludeIndexList = pca_DimensionReduction(training_set) #Call method to perform dimensionality reduction and get excluded column indexes
Extracted_training_set = dataset.iloc[:, dataset.columns.difference(dataset.columns[ExcludeIndexList])].values

Extract_sc = MinMaxScaler()
Extracted_training_set = Extract_sc.fit_transform(Extracted_training_set) #Normalizing all the features
Number_of_Columns = Extracted_training_set.shape[1] #Number of dimensions after applying pca



#Splitting the dataset into training and test samples
X_train,X_test,y_train,y_test = train_test_split(Extracted_training_set,target_variable,
                                                 test_size=0.2,random_state=12)

y_train = y_train.argmax(1) # Converting labels to some value x 1 matrix
y_test = y_test.argmax(1)   # Converting labels to some value x 1 matrix
                        
                                      
def buildClassifierANN(): #Method for creating ANN
    #Building Artificial Neural Network                                      
    classifier = Sequential()  
    
    #Adding hidden layer and input layer
    classifier.add(Dense(units = int((Number_of_Columns+5)/2), 
                         kernel_initializer ='uniform',activation='relu',input_dim=Number_of_Columns ))
    #Adding second hidden layer
    classifier.add(Dense(units = int((Number_of_Columns+5)/2), 
                         kernel_initializer ='uniform',activation='relu'))  
    
    #Adding third hidden layer
    classifier.add(Dense(units = int((Number_of_Columns+5)/2), 
                         kernel_initializer ='uniform',activation='relu'))                                
        
    #Adding output layer (As we have total 5 classes so output_dim=5)
    classifier.add(Dense(units=5,kernel_initializer='uniform',activation='softmax'))
    
    #Compiling ANN
    classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    
    
    #classifier.fit(X_train,y_train,batch_size=16,nb_epoch=100)
    
    return classifier                                      


y_label_binarizer = LabelBinarizer()
y_train = y_label_binarizer.fit_transform(y_train) #Binarizing the labels again to feed it to cross validation function
                                     
classifier = KerasClassifier(build_fn= buildClassifierANN, batch_size = 64, nb_epoch = 100)
accuracies = cross_val_score(estimator=classifier, X= X_train, y= y_train, cv=15)

classifier.fit(X_train, y_train, batch_size = 64, nb_epoch = 100)    #Fitting oversampled data to classifier                                
y_test_pred = classifier.predict(X_test)    #Predicting labels for test data



conf_mat = confusion_matrix(y_test,y_test_pred) #Generating confusion matrix
pre_score = precision_score(y_test, y_test_pred, average='weighted')
recall_score = recall_score(y_test, y_test_pred, average='weighted')
                           
print("Average accuracy after applying k-fold cross validation is"+ str(accuracies.mean()))
print("Average precision is "+str(pre_score))
print("Average recall is "+str(recall_score))
print(confusion_matrix)