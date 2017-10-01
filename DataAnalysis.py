import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.cross_validation import train_test_split


def printClassesRation(target_variable): #Method for showing class distribution in a pie chart
    target_variable = np.array(target_variable)
    
    #Frequencies of class labels
    freq_A = list(target_variable).count(0)
    freq_B = list(target_variable).count(1)
    freq_C = list(target_variable).count(2)
    freq_D = list(target_variable).count(3)
    freq_E = list(target_variable).count(4)
    
    #Pie chart using matplotlib
    labels = ['A','B','C','D','E']
    freq = [freq_A, freq_B, freq_C, freq_D, freq_E]
    
    fig1, ax1 = plot.subplots()
    
    ax1.pie(freq, labels = labels, autopct='%1.1f%%')
    
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    

def pca_DimensionReduction(training_set):
    #Applying PCA Over Matrix of features
    pca = PCA(n_components=None)
    
    training_set = pca.fit_transform(training_set)
    
    explained_variance = pca.explained_variance_ratio_ #Explained variance of all columns
    
    avg_variance = np.average(explained_variance) #Taking average of explained variance of all columns
    
    indexes = (explained_variance >= avg_variance) #A boolean matrix, TRUE if explained_variance >= avg_variance
    
    indexes = list(np.where(indexes == False)) #Indexes those could not agree with the constraint
    
    return indexes

def kmeansClustering(dataset, target_variable): #Apply KMeans clustering
    kmeans = KMeans(n_clusters=5, init = 'k-means++',random_state=0)
    
    y = kmeans.fit_predict(dataset)    
    
    for i in range(0,5):
        cluster_index = (y == i)
        
        
        cluster_index = list(np.where(cluster_index == True))
        
        
        determineCluster(cluster_index, target_variable)
    

def determineCluster(indexList, target_variable): #Specifying cluster members and measure homogenity 
    target_variable = target_variable[indexList]
    
    rep_freq = np.count_nonzero(target_variable == np.bincount(target_variable).argmax()) #Number of dominated class members in a cluster
    
    print("Cluster is dominated by class "+str(np.bincount(target_variable).argmax()))
    print("Dominated class percentage "+str(rep_freq/len(target_variable)))
    

dataset = pd.read_csv("sample.csv", header = None) #Import Dataset

colm2Exclude = [295]

target_variable = dataset.iloc[:,295] #Extracting target variable

y_labelEncoder = LabelEncoder()
target_variable = y_labelEncoder.fit_transform(target_variable) #Encoding label with Numeric values

#Excludng target variable and generate Matrix of features                                              
dataset = dataset.iloc[:, dataset.columns.difference(dataset.columns[colm2Exclude])]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(dataset) #Normalizing all feature values

                               
dfT = pd.DataFrame(training_set) #Coverting the normalized features into DataFrame again

correlation_matrix = dfT.corr() #Generating correlation matrix

print(correlation_matrix)

indexes = pca_DimensionReduction(training_set) #This method performs PCA over dataset and returns column indexes to be excluded




#Extracting features from Training set after applying PCA
dataset = dataset.iloc[:, dataset.columns.difference(dataset.columns[indexes])]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
extracted_training_set = sc.fit_transform(dataset)


printClassesRation(target_variable) #Shows distribution of classes in a pie chart



kmeansClustering(extracted_training_set,target_variable)


















