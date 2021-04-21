# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:14:16 2020

@author: Masomeh ALi Heydarloo
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

##########Read dataset
path='F:/Python/Homework7/blocks.csv'
ds=pd.read_csv(path)

#########get the information
ds.info()
##Desc: no missing value
ds.head()
ds.shape  # (676, 11)
###################
import matplotlib.pyplot as plt
plt.hist('block',data = ds, color='g')
plt.xlabel('block category')
plt.ylabel('Quantity')
plt.title('Class (Y) Distribution')
###################
sns.pairplot(df,hue='Kyphosis',palette='Set1')
#########Convert non-numeric to numeric 'block' feature
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(ds['block'])
ds['block'] = enc.transform(ds['block'])

########Define X,y
y = ds['block']
X = ds.drop('block',axis=1)
#######Scale X
from sklearn.preprocessing import scale
X=scale(X)


##################################KNN Model with crossvalidation
nneighbors=np.arange(1,10) # the number of neighbour is 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#Cross validation to find the best number of neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

cv=10 # the number of fold

cvscores=np.empty((len(nneighbors),2)) 

counter=-1

for k in nneighbors:
 counter=counter+1    
 Knno=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
 cvscores[counter,:]=np.array([k,np.mean(cross_val_score(Knno, X_train, y_train, cv=cv))])


cvscores[np.argmax(cvscores[:,1]),:] #[1.        , 0.8931694], k=1 is the best
# print("The optimal number of neighbors is {}".format(optimal_k))


############################# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

meanscorelda=np.empty((1000,2))  # Will store the average test accuracy
meanscoreqda=np.empty((1000,2)) # Will store the average test accuracy
meanscoregnb=np.empty((1000,2)) # Will store the average test accuracy
meanscorelr=np.empty((1000,2)) 
meanscoreknn=np.empty((1000,2))

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
gnb = GaussianNB()
lr=LogisticRegression(penalty='none',solver='newton-cg',max_iter=2000)
Knno=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)

counter=-1

#Repeat 1000 times to compute the mean of accuracy score
for i in range(999):
 counter=counter+1 
 #split test and train data
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

 #Fit the model

 lda.fit(X_train,y_train)
 #Store the mean of accuracy score
 meanscorelda[counter,:]=np.array([i,np.mean(lda.score(X_test,y_test))])
 
 #Fit the model
 
 qda.fit(X_train,y_train)
 #Store the mean of accuracy score
 meanscoreqda[counter,:]=np.array([i,np.mean(qda.score(X_test,y_test))])


 #Fit the model 

 gnb.fit(X_train,y_train)
 #Store the mean of accuracy score
 meanscoregnb[counter,:]=np.array([i,np.mean(gnb.score(X_test,y_test))])

 #Fit the model 

 lr.fit(X_train,y_train)
  #Store the mean of accuracy score
 meanscorelr[counter,:]=np.array([i,np.mean(lr.score(X_test,y_test))])
  
 
 #Fit the model 

 Knno.fit(X_train,y_train)
 meanscoreknn[counter,:]=np.array([i,np.mean(Knno.score(X_test,y_test))])

# evaluate accuracy
# print("accuracy: {}".format(100*accuracy_score(y_test, pred)))

meanscorelr[np.argmax(meanscorelr[:,1]),:] # array([0.        , 0.92647059])
#Max value of the average test accuracy
meanscorelda[np.argmax(meanscorelda[:,1]),:] # array([889.        ,0.92647059])
#Max value of the average test accuracy
meanscoreqda[np.argmax(meanscoreqda[:,1]),:] # array([0.        , 0.94117647])
#Max value of the average test accuracy
meanscoregnb[np.argmax(meanscoregnb[:,1]),:] # array([0.        , 0.94117647])

meanscoreknn[np.argmax(meanscoreknn[:,1]),:] # array([154.        ,   0.98529412])



############confusion Matrix
confusion_matrix(y_test,lda.predict(X_test))


############confusion Matrix of QDA
confusion_matrix(y_test,qda.predict(X_test))


############confusion Matrix
confusion_matrix(y_test,gnb.predict(X_test))

############confusion Matrix
confusion_matrix(y_test,lr.predict(X_test))
# 0 numbers in class 0: 522+233=755
# 1 numbers in class 1: 201+496=697

############confusion Matrix
confusion_matrix(y_test,Knno.predict(X_test))

####################################################
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train)

nneighbors=np.arange(1,10) # the number of neighbour is 10
comp=np.arange(10,100,10)#[10,20,....,90] c=10 means we select 10 pcscores
#cvscorespca is an empty matrix with len(comp)*len(nneighbors) as rows number and 3 columns
#column 0 has pcscore number ,column 1 has k value(0 to 10) and column2 has mean
cvscorespca=np.empty((len(comp)*len(nneighbors),3))

counter=-1

for c in comp:
    X_train_pca = pca.transform(X_train)[:,:c]
    for k in nneighbors:
     counter=counter+1   
     Knnpca=KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
     cvscorespca[counter,:]=np.array([c,k,np.mean(cross_val_score(Knnpca, X_train_pca, y_train, cv=cv))])

cvscorespca[np.argmax(cvscorespca[:,2]),:]   #[10.       ,  1.       ,  0.90625683]
# 
##########################################################
#After cross validation, you can compare your methods
#we get k=1, c=10 are the best, now check test accuracy

meanscore=np.empty((1000,3)) 
counter=-1

for i in range(999):
 counter=counter+1 
 #split test and train data
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
 #Fit the model 
 Knno=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
 Knnpca=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
 X_train_pca = pca.transform(X_train)[:,:10]
 X_test_pca = pca.transform(X_test)[:,:10]
 Knno.fit(X_train,y_train)
 Knnpca.fit(X_train_pca,y_train)
 meanscore[counter,:]=np.array([i,np.mean(Knno.score(X_test,y_test)),np.mean(Knnpca.score(X_test,y_test))])

meanscore[np.argmax(meanscore[:,1]),:]            
meanscore[np.argmax(meanscore[:,2]),:]              


  # array([372.        ,   0.98529412,   0.44117647])#without PCA
   # array([380.        ,   0.85294118,   0.57352941])#with PCA









####################################

########Error rate Plot
import matplotlib.pyplot as plt
error_rate = []
from sklearn.neighbors import KNeighborsClassifier
# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

###############KNN

#4 has min Error rate
Knno=KNeighborsClassifier(n_neighbors=4,metric='minkowski',p=2)
Knno.fit(X_train,y_train)

print('Knn-PCA Prediction Accuracy:',Knno.score(X_test,y_test))#0.925