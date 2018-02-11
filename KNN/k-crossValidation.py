from sklearn.datasets import load_iris  #load iris is a func
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
iris=load_iris() 


x=iris.data #matric
y=iris.target #vector

knn=KNeighborsClassifier(n_neighbors=5) 
scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy') 
'''Here cv means divide datasets in 10 folds. 
 We have total 150 records so divide it in 10 sets. One fold is
treated as testing set at a time and rest as training set.
So scores has total 10 observations depicting the accuracy of each fold.

''' 
print(scores)
print(scores.mean()) # calculate mean of accuracy of 10 sets.
