'''Here we are using 10 fold cross validation with varing value of k nearest neighbors in knn and anaylsing them through graph'''

from sklearn.datasets import load_iris  #load iris is a func
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt

iris=load_iris() 


x=iris.data #matric
y=iris.target #vector

k_range=range(1,31)
k_score=[]
for i in k_range:
	knn=KNeighborsClassifier(n_neighbors=i) 
	scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')
	k_score.append(scores.mean())

print(k_score)
print("This graph shows how cross validation in knn algo modifies with value of k")
plt.plot(k_range,k_score)
plt.xlabel("Value of K in KNN")
plt.ylabel("Testing Accuracy ")
plt.show()


 


