##USE PYTHON3 HERE COZ TRAIN_TEST_SPLIT HAS DEPRECIATED IN PYTHON2

from sklearn.datasets import load_iris  #load iris is a func
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

iris=load_iris() 

x=iris.data #matric
y=iris.target #vector

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=4) #random state split data in same way


k_range=range(1,26)
scores=[]
for i in k_range:
	knn=KNeighborsClassifier(n_neighbors=i) # n_neighbors tells values of k, knn is object
	knn.fit(x_train,y_train)
	result=knn.predict(x_test)
	scores.append(metrics.accuracy_score(y_test,result))
	


print(scores)
print("This graph shows how knn algo modifies with value of k")
plt.plot(k_range,scores)
plt.xlabel("Value of K in KNN")
plt.ylabel("Testing Accuracy")
plt.show()






