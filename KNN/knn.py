from sklearn.datasets import load_iris  #load iris is a func
from sklearn.neighbors import KNeighborsClassifier
iris=load_iris() 
print(type(iris))

print (iris.data)

x=iris.data #matric
y=iris.target #vector
'''
print x.shape
print y.shape
'''
knn=KNeighborsClassifier(n_neighbors=1) # n_neighbors tells values of k, knn is object
knn.fit(x,y)
#print(knn.predict([3,5,4,2])) #returns an object

x_new=[[3,5,4,2]]
print(knn.predict(x_new))  #here 0,1,2 are targets, they have target_names associated with them