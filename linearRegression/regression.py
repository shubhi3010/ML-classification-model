from sklearn.datasets import load_iris  #load iris is a func
from sklearn.linear_model import LogisticRegression
iris=load_iris() 


print (iris.data)

x=iris.data #matric
y=iris.target #vector

logreg=LogisticRegression() 

logreg.fit(x,y)

print(logreg.predict(x)) #pass the entire matrix x
result=logreg.predict(x)
print(len(result))
