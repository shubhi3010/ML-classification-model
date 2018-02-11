##USE PYTHON3 HERE COZ CROSS VALIDATION HAS DEPRECIATED IN PYTHON2

from sklearn.datasets import load_iris  
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

iris=load_iris() 



x=iris.data #matric
y=iris.target #vector

logreg=LogisticRegression() 
scores=cross_val_score(logreg,x,y,cv=10,scoring='accuracy')
print(scores)
print(scores.mean())

