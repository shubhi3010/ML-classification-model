from sklearn.datasets import load_iris  #load iris is a func
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
iris=load_iris() 

x=iris.data #matric
y=iris.target #vector

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=4) #random state split data in same way
logreg=LogisticRegression() 

logreg.fit(x_train,y_train)


result=logreg.predict(x_test)


print(metrics.accuracy_score(y_test,result)) 
#here y is the original classification value and result is the predicted value. SO we are calculating the accuracy
