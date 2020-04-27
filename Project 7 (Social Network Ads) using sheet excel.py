'''بسم الله الرحمن الرحيم'''               
 
                   #Project 7 (Social Network Ads) using sheet excel 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv(r"F:\deploma\PYTHON\sample sheets\Social_Network_Ads.csv")                     
x=data.iloc[:,[2,3]]
y=data.iloc[:,4]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test =sc.transform(x_test)

from sklearn.svm import SVC
reg =SVC(kernel ="rbf",random_state=0)
reg.fit(x_train,y_train)
ypred =reg.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)

from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1,x2 =np.meshgrid(np.arange(start = x_set[:,0].min() -1, stop = x_set[:,0].max() +1, step = 0.01),
                  np.arange(start = x_set[:,1].min() -1, stop = x_set[:,1].max() +1, step = 0.01))
plt.contourf(x1, x2, reg.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate (np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0],x_set[y_set == j, 1],
                c = ListedColormap(("red", "green"))(i), label = j)
plt.title("kernel SVM(Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
**************************
weight=int(input("weight in kg : "))
length = int(input( 'length in meter:'))
weight_lbs =float(weight)*2.45
BMI = weight/(length*length)
if BMI>25:
    print (' over weight')
elif BMI< 18.5:
    print ( ' under weight')
elif BMI >18.5 and BMI<24:
    print(' you are in healthy range')