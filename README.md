# EX6 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize Parameters: Randomly initialize the weights and bias.

2.Compute Prediction: Apply the sigmoid function on the linear combination of inputs and weights.

3.Calculate Loss: Compute the binary cross-entropy loss between predicted and actual values.

4.Compute Gradients: Calculate the gradients of weights and bias using partial derivatives of the loss function.

5.Update Parameters: Adjust the weights and bias by subtracting the learning rate multiplied by the respective gradients.

## Program:

Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: SARANYA S.

RegisterNumber: 212223220101


```
import pandas as pd
import numpy as np
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["gender"].astype('category')
dataset.dtypes
#labelling the columbs
dataset["gender"] = dataset["gender"].astype('category').cat.codes
dataset["ssc_b"] = dataset["ssc_b"].astype('category').cat.codes
dataset["hsc_b"] = dataset["hsc_b"].astype('category').cat.codes
dataset["degree_t"] = dataset["degree_t"].astype('category').cat.codes
dataset["workex"] = dataset["workex"].astype('category').cat.codes
dataset["specialisation"] = dataset["specialisation"].astype('category').cat.codes
dataset["status"] = dataset["status"].astype('category').cat.codes
dataset["hsc_s"] = dataset["gender"].astype('category').cat.codes
#display dataset
dataset
#selecting the features and labels
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#display dependent variables
y
#Intialize the model parameters
theta = np.random.randn(x.shape[1])
a=y
#define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#define the loss function
def loss(theta,x,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h) + (1-y) * np.log(1-h))
#define the gradient descent algorithm
#defint the gradient descent alogrithm

def gradient_descent(theta,x,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient = x.T.dot(h -y )/m
        theta-=alpha*gradient
    return theta
#train the model

theta = gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
#make predictions.
def predict(theta,x):
    h =sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5,1,0)
    return y_pred
y_pred = predict(theta,x)

#evaluate the model
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:",accuracy)
print(y_pred)
print(a)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
```

## Output:
ACCURACY,PREDICTED AND PREDICTED VALUE

![image](https://github.com/user-attachments/assets/011542db-7e20-48db-b635-f945fde3afdc)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

