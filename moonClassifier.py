import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

#Leyendo datos
data = pd.read_csv('MoonImgsXData.csv')
X = data.iloc[:,1:1024]
X = np.array(X)
X = X.reshape((422,1024))
print(X)
print(X.shape)

data = pd.read_csv('MoonImgsYData.csv')
y = (data.iloc[:,1]).values.flatten()
y = np.array(y)
print(y)
print(y.shape)

#Inicializacion aleatoria del vector Theta
print("Generando Theta aleatorio")
theta = (np.random.rand(1024) * (2.0/1024)).reshape((1024,1))
print(theta)
print(theta.shape)

#--------------------------------------------------------------------------------------------------------
#Funcion Sigmoide
def sigmoide(Z):
    return 1/(1+np.exp(-Z))

#CostFunction and Gradient
def costFunc(X,Y,Theta):
    J =  -np.sum( Y*np.log(sigmoide(X@Theta)) + (1-Y)*np.log(1-sigmoide(X@Theta)))/(Y.shape[0])
    J_grad = (X.T@(sigmoide(X@Theta) - Y))/(Y.shape[0])
    return J, J_grad

#Actualiced Theta
def act_Theta(Theta,J_grad,alpha):
    return  Theta - alpha*J_grad

#CostFunction and Gradient Descent
def gradientDescent(X,Y,Theta_gd,alpha=0.01,iteraciones=400):
    histJ = []
    histJ_grad= []
    J , J_grad = costFunc(X,Y.reshape(-1,1),Theta_gd)
    histJ.append(J)
    histJ_grad.append(J_grad)
    for i in range(1,iteraciones+1):
        Theta_gd = act_Theta(Theta_gd,J_grad,alpha=alpha)
        J , J_grad = costFunc(X,Y.reshape(-1,1),Theta_gd)
        histJ.append(J)
        histJ_grad.append(J_grad)
        if i%100 ==0:
            print("Función de costo en la iteración ", i, ": ",round(J,6))
    return Theta_gd , histJ, histJ_grad

#Conteo de aciertos
def aciertos(X,Theta_opt,Y): 
    Salida = np.where(sigmoide(X@Theta_opt)>= 0.5, 1, 0)
    return Salida, np.sum(Y.reshape(-1,1) == Salida)/(Y.shape[0])
#-------------------------------------------------------------------------------------------------------------
#Aplicando el GradientDescent
Theta_opt , J_hist , histJ_grad = gradientDescent(X,y,theta, alpha=0.01)

#Graficamos la funcion de coito
plt.plot(J_hist)
plt.show()

#Conteo de Aciertos usando el mismo DataSet
print("Por ultimo la probabilidad clasificando el mismo DataSet con los Theta optimos es:")
Y_hat , count = aciertos(X,Theta_opt,y)
print(count)

#Prueba aleatoria Unitaria
indTest = randint(0,y.shape[0])
print("Data elegida: ",indTest)

xTest = X[indTest]
print("Datos: ",xTest)
yTest = y[indTest]
if(yTest):
    print("Resultado: ","Luna")
else:   
    print("Resultado: ","No luna")

pred = sigmoide(xTest@Theta_opt)
if(pred >= 0.5):
    print("Prediccion:","Luna")
else:
    print("Prediccion:","NO luna")
    
x2img = (xTest*255).reshape(32,32)
plt.imshow( x2img, cmap="gray", vmin=0, vmax=255)
plt.colorbar()
plt.show()