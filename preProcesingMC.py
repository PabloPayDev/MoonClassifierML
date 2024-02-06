import cv2
import glob
import numpy as np
import pandas as pd

image_listmoon=[]
image_listnomoon=[]
matrix_data_of_images=[]
y = []

print("Comenzando Pre-Procesamiento")
#leendo las imagenes de la carpeta 'moon'
for filename in glob.glob('moon/*.jpg'):
	im=cv2.imread(filename)
	im=cv2.resize(im,(32,32))
	image_listmoon.append(im)
	y.append(1)

#leendo las imagenes de la carpeta 'notmoon'
for filename in glob.glob('notmoon/*.*'):
	im=cv2.imread(filename)
	im=cv2.resize(im,(32,32))
	image_listnomoon.append(im)
	y.append(0)

#cambiando a blanco y negro y llenando la matriz de datos final
for imagen in image_listmoon:
	gray=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
	gray=(255-gray)/255
	matrix_data_of_images.append(gray.flatten())
for imagen in image_listnomoon:
	gray=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
	gray=(255-gray)/255
	matrix_data_of_images.append(gray.flatten())

#convertimos la matriz en un array	
matrix_data_of_images=np.array(matrix_data_of_images)
print(matrix_data_of_images.shape)
matrix_data_of_images=matrix_data_of_images.flatten()
matrix_data_of_images=np.array(matrix_data_of_images)
#guardamos los datos en un csv
matrix_data_of_images=pd.DataFrame(matrix_data_of_images)
print(matrix_data_of_images)
matrix_data_of_images.to_csv('MoonImgsXData.csv', sep=',', encoding='utf-8')

#guardar el vector de soluciones "y"
y = np.array(y)
y = y.flatten()
y = np.array(y)
y =	pd.DataFrame(y)
print(y)
y.to_csv('MoonImgsYData.csv', sep=',', encoding='utf-8')