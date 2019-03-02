import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import PIL
from PIL import Image
import warnings

def createAnalysis(classifier,x_train,y_train,tx,ty,random, type,eje):
    classifier.fit(x_train,y_train)
    print('Precisión SGD:',classifier.score(tx,ty))
    print('Cross Validation:', cross_val_score(classifier,tx,ty,cv=3, scoring="accuracy"))
    if type=='SGD':
        y_scores=cross_val_predict(classifier,x_train,y_train,cv=3,method="decision_function")
    else:
        y_scores=cross_val_predict(classifier,x_train,y_train,cv=3,method="predict_proba")
    print('Valor real', ty[random])
    print('La computadora predijo: ',classifier.predict(tx[random].reshape(1, -1)))
    print("Matriz de confusión:\n%s" % metrics.confusion_matrix(ty, classifier.predict(tx)))
    print("Predicción dato de usuario",classifier.predict(eje))
    plt.imshow(digits.images[1497+rand],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.show()
def imageprepare(argv):
    im=Image.open(argv).convert('L')
    basewidth=8
    wpercent=(basewidth/float(im.size[0]))
    hsize=int((float(im.size[1])*float(wpercent)))
    img=im.resize((basewidth,hsize),PIL.Image.ANTIALIAS)
    tv=list(img.getdata())
    tva=[(255-m)*1.0/255.0 for m in tv]
    return tva
warnings.filterwarnings("ignore")
#carga los datos en digits
digits = datasets.load_digits()
#en x se guardan los arreglos y en y los valores reales de los datos a entrenar
x,y = digits.data[:-300],digits.target[:-300]
#en tx se guardan los arreglos y en ty los valores reales de los datos con los cuales hacer el testing
tx,ty = digits.data[-300:],digits.target[-300:]

#random de ejemplo para ver cómo lo predicen ambos clasificadores.
rand=random.randint(0,len(tx))


imagen=[imageprepare('./Imagen.png')]
newArr=[[0 for d in range(8)] for y in range(8)]
k=0
for i in range(8):
    for j in range(8):
        newArr[i][j]=imagen[0][k]
        k=k+1
print(newArr)
print(digits.data[0])
plt.imshow(newArr,interpolation="nearest")
plt.show()
pre=np.array(newArr)
nuevo=pre.flatten().reshape(1,-1)
print(nuevo)
#Clasificador SGDC
clf = SGDClassifier()
createAnalysis(clf,x,y,tx,ty,rand,'SGD',nuevo)
rfc=RandomForestClassifier(n_jobs=-1, n_estimators=10)
createAnalysis(rfc,x,y,tx,ty,rand,'RF',nuevo)
from roc import ROC_Curve
ROC_Curve(x,y)
