import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv(r"C:\Users\merts\OneDrive\Belgeler\GitHub\Deep-Learning\Churn_Modelling.csv")

X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values
#Ülke ve cinsiyetleri encode etmeliyiz şimdi çünkü YSA sadece 0-1 değer alır

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])

le=preprocessing.LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe=ColumnTransformer([("ohe",OneHotEncoder(dtype="float"),[1])],
                      remainder="passthrough"
                      )
X=ohe.fit_transform(X)
X=X[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#yapay sinir ağları oluşturucaz
import keras
from keras.models import Sequential
from keras.layers import Dense,Input
classifier = Sequential()

# İlk katman, giriş katman olarak Input katmanı ekleniyor
classifier.add(Input(shape=(11,)))

#gizli katmanlar
classifier.add(Dense(6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(6, kernel_initializer="uniform", activation="relu"))

#çıkış katmanı
classifier.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

# tahmini yaptıracağımız değerler 1 ve 0 dan oluştuğundan binary entropy kullanıcaz
classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"]) 

#epoch(çağ,tekrar) kaç kere çalışacağıdır
#loss'u düşük tutup,accuracy yüksek tutmaya çalışıyoruz
classifier.fit(X_train,y_train, epochs=50)

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

print(cm)




