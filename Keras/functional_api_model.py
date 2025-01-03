# -*- coding: utf-8 -*-
"""functional_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1X3W_s-30KFws9wfSm3NCC5KzLzCcoxYb

# Functional API Keras


*  Functional API, çok daha esnek ve karmaşık model yapılarının oluşturulmasına olanak tanır.
* Bu API, katmanları birer fonksiyon gibi tanımlar ve bir katman başka bir katmanın çıktısını alacak şekilde modeldeki bağlantıları tanımlamanıza izin verir.
* Dallanmış, paylaşılan katmanlar, çoklu giriş/çıkışlar gibi karmaşık yapıları rahatça oluşturabilirsiniz.
"""

from sklearn.datasets import fetch_california_housing
housing=fetch_california_housing()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(housing.data,housing.target,random_state=42)

import tensorflow as tf

tf.random.set_seed(42)

"""öncelikle katmanlar oluşturulur sonra bunlarla model kurulur"""

normalizaion_layer=tf.keras.layers.Normalization()
hidden_layer1=tf.keras.layers.Dense(30,activation="relu")
hidden_layer2=tf.keras.layers.Dense(30,activation="relu")
concat_layer=tf.keras.layers.Concatenate()
output_layer=tf.keras.layers.Dense(1)

input_=tf.keras.layers.Input(shape=X_train.shape[1:]) #öznitelikleri alıyoruz
normalized=normalizaion_layer(input_) #normalize ediyor giren öznitelikleri
hidden1=hidden_layer1(normalized)
hidden2=hidden_layer2(hidden1)
concat=concat_layer([normalized,hidden2])
output=output_layer(concat)
model=tf.keras.Model(inputs=[input_],outputs=[output])
model.summary()

optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3)
model.compile(loss="mse",optimizer=optimizer,metrics=["RootMeanSquaredError"])

normalizaion_layer.adapt(X_train) #normalleştirme için eğitim verileri ortalamasını ve standart sapmasını sağlayacak
history=model.fit(X_train,y_train,validation_split=0.2,epochs=20)

mse_test=model.evaluate(X_test,y_test)

import pandas as pd
X_new=X_test[:3]
pd.DataFrame(X_new)

y_pred=model.predict(X_new)
y_pred

pd.DataFrame(y_test[:3])

"""# Birden fazla girdi için functional api"""

input_wide=tf.keras.layers.Input(shape=[5])
input_deep=tf.keras.layers.Input(shape=[6])

norm_layer_wide=tf.keras.layers.Normalization()
norm_layer_deep=tf.keras.layers.Normalization()

norm_wide=norm_layer_wide(input_wide)
norm_deep=norm_layer_deep(input_deep)

hidden1=tf.keras.layers.Dense(30,activation="relu")(norm_wide)
hidden2=tf.keras.layers.Dense(30,activation="relu")(hidden1)
concat=tf.keras.layers.Concatenate()([norm_wide,hidden2])
output=tf.keras.layers.Dense(1)(concat) #sağda yazılan bir önceki katmanın ismi yazılır girdi olarak yani, 1 çıkış var 1 diye 1 seçildi

model=tf.keras.Model(inputs=[input_wide,input_deep],outputs=[output])
model.summary()

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse",optimizer=optimizer,metrics=["RootMeanSquaredError"])

X_train_wide=X_train[:,:5]
X_train_deep=X_train[:,2:]
X_test_wide=X_test[:,:5]
X_test_deep=X_test[:,2:]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)

history=model.fit((X_train_wide,X_train_deep),y_train,validation_split=0.2,epochs=20)

mse_test=model.evaluate((X_test_wide,X_test_deep),y_test)

X_new_wide=X_test_wide[:3]
X_new_deep=X_test_deep[:3]
y_pred=model.predict((X_new_wide,X_new_deep))
y_pred

"""# 2 çıkışlı sinir ağı oluşturmak functional api ile
mesela bir yüz fotoğrafından gülüyor mu şaşkın mı gibi duygu analizi yapmak ve gözlüklü mü değil mi diye analiz yapmak yani iki problem iki çıkış.
"""

input_wide=tf.keras.layers.Input(shape=[5])
input_deep=tf.keras.layers.Input(shape=[6])

norm_layer_wide=tf.keras.layers.Normalization()
norm_layer_deep=tf.keras.layers.Normalization()

norm_wide=norm_layer_wide(input_wide)
norm_deep=norm_layer_deep(input_deep)

hidden1=tf.keras.layers.Dense(30,activation="relu")(norm_wide)
hidden2=tf.keras.layers.Dense(30,activation="relu")(hidden1)

concat=tf.keras.layers.Concatenate()([norm_wide,hidden2])

output=tf.keras.layers.Dense(1)(concat)
aux_output=tf.keras.layers.Dense(1)(hidden2)

model=tf.keras.Model(inputs=[input_wide,input_deep],outputs=[output,aux_output])

optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(
    loss=["mse","mse"],
    loss_weights=[0.9,0.1],
    optimizer=optimizer,
    metrics=["RootMeanSquaredError", "RootMeanSquaredError"]
)

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)

history=model.fit(
    (X_train_wide,X_train_deep),
    (y_train,y_train),
    validation_split=0.2,
    epochs=20
)

eval_results=model.evaluate((X_test_wide,X_test_deep),(y_test,y_test))

y_pred_main,y_pred_aux=model.predict((X_new_wide,X_new_deep))

y_pred_main

y_pred_aux