# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mZuraXVlOx-r65Vhda0-3VAOXK3xppAg

#Subclassing api ile dinamik model oluşturabiliriz
1.diğer yöntemlere göre daha zordur.

2.döngüler, koşullu dallanmalar gibi şeyleri içerebilir biz kurduğumuzdan.
"""

import tensorflow as tf
from tensorflow import keras

#lineer bir dense katmanı oluşturuyoruz.
class Linear(keras.layers.Layer):
  def __init__(self,units=32,input_dim=32):
    super(Linear,self).__init__()
    #şimdi ağırlık niteliği oluşturucaz
    w_init=tf.random_normal_initializer()
    #ağırlık niteliği fonksiyonu
    self.w=tf.Variable(
        initial_value=w_init(shape=(input_dim,units),dtype="float32"),
        trainable=True #ağırlıkların eğitilebiliceğini ifade eder geri dönüş yöntemiyle mesela

    )
    #şimdi de bias niteliğini oluşturuyoruz.
    b_init=tf.zeros_initializer()
    #bias biteliği fonksiyonu
    self.b=tf.Variable(
        initial_value=b_init(shape=(units,),dtype="float32"),
        trainable=True
    )

  def call(self,inputs):
    return tf.matmul(inputs,self.w)+self.b

x=tf.ones((2,2))
x

linear_layer=Linear(4,2) # 4 nöronu ve girdi boyutu da 2
y=linear_layer(x)
print(y)

"""Girdi boyutunu bilmeden, yazmadan katman ekleme"""

class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

linear_layer=Linear(32,)
y=linear_layer(x)
y

"""Art arda gelen katmanlar"""

class MLPBlock(tf.keras.layers.Layer):
  def __init__(self):
    super(MLPBlock,self).__init__()
    #girdi girmeye gerek yok çünkü bunlar ara katman
    self.linear_1=Linear(32) # ara katman 1
    self.linear_2=Linear(32) #ara katman 2
    self.linear_3=Linear(1) #output katmanı

  def call(self,inputs):
    x=self.linear_1(inputs) # ara katmana gidiyor ve relu aktivasyonundan geçiyor
    x=tf.nn.relu(x)

    x=self.linear_2(x)
    x=tf.nn.relu(x)

    return self.linear_3(x) #en son outputa gidiyor

mlp=MLPBlock()
y=mlp(tf.ones(shape=(3,64)))

mlp.weights