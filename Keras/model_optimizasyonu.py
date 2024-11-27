# -*- coding: utf-8 -*-
"""Model_optimizasyonu.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14SkGCB-xi3RSNEbjpUCHNZ5VcHwIlK9C

#Model Performansı Arttırma

1.   feeding eğitim verilerinin batchler halinde sinir ağından geçmesidir
2.   batch eğitim verilerinden rastgele seçilir
3.   bütün verinin de sinir ağından 1 kez geçmesine epoch denir her epochda batchler rastgele seçilir
"""

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #bir boyuta çekiyoruz veri setimizi
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax") # 10 kategoriden oluştuğu için ve çok kategorili sınıflandırma olduğu için aktivasyona softmax girdik
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", # one hot coding yapmak için kısa yol ya da to_categorical() fonksiyonu ile yapmış olsaydık öncesinde şimdi categorical_crossentropy kullanırdık
              metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10)

#normalleştirme verileri 0 1 gibi belli bir aralığa getirmek demektir ve optimum sonuç için gereklidir çoğu zaman
#numpyda ondalıklı sayılar öntanımlı 32 bittir ama kerasta 64 bittir o yüzden 32 bite çeviriyoruz
import numpy as np
x_train=(x_train/255).astype(np.float32) #uğraştığımız veriler 255 bitlik resimler
x_test=(x_test/255).astype(np.float32)

model.fit(x_train,y_train,epochs=10)

"""**Normalleştirmeden önce 10 epoch sonunda bile 0.99 başarıya ulaşamamışken normalleştirdiğimizde 2. epochda 0.99'a ulaştı**"""

model.evaluate(x_test,y_test) # arada başarı oranı farkı var biraz demek ki bu modelde ezberleme sorunu olabilir

#normalleştirme yerine standartlaştırma da denenebilir verileri standartlaştırmak iyi bir model için daha iyidir
(x_train, y_train), (x_test, y_test) = mnist.load_data()
mean = np.mean(x_train)  #veri setinin ortalaması
std = np.std(x_train)  #standart sapma
x_train= ((x_train - mean) / std).astype(np.float32)

model.fit(x_train,y_train,epochs=10) #oranlar normalleştirmeye göre çok daha iyi artık