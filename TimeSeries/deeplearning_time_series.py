# -*- coding: utf-8 -*-
"""deeplearning_time_series.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BNUZRY86JMDeW09Ovc5SWtSf2alqkg-H
"""

import tensorflow as tf

zip_path=tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

import os
data_dir = os.path.dirname(zip_path)
for root, dirs, files in os.walk(data_dir):
    if 'jena_climate_2009_2016.csv' in files:
        csv_path = os.path.join(root, 'jena_climate_2009_2016.csv')
        break

with open(csv_path) as f:
    data = f.read()

lines=data.split('\n')
header=lines[0].split(',')
lines=lines[1:]

print(header)
print(len(lines))

import numpy as np #datetime çıkardık bu kodda
temperature=np.zeros((len(lines)))
raw_data=np.zeros((len(lines),len(header)-1))

for i in range(len(lines)):
    values=[float(x) for x in lines[i].split(',')[1:]]
    raw_data[i,:]=values[:]
    temperature[i]=values[1]

import matplotlib.pyplot as plt
plt.plot(range(len(temperature)),temperature) #görüldüğü üzere sıcaklık bir patterne sahip yaz aylarında artıp kış aylarında azalıyor

num_train_samples=int(0.5*len(raw_data))
num_val_samples=int(0.25*len(raw_data))
num_test_samples=int(0.25*len(raw_data))

print("Number of training samples:", num_train_samples)
print("Number of validation samples:", num_val_samples)
print("Number of testing samples:", num_test_samples)

mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
#veri standartlaştırma işlemi yaptık ortalamayı veriden çıkardık sonra standart sapmaya böldük veriyi

int_sequence=np.arange(10)
int_sequence

dummy_dataset=tf.keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],
    targets=int_sequence[3:],
    sequence_length=3,
    batch_size=2,
) #verileri yine pncereleştiriyoruz

for inputs,targets in dummy_dataset:
  print(inputs.numpy())
  print(targets.numpy())

sampling_rate=6 #veriler 10 dakikalık olarak olduğundan 1 saat olarak almak için
sequence_length=120 # 5 günlük verilere göre tahmin yapacağımızdan 5 günlük 24x5'den
delay=sampling_rate*(sequence_length+24-1)
batch_size=256

train_dataset=tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
)

val_dataset=tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples+num_val_samples,
)

test_dataset=tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples+num_val_samples,

)

for samples,targets in train_dataset:
  print("sample shape: ",samples.shape)
  print("target shape: ",targets.shape)
  break

def evaluate_naive_method(dataset):
    total_abs_err = 0.
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")
#gelecek 24 saat sıcaklğı ile şimdiki sıcaklık aynı olsaydı yaklaşık 2.5 derece bir hata ile tahmin yapılacaktı

from tensorflow import keras
from tensorflow.keras import layers

for samples, _ in train_dataset.take(1):
    input_shape = samples.shape[1:]
    break

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history=model.fit(train_dataset,epochs=10,validation_data=val_dataset,callbacks=callbacks)

