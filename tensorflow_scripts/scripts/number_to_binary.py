import os,io,datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

def train(data_csv, label_column, model_path):

    # import file and define labels and feature coplumns
    data_all = pd.read_csv(data_csv).iloc[: , 1:]
    labels_all = data_all[label_column]
    features_all = data_all.drop(columns =[label_column])

    # train/test split
    last_train = int(len(data_all)*0.75)
    train_index = data_all.iloc[0:last_train].index.values
    test_index = data_all.iloc[last_train+1:].index.values
    train_set = [features_all.iloc[train_index].values,labels_all.iloc[train_index].values]
    test_set = [features_all.iloc[test_index].values,labels_all.iloc[test_index].values]

    train_data, train_labels = train_set[0], train_set[1]
    test_data, test_labels = test_set[0], test_set[1]

    # define model structure
    model = keras.Sequential([
    keras.layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4)),
    keras.layers.Dropout(.1),
    keras.layers.Dense(1, activation = 'sigmoid')
    ])

    # define optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001)

    # callback for tensorboard
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard/', histogram_freq=1)

    # train the model
    model.compile(optimizer = opt, loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs = 100, validation_data=(test_data, test_labels)) # , callbacks=[tensorboard_callback]

    # evaluate model
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    # save model
    model.save(model_path, 'models/ml_model_ntb')


def predict(data_csv, label_column, model_path):

    # import and truncate data
    data_all = pd.read_csv(data_csv).iloc[: , 1:]
    labels_all = data_all[label_column]
    features_all = data_all.drop(columns =[label_column])

    train_index = data_all.iloc[0:len(data_all)].index.values
    train_set = [features_all.iloc[train_index].values,labels_all.iloc[train_index].values]
    train_data, train_labels = train_set[0], train_set[1]

    model = keras.models.load_model('models/ml_model_ntb')
    predicted_labels = model.predict(train_data)

    return predicted_labels

if __name__ == '__main__':
    train('../data/number_to_binary_data.csv', 'target', 'models/ml_model_ntb')
    print(predict('../data/number_to_binary_data.csv', 'target', 'models/ml_model_ntb'))