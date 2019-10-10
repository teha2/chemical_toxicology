# -*- coding: utf-8 -*-
"""
Model_Generator.py

author: Timothy E H Allen
"""
#%%

# Import Modules

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight

# DEFINE INPUTS FOR MODEL TRAINING

'''
input_data = dataset to be split into training and test tests
rng_1 and rng_2 = random numbers for dataset shuffle and train/test split
test_proportion = fraction of data to be used as test set
beta = l2 regularisation rate
neurons = neurons per hidden layer
hidden_layers = number of hidden layers, must be 1, 2 or 3
LR = learning rate
epochs = number of training iterations
model_path = location to save model file post training
'''

input_data = "/content/drive/My Drive/data/AR fingerprints ECFP6.csv"
rng_1 = 1
rng_2 = 2
test_proportion = 0.25
beta = 0.01
neurons = 100
hidden_layers = 1
LR = 0.01
epochs = 1000
model_path = "/content/drive/My Drive/data/AR model 1.h5"

print("Welcome to ChAI")
print("Dataset loading...")

# Reading The Dataset

def read_dataset():
    df = pd.read_csv(input_data)
    X = df[df.columns[0:5000]].values
    y = df[df.columns[5000]]

    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y)
    Y = encoder.transform(y)
    print("X.shape =", X.shape)
    print("Y.shape =", Y.shape)
    print("y.shape =", y.shape)
    return (X, Y)

X, Y = read_dataset()

# Shuffle the dataset
 
X, Y = shuffle(X, Y, random_state=rng_1)

# Convert the dataset into train and test sets

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size =test_proportion, random_state=rng_2)

# Inspect the shape of the training and test data

print("Dimensionality of data:")
print("Train x shape =", train_x.shape)
print("Train y shape =", train_y.shape)
print("Test x shape =", test_x.shape)
print("Test y shape =", test_y.shape)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_y),
                                                 train_y)

# Define the model in keras

print("Constructing model architecture")

if hidden_layers == 1:
    inputs = keras.Input(shape=(5000,), name='digits')
    x = layers.Dense(neurons, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(beta), name='dense_1')(inputs)
    outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
elif hidden_layers == 2:
    inputs = keras.Input(shape=(5000,), name='digits')
    x = layers.Dense(neurons, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(beta), name='dense_1')(inputs)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(beta), name='dense_2')(x)
    outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
elif hidden_layers == 3:
    inputs = keras.Input(shape=(5000,), name='digits')
    x = layers.Dense(neurons, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(beta), name='dense_1')(inputs)
    x = layers.Dense(neurons, activation='sigmoid', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(beta), name='dense_2')(x)
    x = layers.Dense(neurons, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(beta), name='dense_3')(x)
    outputs = layers.Dense(2, activation='softmax', name='predictions')(x)
else:
    print("Number of hidden layers outside this model scope, please choose 1, 2 or 3")

model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(optimizer=keras.optimizers.Adam(lr=LR),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

print('Commencing model training...')
history = model.fit(train_x, train_y,
                    batch_size=128,
                    epochs=epochs,
                    class_weight=class_weights,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(test_x, test_y))

# The returned "history" object holds a record
# of the loss values and metric values during training

# Evaluate the model on the training and test data

print('\n# Evaluate on training data')
train_results = model.evaluate(train_x, train_y, batch_size=128)
print('train loss, train acc:', train_results)

print('\n# Evaluate on test data')
test_results = model.evaluate(test_x, test_y, batch_size=128)
print('test loss, test acc:', test_results)

# Save the model

model.save(model_path)
print('Model saved to ' + model_path)

pred_test_y = model.predict(test_x, verbose=1)
pred_train_y = model.predict(train_x, verbose=1)

# Plot history of loss values

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot history of accuracy values
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Define experimental and predicted values using argmax

pred_train_y_binary = np.argmax(pred_train_y, axis=1)
pred_test_y_binary = np.argmax(pred_test_y, axis=1)

# Calculate and display confusion matricies

cm = confusion_matrix(train_y, pred_train_y_binary)
np.set_printoptions(precision=2)
print("Confusion matrix (Training), without normalisation")
print(cm)

cm = confusion_matrix(test_y, pred_test_y_binary)
np.set_printoptions(precision=2)
print("Confusion matrix (Test), without normalisation")
print(cm)

# Define a ROC curve generator

def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y,pred)
    roc_auc = auc(fpr,tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0,1], [0,1], "k--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()

# First plot Training data ROC

print("Training data ROC Curve")
    
y_score = np.array(pred_train_y)[:,1]
y_true = np.array(train_y)
plot_roc(y_score, y_true)

# Then plot Test data ROC

print("Test data ROC Curve")

y_score_2 = np.array(pred_test_y)[:,1]
y_true_2 = np.array(test_y)
plot_roc(y_score_2, y_true_2)

#End the cycle

tf.reset_default_graph()

print("END")