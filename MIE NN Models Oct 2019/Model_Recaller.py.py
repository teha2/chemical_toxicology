# -*- coding: utf-8 -*-
"""
Model_Recaller.py

author: Timothy E H Allen & Elena Gelzintye
"""

# Import modules

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance

# DEFINE INPUTS FOR MODEL RECALL

'''
trainging_data = oringinal training data for similarity calculations (.csv)
prediction_data = fingerprints to make predictions on (.csv)
model_location = model file (.h5)
prediction_output_location = file to write predictions to (.csv)
network_activation_strings = location to put activation strings for prediction compounds
network_activation_strings_training = location to put activation strings for training compounds
similarity_output_location = location to put similarity outputs
'''

training_data = "/content/drive/My Drive/data/AR fingerprints ECFP6.csv"
prediction_data = "/content/drive/My Drive/data/AR validation fingerprints ECFP6.csv"
model_location = "/content/drive/My Drive/data/AR model 1.h5"
prediction_output_location = "/content/drive/My Drive/data/AR model 1 predictions.csv"
network_activation_strings = "/content/drive/My Drive/data/AR model 1 NAS"
network_activation_strings_training = "/content/drive/My Drive/data/AR model 1 NAS Training"
similarity_output_location = "/content/drive/My Drive/data/AR model 1 similarities"

print("Welcome to ChAI")
print("Dataset loading...")

# Reading the prediction dataset

def read_prediction_dataset():
    df = pd.read_csv(prediction_data)
    #print(len(df.columns))
    X = df[df.columns[0:5000]].values
    y1 = df[df.columns[5000]]

    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    print("X.shape =", X.shape)
    print("Y.shape =", Y.shape)
    print("y.shape =", y.shape)
    return (X, Y, y1)

# Define the encoder function

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

X, Y, y1 = read_prediction_dataset()

n_compounds = X.shape[0]
n_dim = X.shape[1]
n_class = 2

print("Imput", n_compounds, "compounds for prediction")

# Call and use pretrained model

pretrained_model = tf.keras.models.load_model(model_location)
pretrained_model.summary()
y = pretrained_model.predict(X, verbose=1)

y_ten = tf.convert_to_tensor(y, dtype=tf.float32)
y1_ten = tf.convert_to_tensor(Y, dtype=tf.float32)

with tf.Session() as sess:
    prediction = sess.run(tf.argmax(y_ten, 1))
    accuracy = sess.run(tf.cast(tf.equal(tf.argmax(y_ten, 1), tf.argmax(y1_ten, 1)), tf.float32))
    total_accuracy = str(sess.run(tf.reduce_mean(accuracy)))


original_class = y1.values
print("Overall Model Accuracy = " + total_accuracy)

f = open(prediction_output_location, 'w+')

print("Model Recall Initialized")

print("**************************************************")
print("0 stands for miss & 1 stands for hit at the target")
print("**************************************************")
i=0

for i in range(0,n_compounds):
    print("Original Class : ", original_class[i], "Predicted Values : ", prediction[i], "Accuracy : ", accuracy[i], "Probability Active : ", y[i,1])
    print("Original Class,", original_class[i], ",Predicted Values,", prediction[i], ",Accuracy,", accuracy[i], ",Probability Active,", y[i,1] , file=f)

f.close()

print('commencing similarity calculation')

def read_training_dataset():
    df = pd.read_csv(training_data)
    #print(len(df.columns))
    A = df[df.columns[0:5000]].values
    b1 = df[df.columns[5000]]

    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y1)
    b = encoder.transform(b1)
    B = one_hot_encode(b)
    print("A.shape =", A.shape)
    print("B.shape =", B.shape)
    print("b.shape =", b.shape)
    return (A, B, b1)

A, B, b1 = read_training_dataset()

receptor='HERG'
method='random split'
fold=1

# Define functions for similarity calculations

def layer_propagation(x, weight, bias, function):
    
    '''with given x (from previous layer or from input) and given weights and biases
    propagates one layer. NB tensor dimensions must match '''
    
    layer=tf.add(tf.matmul(x, weight), bias)
    if function=='relu':
        layer=tf.nn.relu(layer)
    elif function=='sigmoid':
        layer=tf.nn.sigmoid(layer)    
    
    return layer

def get_euclidean_dist_mx(string1, string2):
    similarities=np.zeros((len(string1), len(string2)))
    for no_1 in range(0, len(string1)):
        for no_2 in range(0, len(string2)):
            
            similarities[no_1, no_2]=distance.euclidean(string1[no_1],string2[no_2])
            
    return similarities

# Get weights and biases from keras model
    
no_hidden_layers = len(pretrained_model.layers) - 2

if no_hidden_layers == 1:
    weights_h1 = pretrained_model.layers[1].get_weights()[0]
    biases_b1 = pretrained_model.layers[1].get_weights()[1]
    
elif no_hidden_layers == 2:
    weights_h1 = pretrained_model.layers[1].get_weights()[0]
    biases_b1 = pretrained_model.layers[1].get_weights()[1]
    weights_h2 = pretrained_model.layers[2].get_weights()[0]
    biases_b2 = pretrained_model.layers[2].get_weights()[1]

else:
    weights_h1 = pretrained_model.layers[1].get_weights()[0]
    biases_b1 = pretrained_model.layers[1].get_weights()[1]
    weights_h2 = pretrained_model.layers[2].get_weights()[0]
    biases_b2 = pretrained_model.layers[2].get_weights()[1]
    weights_h3 = pretrained_model.layers[3].get_weights()[0]
    biases_b3 = pretrained_model.layers[3].get_weights()[1]

# Get network activation strings

X_val=X.astype(np.float32)
A_val=A.astype(np.float32)

n_samples=len(X_val)

len_fp=X_val.shape[1] 

# Get strings for validation compounds
   
fingerprint = tf.placeholder(tf.float32, [None, len_fp], name='fp_placehold')

if no_hidden_layers == 1:
        activ_funs=('sigmoid')
        A1=layer_propagation(X_val, weights_h1, biases_b1, activ_funs[0])

        with tf.Session() as sess:
            A1=sess.run(A1)

        joint=A1
    
elif no_hidden_layers == 2:
        activ_funs=('sigmoid', 'relu')
        A1=layer_propagation(X_val, weights_h1, biases_b1, activ_funs[0])
        A2=layer_propagation(A1, weights_h2, biases_b2, activ_funs[1])

        with tf.Session() as sess:
            A1=sess.run(A1)
            A2=sess.run(A2)

        joint_temp=np.concatenate(([A1], [A2]), axis=1)
        joint = np.squeeze(joint_temp, axis = 0)

else:
        activ_funs=('sigmoid', 'sigmoid', 'relu')
        A1=layer_propagation(X_val, weights_h1, biases_b1, activ_funs[0])
        A2=layer_propagation(A1, weights_h2, biases_b2, activ_funs[1])
        A3=layer_propagation(A2, weights_h3, biases_b3, activ_funs[2])
        
        with tf.Session() as sess:
            A1=sess.run(A1)
            A2=sess.run(A2)
            A3=sess.run(A3)

        joint_temp=np.concatenate(([A1], [A2], [A3]), axis=1)
        joint = np.squeeze(joint_temp, axis = 0)
    
print(joint.shape)
print(type(joint))
print(joint)

np.save(network_activation_strings.format(receptor, method, fold), joint)

# Get strings for training compounds
   
fingerprint_train = tf.placeholder(tf.float32, [None, len_fp], name='fp_placehold')

if no_hidden_layers == 1:
        A1_train=layer_propagation(A_val, weights_h1, biases_b1, activ_funs[0])

        with tf.Session() as sess:
            A1_train=sess.run(A1_train)

        joint_train=A1_train
    
elif no_hidden_layers == 2:
        A1_train=layer_propagation(A_val, weights_h1, biases_b1, activ_funs[0])
        A2_train=layer_propagation(A1_train, weights_h2, biases_b2, activ_funs[1])

        with tf.Session() as sess:
            A1_train=sess.run(A1_train)
            A2_train=sess.run(A2_train)

        joint_train_temp=np.concatenate(([A1_train], [A2_train]), axis=1)
        joint_train = np.squeeze(joint_train_temp, axis = 0)

else:
        A1_train=layer_propagation(A_val, weights_h1, biases_b1, activ_funs[0])
        A2_train=layer_propagation(A1_train, weights_h2, biases_b2, activ_funs[1])
        A3_train=layer_propagation(A2_train, weights_h3, biases_b2, activ_funs[2])

        with tf.Session() as sess:
            A1_train=sess.run(A1_train)
            A2_train=sess.run(A2_train)
            A3_train=sess.run(A3_train)

        joint_train_temp=np.concatenate(([A1_train], [A2_train], [A3_train]), axis=1)
        joint_train = np.squeeze(joint_train_temp, axis = 0)

print(joint_train.shape)
print(type(joint_train))
print(joint_train)

np.save(network_activation_strings_training.format(receptor, method, fold), joint)

#generate similarity matrix - takes a while

print("Generating similarity matrix")
ntw_eucl_dist=get_euclidean_dist_mx(joint, joint_train)
ntw_eucl_sim=1/(1+ntw_eucl_dist)

np.save(similarity_output_location + ".npy".format(receptor, method, fold), ntw_eucl_sim)

ntw_eucl_sim=np.load(similarity_output_location + ".npy".format(receptor, method, fold))

#convert to dataframe, save as csv

print("Saving dataframe as CSV")
ntw_eucl_sim_pandas=pd.DataFrame(ntw_eucl_sim)
ntw_eucl_sim_pandas.to_csv(similarity_output_location + ".csv".format(receptor, method, fold))

#Endgame

print("END")