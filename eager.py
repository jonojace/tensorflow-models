#this code trains a flower classifier
#inputs: length and width measurements of sepals and petals
#outputs: class of flower (three classes in total)

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#Download the dataset
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
print("Local copy of the dataset file: {}".format(train_dataset_fp))

#Define function to parse the CSV file
def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]] #sets fields types
    parsed_line = tf.decode_csv(line, example_defaults)
    #first 4 fields are features, combine into a single tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))
    #last field is the label 
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1) #skip the first row
train_dataset = train_dataset.map(parse_csv) #parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000) #randomise
train_dataset = train_dataset.batch(32)

#view a single example entry from a batch 
features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])	
