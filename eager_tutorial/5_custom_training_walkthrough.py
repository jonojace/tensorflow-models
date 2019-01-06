from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


'''Import and parse the dataset'''

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

#column order as in the CSV data file
column_names = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

#create a tf.data.Dataset
batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

#With eager execution enabled, these Dataset objects are iterable
features, labels = next(iter(train_dataset))

print(features)

'''
#visualise our data
plt.scatter(features['petal_length'],
            features['sepal_length'],
            c=labels.numpy(),
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")

plt.show()
'''

#To simplify the model building step, create a function to repackage the 
# features dictionary into a single array with shape: (batch_size, num_features).


def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

#Then use the tf.data.Dataset.map method to pack the features of each 
# (features,label) pair into the training dataset:
train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

print(features[:5])


'''Select the type of model'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), #input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

#Let's have a quick look at what this model does to a batch of features:
predictions = model(features)
predictions[:5] #logits
print(tf.nn.softmax(predictions[:5])) #probabilities

#Taking the tf.argmax across classes gives us the predicted class index. 
#But, the model hasn't been trained yet, so these aren't good predictions.
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


'''Train the model'''

#define the loss and gradient function
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

l = loss(model, features, labels)
print("Loss test: {}".format(l))

#Use the tf.GradientTape context to calculate the gradients used to 
#optimize our model.
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#setup the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.Variable(0)

#run a single optimization step
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))

#training loop
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  #at each epoch we reset training metrics
  epoch_losses = []
  epoch_accuracies = []

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.variables),
                              global_step)

    # Track progress
    epoch_losses.append(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracies.append(tf.contrib.metrics.accuracy(y, tf.argmax(model(x), axis=1, output_type=tf.int32)))

  # end epoch
  train_loss_results.append(tf.math.reduce_mean(epoch_losses))
  train_accuracy_results.append(tf.math.reduce_mean(epoch_accuracies))

  #every 50 epochs we print out training metrics to terminal
  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                tf.math.reduce_mean(
                                                                    epoch_losses),
                                                                tf.math.reduce_mean(epoch_accuracies)))

#visualise loss function over time
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()

'''Evaluate the model's effectiveness'''

#setup the test dataset
test_url = "http://download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.contrib.data.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

#evaluate the model on the test dataset

for (x, y) in test_dataset:
  print("inside loop")
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy = tf.contrib.metrics.accuracy(y, prediction)
  
print("Test set accuracy: {:.3%}".format(test_accuracy))


'''Use the trained model to make predictions'''


predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5],
    [5.9, 3.0, 4.2, 1.5],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
