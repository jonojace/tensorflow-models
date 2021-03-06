from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

'''
https://www.tensorflow.org/guide/eager
'''

'''Setup and basic usage'''

tf.enable_eager_execution()

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)

# Broadcasting support
b = tf.add(a, 1)
print(b)

# Operator overloading is supported
print(a * b)

# Use NumPy values
c = np.multiply(a, b)
print(c)

# Obtain numpy value from a tensor:
print(a.numpy())

#The tf.contrib.eager module contains symbols available to both eager and 
# graph execution environments and is useful for writing code to work with 
# graphs:
tfe = tf.contrib.eager

'''Dynamic control flow'''

#A major benefit of eager execution is that all the functionality of the 
# host language is available while your model is executing. So, for example, 
# it is easy to write fizzbuzz:

def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('FizzBuzz')
        elif int(num % 3) == 0:
            print('Fizz')
        elif int(num % 5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        counter += 1

'''Build a model'''


class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.kernel = self.add_variable(
        "kernel", [input_shape[-1], self.output_units])

  def call(self, input):
    # Override call() instead of __call__ so we can perform some bookkeeping.
    return tf.matmul(input, self.kernel)

#can quickly build sequential models composed as a series of layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
    tf.keras.layers.Dense(10)
])

#we can define our own models


class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result


model = MNISTModel()

'''Eager training'''

#Automatic differentiation is useful for implementing machine learning 
# algorithms such as backpropagation for training neural networks. 
# During eager execution, use tf.GradientTape to trace operations for 
# computing gradients later.

#tf.GradientTape is an opt-in feature to provide maximal performance when 
# not tracing. Since different operations can occur during each call, all 
# forward-pass operations get recorded to a "tape". To compute the gradient, 
# play the tape backwards and then discard. A particular tf.GradientTape can 
# only compute one gradient; subsequent calls throw a runtime error.

w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)

#train a model
# Fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
     tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

# Build the model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

#Even without training, call the model and inspect the output in eager execution:
for images, labels in dataset.take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())

#training loop implemented with eager
optimizer = tf.train.AdamOptimizer()

loss_history = []

for (batch, (images, labels)) in enumerate(dataset.take(400)):
  if batch % 80 == 0:
    print()
  print('.', end='')
  with tf.GradientTape() as tape:
    logits = mnist_model(images, training=True)
    loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)

  loss_history.append(loss_value.numpy())
  grads = tape.gradient(loss_value, mnist_model.variables)
  optimizer.apply_gradients(zip(grads, mnist_model.variables),
                            global_step=tf.train.get_or_create_global_step())

#plot

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

'''Variables and optimizers'''

#tf.Variable objects store mutable tf.Tensor values accessed during training 
# to make automatic differentiation easier. The parameters of a model can be 
# encapsulated in classes as variables.

#Better encapsulate model parameters by using tf.Variable with 
# tf.GradientTape. For example, the automatic differentiation example 
# above can be rewritten:


class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')

  def call(self, inputs):
    return inputs * self.W + self.B


# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized


def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])


# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(
    loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                            global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(
        i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(
    loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

#object-oriented metrics
m = tfe.metrics.Mean("loss")
m(0)
m(5)
print(m.result())  # => 2.5
m([8, 9])
print(m.result())  # => 5.5
