import tensorflow as tf

tf.enable_eager_execution()

#using python state
x = tf.zeros([10, 10])
x += 2 
print(x)
#x here is a tensor, which are immutable stateless objects

#tensorflow has variables that are mutable and stateful objects
v = tf.Variable(1.0)
assert v.numpy() == 1.0

#re-assign the value 
v.assign(3.0)
assert v.numpy() == 3.0

#use 'v' in a tensorflow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0 

'''
Example: Fitting a linear model

In this tutorial, we'll walk through a trivial example of a simple linear 
model: f(x) = x * W + b, which has two variables - W and b. Furthermore, 
we'll synthesize data such that a well trained model would have W = 3.0 
and b = 2.0.
'''

class Model(object):
    def __init__(self):
        #initialise variable to (5.0, 0.0)
        #in practice, these should be initialised to random values
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b 

model = Model()

assert model(3.0).numpy() == 15.0

#we define a loss fn
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

#we synthesise the training data with some noise
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

#before we train the model lets plot the models predictions in red
#and the training data in blue
import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: ')
print(loss(model(inputs), outputs).numpy())

#we will build up the basic math for gradient descent ourselves 
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

#now we repeatedly run through the training data and see how W and b evolve
model = Model()

#collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    #note the use of the new f string
    print(f'Epoch {epoch}: W={Ws[-1]:.2f} b={bs[-1]:.2f}, loss={current_loss:.5f}')

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()
