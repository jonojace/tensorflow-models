import tensorflow as tf

tf.enable_eager_execution()

# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

# To use a layer, simply call it.
layer(tf.zeros([10, 5]))

# Layers have many useful methods. For example, you can inspect all variables
# in a layer by calling layer.variables. In this case a fully-connected layer
# will have variables for weights and biases.
print(layer.variables)

# The variables are also accessible through nice accessors
print(layer.kernel, layer.bias)

'''
Implementing custom layers

The best way to implement your own layer is extending the tf.keras.Layer class 
and implementing: * __init__ , where you can do all input-independent 
initialization * build, where you know the shapes of the input tensors and 
can do the rest of the initialization * call, where you do the forward 
computation

Note that you don't have to wait until build is called to create your 
variables, you can also create them in __init__. However, the advantage 
of creating them in build is that it enables late variable creation based 
on the shape of the inputs the layer will operate on. On the other hand, 
creating variables in __init__ would mean that shapes required to create 
the variables will need to be explicitly specified.
'''

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        print('***inside init')
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        print('***inside build')
        print('input shape is', input_shape)
        #kernel is the matrix of weights in the layer
        self.kernel = self.add_variable("kernel", 
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])
                                        
    def call(self, input):
        print('***inside call')
        print('kernel is', self.kernel)
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
print('here', layer(tf.zeros([10, 5])))
print('there', layer.variables)


'''
Models: composing layers

Many interesting layer-like things in machine learning models are implemented 
by composing existing layers. For example, each residual block in a resnet is 
a composition of convolutions, batch normalizations, and a shortcut.

The main class used when creating a layer-like thing which contains other 
layers is tf.keras.Model. Implementing one is done by inheriting from 
tf.keras.Model.
'''

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()
        #no need to initialise relu here as it isn't a layer with learnable
        #stateful weights that we need to keep track of and update.
    
    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.variables])

my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(2, 1,
                                                     padding='same'),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(3, (1, 1)),
                              tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))
