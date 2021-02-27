
import tensorflow as tf

class SimpleMod(keras.Model):
    # reg controls regression or classification output
  def __init__(self, output_dim, nodes,name=None,reg = True):
    super(SimpleMod,self).__init__()

    self.output_dim = output_dim
    self.initial_node_dim = nodes

    self.dense1 = layers.Dense(self.initial_node_dim)
    self.dense2 = layers.Dense(self.initial_node_dim)
    self.dense_out = layers.Dense(self.output_dim)

  def call(self,input):
    x = tf.nn.relu(self.dense1(input))
    x = tf.nn.relu(self.dense2(input))

    if reg:
        return tf.nn.linear(self.dense_out(x))
    else:
        return tf.nn.sigmoid(self.dense_out(x))