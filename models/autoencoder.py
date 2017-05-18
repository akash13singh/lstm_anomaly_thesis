import tensorflow as tf

class SimpleAutoencoder():

    def __init__(self,input_size,hidden_dimensions,learning_rate):
        self.hidden_dimensions = hidden_dimensions
        self.input_placeholder = tf.placeholder(tf.float32,[None, input_size],name = "ae_input_placeholder")
        self.input_size = input_size
        self.learning_rate = learning_rate

        self.weights = {
            'w1': tf.Variable(tf.random_normal([self.input_size, self.hidden_dimensions[0]]), name= "fc1_weights"),
            'w2': tf.Variable(tf.random_normal([self.hidden_dimensions[0], self.hidden_dimensions[1]]),name= "fc2_weights"),
            'w3': tf.Variable(tf.random_normal([self.hidden_dimensions[1], self.hidden_dimensions[2]]),name= "fc3_weights"),
            'w4': tf.Variable(tf.random_normal([ self.hidden_dimensions[2], input_size]),name= "fc4_weights"),
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.hidden_dimensions[0]]),name = "fc1_bias" ),
            'b2': tf.Variable(tf.random_normal([self.hidden_dimensions[1]]),name = "fc2_bias"),
            'b3': tf.Variable(tf.random_normal([self.hidden_dimensions[2]]),name = "fc3_bias"),
            'b4': tf.Variable(tf.random_normal([self.input_size]),name = "fc4_bias"),
        }


    def construct_model(self):
        input = self.input_placeholder
        with tf.name_scope("ae"):
            fc1 = tf.nn.sigmoid(tf.add(tf.matmul(input,self.weights['w1']),self.biases['b1']),name="fc1")
            self.fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1, self.weights['w2']), self.biases['b2']), name="fc2")
            fc3 = tf.nn.sigmoid(tf.add(tf.matmul(self.fc2, self.weights['w3']), self.biases['b3']), name="fc3")
            self.reconstuction = tf.nn.sigmoid(tf.add(tf.matmul(fc3, self.weights['w4']), self.biases['b4']), name="fc4_reconstuction")
            self.loss = tf.reduce_mean(tf.pow(tf.sub(self.input_placeholder,self.reconstuction), 2))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)