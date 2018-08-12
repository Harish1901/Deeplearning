import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/Santhosh/Downloads/edureka/Module 1/wine.csv") #print (df.describe()) print(df.shape)
sess = tf.Session()
print(len(df.columns))
X = df[df.columns[1:13]].values
y = df['wine'].values-1
Y = tf.one_hot(indices = y, depth=3, on_value = 1., off_value = 0., axis = 1 ,name ="a").eval(session=sess)
X, Y = shuffle(X, Y) 
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
X,Y =shuffle (X, Y, random_state=1)
Xtr=X[0:140,:]
Ytr=Y[0:140,:]
Xt=X[140:178,:]
Yt=Y[140:178,:]
Xtr, Ytr = shuffle (Xtr, Ytr, random_state=0)
batch_xs, batch_ys = Xtr , Ytr
cost_history = np.empty(shape=[1],dtype=float)
training_epochs = 1000
learning_rate = 0.1
n_dim = X.shape[1]
n_class = 3

n_hidden_1 = 15
n_hidden_2 = 15
n_hidden_3 = 20
x= tf.placeholder(tf.float32, [None, n_dim])
W= tf.Variable(tf.zeros([n_dim,n_class]))
b= tf.Variable(tf.zeros([n_class]))
def multilayer_perceptron(x, weights, biases):
# Hidden layer with RELU activation layer_1 = tf.add(tf.matmul(x, weights[&#39;h1&#39;]), biases[&#39;b1&#39;])
layer_1 = tf.nn.softmax(layer_1)
# Hidden layer with RELU activation
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
layer_2 = tf.nn.softmax(layer_2) # Hidden layer with RELU activation
layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases[b3'])
layer_3 = tf.nn.relu(layer_3) # Output layer with linear activation
out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
return out_layer
weights = {
'h1': tf.Variable(tf.random_normal([n_dim, n_hidden_1])),
'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
'out': tf.Variable(tf.random_normal([n_hidden_3, n_class]))
}
biases = {
'b1': tf.Variable(tf.random_normal([n_hidden_1])),
'b2': tf.Variable(tf.random_normal([n_hidden_2])),

'b3': tf.Variable(tf.random_normal([n_hidden_3])),
'out': tf.Variable(tf.random_normal([n_class]))
}
init = tf.global_variables_initializer()
# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 3])
y= multilayer_perceptron(x,weights,biases)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
sess = tf.Session()
sess.run(init)
for i in range(training_epochs):
sess.run(train_step,feed_dict=({x: batch_xs, y_: batch_ys}))
cost = sess.run (cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
cost_history= np.append(cost_history,cost)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(&quot;Accuracy&quot;,(sess.run(accuracy,feed_dict={x: Xt, y_: Yt})))