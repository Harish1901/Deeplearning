import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#reading data
df = pd.read_csv("/Users/Santhosh/Downloads/edureka/Module 1/wine.csv")
# initialising session
sess = tf.Session() # starting tensor flow session
# defining variables from dataset 
X = df.iloc[:,1:14].values
y=df.iloc[:,0].values
    #encode the depedent variable, single it has more than one class
#X = df[df.columns[1:14]]
#y = df["Wine"].values - 1
    # one hot encoding on dependent since there are 3 class of wines
Y = tf.one_hot(indices = y, depth=3, on_value = 1., off_value = 0., axis = 1, name = "a").eval(session=sess)

#some preprocessing on data
X, Y = shuffle (X, Y)
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X) #normalising
X,Y =shuffle (X, Y, random_state=1)

# splitting data for testing and training
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=42)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

#creating cost hisotry array
cost_history = np.empty(shape=[1],dtype=float)
#definging network adjustable parameters
training_epochs = 1000
learning_rate = 0.01
n_dim = X.shape[1]
n_class = 3 ##3 class of wines 
x= tf.placeholder(tf.float32, [None, 13])
W= tf.Variable(tf.zeros([13, 3]))
b= tf.Variable(tf.zeros([3]))
init = tf.global_variables_initializer()
y_ = tf.placeholder(tf.float32, [None, 3])
y= tf.nn.softmax(tf.matmul(x, W) + b)
#prediciting cost function
cost_function = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
#performing optimizer to reduce error
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)
mse_history = []
for epoch in range(100): 
    sess.run(training_step,feed_dict=({x: train_x, y_:train_y}))
    cost = sess.run (cost_function, feed_dict={x: train_x, y_:train_y})
    cost_history= np.append(cost_history,cost)
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_history.append(sess.run(mse))
    
#print the final mean square error
print("MSE:",mse_history)
plt.plot(mse_history, 'ro-')
plt.show()
    
pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32)) 
print("------------Print Accuracy----------") 
print("Accuracy",(sess.run(accuracy,feed_dict={x: test_x, y_: test_y}))) # checking the accuracy of our model. plt.plot(range(len(cost_history)),cost_history)
#plotting the graph for the same
plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()