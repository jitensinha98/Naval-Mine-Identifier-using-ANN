'''
The saved model is restored to perform some tests and classification
'''

# Importing required modules
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Model saved path
model_path="saved_model/model.ckpt"

def prep_dataset(dataset):
	x=dataset[dataset.columns[0:60]].values
	y=dataset[dataset.columns[60]]

	encoder = LabelEncoder()
	encoder.fit(y)
	y_=encoder.transform(y)
	
	# one-hot encoding the classes
	y_=pd.get_dummies(y_)

	# Shuffling the data
	x_original,y_original=shuffle(x,y_,random_state=1)

	# Splitting dataset
	train_x,test_x,train_y,test_y=train_test_split(x_original,y_original,test_size=0.20)

	return (train_x,test_x,train_y,test_y,x,y)

def neural_network_model(data):
	hidden_layer_1={'weights':tf.Variable(tf.random_normal([n_cols,hl1_nodes])),'biases':tf.Variable(tf.random_normal([hl1_nodes]))}
	hidden_layer_2={'weights':tf.Variable(tf.random_normal([hl1_nodes,hl2_nodes])),'biases':tf.Variable(tf.random_normal([hl2_nodes]))}
	hidden_layer_3={'weights':tf.Variable(tf.random_normal([hl2_nodes,hl3_nodes])),'biases':tf.Variable(tf.random_normal([hl3_nodes]))}
	hidden_layer_4={'weights':tf.Variable(tf.random_normal([hl3_nodes,hl4_nodes])),'biases':tf.Variable(tf.random_normal([hl4_nodes]))}
	output_layer={'weights':tf.Variable(tf.random_normal([hl4_nodes,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
	# Activation function
	tf.nn.relu(l1)

	l2=tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
	# Activation function
	tf.nn.relu(l2)

	l3=tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
	# Activation function
	tf.nn.relu(l3)

	l4=tf.add(tf.matmul(l3,hidden_layer_4['weights']),hidden_layer_4['biases'])
	# Activation function
	tf.nn.relu(l4)

	output=tf.add(tf.matmul(l4,output_layer['weights']),output_layer['biases'])

	return output
def train_neural_network(X):	
	prediction = neural_network_model(X)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=Y))
	optimizer=tf.train.AdamOptimizer().minimize(cost)
	
	# Creating a saver object
	saver=tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# Restoring saved model		
		saver.restore(sess, model_path)
		prediction_ = tf.argmax(prediction,1)
		correct_prediction=tf.equal(prediction_,tf.argmax(Y,1))
		accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
		test_accuracy_value=sess.run(accuracy,feed_dict={X:x_test,Y:y_test})
		print("Final Test Accuracy = ",test_accuracy_value*100," %")
		
		# Testing the similarity between predicted and original class
		for i in range(1,200):
			prediction_run=sess.run(prediction_,feed_dict={X:original_x[i].reshape(1,60)})
			print("Original Class = ",original_y[i]," Predicted Class = ",prediction_run)

sonar_data=pd.read_csv("sonar.csv")

learn_rate=0.3
hm_epochs=1300
n_classes=2
	
hl1_nodes=60
hl2_nodes=60
hl3_nodes=60
hl4_nodes=60

x_train,x_test,y_train,y_test,original_x,original_y=prep_dataset(sonar_data)

n_cols=x_train.shape[1]

X = tf.placeholder('float',[None,n_cols])
Y = tf.placeholder('float')

train_neural_network(X)


	


