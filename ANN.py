'''
This is a Deep Neural Network Model to classify Rock and Mines.
Useful in Naval Mine Detection.
'''

# Importing essentials modules

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Model save path
model_path="saved_model/model.ckpt"

# Prepairing and Splitting dataset for training and validation
def prep_dataset(dataset):
	print("----------DATASET DIMENSION---------")
	print("\n")
	print(dataset.shape)
	print("\n")
	print("----------DATASET SAMPLE-----------")
	print("\n")
	print(dataset.head(5))
	print("\n")

	x=dataset[dataset.columns[0:60]].values
	y=dataset[dataset.columns[60]]

	colors=['red','blue']
	plt.title("Visualization of Rock(R) and Mine(M) labels in the dataset")
	plt.xlabel("Number of detections")
	plt.ylabel("Object")
	plt.xlim(0,250)
	y.value_counts().plot.barh(figsize=(15, 5), grid=True,color=colors)
	plt.show()

	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	
	# one-hot encoding the classes
	y=pd.get_dummies(y)

	# Shuffling the data
	x,y=shuffle(x,y,random_state=1)

	# Splitting dataset
	train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.20)

	return (train_x,test_x,train_y,test_y)

# Computational Graph
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
	
	# Cost function
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=Y))

	# Using Adam Optimizer
	optimizer=tf.train.AdamOptimizer().minimize(cost)
	
	# Creating a saver object
	saver=tf.train.Saver()
	
	# Starting session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())		
		for epoch in range(hm_epochs):
			epoch_list.append(epoch)
			sess.run(optimizer,feed_dict={X:x_train,Y:y_train})
			cost_value=sess.run(cost,feed_dict={X:x_train,Y:y_train})
			cost_list.append(cost_value)
			correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
			accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
			accuracy_value=sess.run(accuracy,feed_dict={X:x_train,Y:y_train})
			train_accuracy_list.append(accuracy_value)
			print("Epoch ",epoch," cost ",cost_value," Train Accuracy ",accuracy_value)

		correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
		test_accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
		test_accuracy_value=sess.run(test_accuracy,feed_dict={X:x_test,Y:y_test})
		print("Final Test Accuracy = ",test_accuracy_value*100," %")
		save_path=saver.save(sess,model_path)
		print("\n")
		print("MODEL SAVED ALONG WITH WEIGHTS AND BIASES...")

# Visualizing Loss and training Accuracy
def visualize_cost_accuracy_graph():
	plt.plot(epoch_list,cost_list,color='blue')
	plt.title("Cost Function vs Epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Cost Function")
	plt.show()

	plt.plot(epoch_list,train_accuracy_list,color='blue')
	plt.title("Training Accuracy vs Epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Training Accuracy")
	plt.show()

# Reading Dataset
sonar_data=pd.read_csv("sonar.csv")

learn_rate=0.3
hm_epochs=1000
n_classes=2

# Setting number of nodes in each layer	
hl1_nodes=60
hl2_nodes=60
hl3_nodes=60
hl4_nodes=60

x_train,x_test,y_train,y_test=prep_dataset(sonar_data)

n_cols=x_train.shape[1]

X = tf.placeholder('float',[None,n_cols])
Y = tf.placeholder('float')

cost_list=[]
train_accuracy_list=[]
epoch_list=[]

train_neural_network(X)
visualize_cost_accuracy_graph()


