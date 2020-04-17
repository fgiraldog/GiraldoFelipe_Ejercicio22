import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection

numeros = sklearn.datasets.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)

data = imagenes.reshape((n_imagenes, -1))

scaler = sklearn.preprocessing.StandardScaler()
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, train_size=0.5)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

neurons = np.arange(1,20,1)
loss = []
f1_train = []
f1_test = []

for element in neurons:
	mlp = sklearn.neural_network.MLPClassifier(activation = 'logistic', 
		hidden_layer_sizes = element, max_iter = 10000)

	mlp.fit(x_train,y_train)

	loss.append(mlp.loss_)

	f1_train.append(sklearn.metrics.f1_score(y_train,mlp.predict(x_train),average = 'macro'))
	f1_test.append(sklearn.metrics.f1_score(y_test,mlp.predict(x_test),average = 'macro'))

plt.figure()
plt.subplot(211)
plt.plot(neurons,loss, label = 'max_iter = 10000')
plt.xlabel('Neurons')
plt.ylabel('Loss')
plt.legend()

plt.subplot(212)
plt.plot(neurons,f1_train, label = 'f1_train')
plt.plot(neurons,f1_test, label = 'f1_test')
plt.xlabel('Neurons')
plt.ylabel('F1 score')
plt.legend()

plt.tight_layout()
plt.savefig('loss_f1.png')

mlp_best = sklearn.neural_network.MLPClassifier(activation = 'logistic', 
	hidden_layer_sizes = 5, max_iter = 10000)

mlp_best.fit(x_train,y_train)

plt.figure()
for i in range(0,5):
	plt.subplot(2,3,i+1)
	plt.imshow(mlp_best.coefs_[0][:,i].reshape(8,8))
	plt.title('Neuron {}'.format(i+1))

plt.tight_layout()
plt.savefig('neuronas.png')