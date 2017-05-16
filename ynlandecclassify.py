# Author: Aparajita Haldar (@ahaldar)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
X = numpy.loadtxt("inputvalyn_landec.txt", delimiter=',')

pca = PCA()
pca.fit(X)
X = pca.transform(X)

#X = normalize(X)

y1 = numpy.zeros((107,), dtype=numpy.int)
y2 = numpy.ones((107,), dtype=numpy.int)
arr=[]
for i in range(107):
	arr.append(1)
	arr.append(0)
Y = arr

# create model
model = Sequential()
model.add(Dense(5, input_dim=33, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


epochs = 300
learning_rate = 0.9
decay_rate = learning_rate / epochs
momentum = 0.9

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, validation_split = 0.3, nb_epoch=300, batch_size=30)

# evaluate the model
scores = model.evaluate(X, Y)
print "\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x) for x in predictions]
print rounded
print Y
