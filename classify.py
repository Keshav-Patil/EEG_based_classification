# Author: Aparajita Haldar (@ahaldar)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
X = numpy.loadtxt("inputval_dec.txt", delimiter=',')
y1 = numpy.zeros((29,), dtype=numpy.int)
y2 = numpy.ones((29,), dtype=numpy.int)
Y = numpy.concatenate((y2, y1))

# create model
model = Sequential()
model.add(Dense(12, input_dim=18, init='normal', activation='relu'))
model.add(Dense(15, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))

'''
epochs = 500
learning_rate = 0.5
decay_rate = learning_rate / epochs
momentum = 0.9

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
'''
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, validation_split = 0.33, nb_epoch=200, batch_size=20)

# evaluate the model
scores = model.evaluate(X, Y)
print "\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100)

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x) for x in predictions]
print rounded

