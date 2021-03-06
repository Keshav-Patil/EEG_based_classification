# Author: Aparajita Haldar (@ahaldar)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import numpy
from sklearn.decomposition import PCA,RandomizedPCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Accent):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')






# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
X = numpy.loadtxt("inputvalrand_dec.txt", delimiter=',')
Y = numpy.loadtxt("outputvalrand_dec.txt", delimiter='\n')
pca = PCA(n_components=15,copy=True,whiten=True)
pca.fit(X)
X = pca.transform(X)

# create model
model = Sequential()
model.add(Dense(5, input_dim=15, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


epochs = [100,500,1000,2000,2500]
batch_size = [5,10,30,50,100]
#learning_rate = 0.9
#decay_rate = learning_rate / epochs
#momentum = 0.9

#sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# compile model

print "5  10  30  50  100\n"
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=7)
# fit the model
for i in xrange(len(epochs)):
	for j in xrange(len(batch_size)):
		model.fit(X, Y, validation_split = 0.33, nb_epoch=epochs[i], batch_size=batch_size[j], verbose=0)
		# evaluate the model
		scores = model.evaluate(X,Y)
		print ("%.4f" % scores[1]), "  ",
	print"\n"

'''
# calculate predictions
Y_predictions=model.predict(X_test)
predictions = model.predict(X)
Y_pred = [round(x) for x in Y_predictions]

# round predictions
rounded = [round(x) for x in predictions]
print rounded
yint =[]
tp=1
fn=1
tn=1
fp=1
for i in Y:
	yint.append(i)
print yint
for i in range(len(yint)):
	if yint[i]==rounded[i]:
		if yint[i]==1:
			tp+=1
		else:
			tn+=1
	else:
		if yint[i]==1:
			fn+=1
		else:
			fp+=1
print "true positive: ",tp
print "false positive: ",fp
print "false negative: ",fn
print "true negative: ",tn

conf = confusion_matrix(Y_test, Y_pred,labels=[1,0])
class_names=[0,1]
print conf
plt.figure()
plot_confusion_matrix(conf, classes=class_names,title='Confusion matrix, without normalization')
plt.show()
'''
