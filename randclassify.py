import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import cross_val_score
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, datasets
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Accent):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
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






C = 1
kernel = 'rbf'
iterations = 300
m_accuracysvm = []
m_accuracyrf = []
m_accuracyab = []
X = pd.read_csv('inputvalrand_dec.txt',sep=',',header = None)
Y = np.loadtxt("outputvalrand_dec.txt", delimiter='\n')
#X = normalize(X)
pca = PCA(n_components=15,copy=True,whiten=True)
pca.fit(X)
X = pca.transform(X)
fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
cxaxis=[]
cyaxis=[]
czaxis=[]
target = open("rand_decscores.txt", 'w')
line="Estimators   \t| 40  50  60  70\n"
target.write(line)
line="    \t_______________________\n"
target.write(line)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=7)
estim = 40
max=0
maxe=0
maxi=0
while iterations < 601:
	estim = 40 
	line=""
	print "Iterations: ",iterations," | ",
	line+="Iterations "+str(iterations)+"\t| "
	while estim < 71:
		cxaxis.append(iterations)
        	cyaxis.append(estim)
		rf = RandomForestClassifier(n_estimators = estim)
		for _ in range(iterations):
			rf.fit(X_train,y_train)
			m_accuracyrf.append(rf.score(X_test,y_test))
		y_pred = rf.predict(X_test)
		predictions = rf.predict(X)
		nmean =  np.mean(m_accuracyrf)
		print nmean,
		czaxis.append(nmean)
		if max < nmean:
                        max = nmean
                        maxe = estim
                        maxi = iterations
		line+=str("%.4f" % nmean)+" "
		estim+=10
	iterations+=100
	line+="\n"
	target.write(str(line))
	print 
print "Score: ",max," Iterations: ",maxi," Estimators: ",maxe
line="MaxScore: "+str(max)+" Iterations: "+str(maxi)+" Estimators: "+str(maxe)
target.write(line)
target.close()
xaxis=np.array(cxaxis)
yaxis=np.array(cyaxis)
hist, xedges, yedges = np.histogram2d(xaxis, yaxis)
xpos, ypos = np.meshgrid(xedges-50, yedges-5)
xpos = xaxis.flatten('F')-50
ypos = yaxis.flatten('F')-5
zpos= np.zeros_like(xpos)+0.58
print xpos.shape
print ypos.shape
print zpos.shape
# Construct arrays with the dimensions for the 16 bars.
dx = 100 * np.ones_like(zpos)
dy = 10 * np.ones_like(zpos)
dz = np.array(czaxis).flatten()-0.58
colors = ['r','g','b']
surf = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='g', zsort='average')
ax.set_xlim3d(250, 750)
ax.set_ylim3d(35,75)
ax.set_zlim3d(0.58,0.60)
plt.show()

"""
y_pred = rf.predict(X_test)
predictions = rf.predict(X)
# round predictions
rounded = [round(x) for x in predictions]
print rounded

#	m_accuracyab.append(svm.score(X_test,y_test))
#m_accuracyrf.append(rf.score(X_test,y_test))
#	m_accuracyab.append(ab.score(X_test,y_test))
	
#print "Accuracies:"	
#print "SVM:", np.mean(m_accuracysvm)

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

conf = confusion_matrix(y_test, y_pred,labels=[1,0])
class_names=[0,1]
print conf
plt.figure()
plot_confusion_matrix(conf, classes=class_names,title='Confusion matrix, without normalization')
plt.show()

"""
