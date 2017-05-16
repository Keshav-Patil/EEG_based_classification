import pyeeg
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat("/home/keshav/eeg/LRI-Shubham-Hindi-Feb1_20170201_121230_fil_seg.mat")
data1 = sio.loadmat("/home/keshav/eeg/LRI-BHAKTI-Feb1-HINDI_20170201_062016_fil_seg.mat")
data2 = sio.loadmat("/home/keshav/eeg/LRI-Gourav-Feb2-Hindi_20170202_122049_fil_seg.mat")
data3 = sio.loadmat("/home/keshav/eeg/LRI_Shubham_Feb1_20170201_120407_fil_seg.mat")
data4 = sio.loadmat("/home/keshav/eeg/LRI-Gourav-Feb2-English_20170202_121310_fil_seg.mat")
data5 = sio.loadmat("/home/keshav/eeg/lri_bhakti_english_20170201_061129_fil_seg.mat")
data7 =  sio.loadmat("/home/keshav/eeg/LRI-Afroze-English-Feb10_20170210_011033_fil_seg.mat")
data8 =  sio.loadmat("/home/keshav/eeg/LRI-Afroze-Hindi-Feb10_20170210_013258_fil_seg.mat")
data9 =  sio.loadmat("/home/keshav/eeg/LRI-Sarthak-Feb3-English_20170203_120716_fil_seg.mat")
data10 =  sio.loadmat("/home/keshav/eeg/LRI-Sarthak-Feb3-Hindi_20170203_121504_fil_seg.mat")

var = sio.whosmat("/home/keshav/eeg/LRI-Shubham-Hindi-Feb1_20170201_121230_fil_seg.mat")
var1 = sio.whosmat("/home/keshav/eeg/LRI-BHAKTI-Feb1-HINDI_20170201_062016_fil_seg.mat")
var2 = sio.whosmat("/home/keshav/eeg/LRI-Gourav-Feb2-Hindi_20170202_122049_fil_seg.mat")
var3 = sio.whosmat("/home/keshav/eeg/LRI_Shubham_Feb1_20170201_120407_fil_seg.mat")
var4 = sio.whosmat("/home/keshav/eeg/LRI-Gourav-Feb2-English_20170202_121310_fil_seg.mat")
var5 = sio.whosmat("/home/keshav/eeg/lri_bhakti_english_20170201_061129_fil_seg.mat")
var7 =  sio.whosmat("/home/keshav/eeg/LRI-Afroze-English-Feb10_20170210_011033_fil_seg.mat")
var8 =  sio.whosmat("/home/keshav/eeg/LRI-Afroze-Hindi-Feb10_20170210_013258_fil_seg.mat")
var9 =  sio.whosmat("/home/keshav/eeg/LRI-Sarthak-Feb3-English_20170203_120716_fil_seg.mat")
var10 =  sio.whosmat("/home/keshav/eeg/LRI-Sarthak-Feb3-Hindi_20170203_121504_fil_seg.mat")

x=[i for i in range(2250)]
y=[np.zeros(9) for i in range(2250)]	
def Window(d,n):
	relev=[11,13,15,5,7,1,27,2,3,17,4]	
	tpower=[[0,0,0,0,0] for i in relev]
	avpower=[[0,0,0] for i in relev]
	print 'window size: ',n
	print len(d[0])
	curry=[]	
	for i in range(0,len(d[0])-n+2):
		window=d.T[i:i+n].T
		cpower=Power(window)
		curry.append(cpower[0][2])				
		for j in range(0,len(tpower[0])):
			for ind in range(len(relev)):
				tpower[ind][j]+=cpower[ind][j]
	print 'total number of windows: ',i
	for j in range(0,len(tpower[0])-2):
		for ind in range(len(relev)):
			avpower[ind][j]=tpower[ind][j+2]/i		
	return avpower,curry
		
def Power(d):
	Band=[0.5,4,8,13,30,40]
	relev=[11,13,15,5,7,1,27,2,3,17,4]
	power=[np.zeros(len(Band) -1) for i in relev]
	count=0
	for ind in range(len(power)):
		da=d[relev[ind]]
		C=np.fft.fft(da)
		C=abs(C)
		for FI in range(0,len(Band) -1):
			F=float(Band[FI])
			NF=float(Band[FI+1])
			power[ind][FI]=sum(C[int(np.floor(F/250*len(da))):int(np.floor(NF/250*len(da)))])
			Power_Ratio = power[ind] / sum(power[ind])
	return power			

yes=[18,19,21,22,26,28,29,30,31]
no=[20,23,24,25,27,32,33,34,35]
yes1 = [18,20,25,26,27,30,32,34,35]
no1 =  [19,21,22,23,24,28,29,31,33]
yesp = [20,21,26,28,30,31,32,33,35]
nop = [18,19,22,23,24,25,27,29,34]
inp = open('inputval_all.txt', 'w')
for que in range(18,36):
	d =data[var[que][0]]
	print var[que][0]
	cwin,cy = Window(d,250)
	print 'length of the list: ',len(cwin)	
	flatten=[x for sublist in cwin for x in sublist]
	print flatten
	print 'length of the list: ',len(flatten)
	flatstring = ', '.join(map(str, flatten))
	if que in yes:
  		inp.write("1 %s\n" % flatstring)
	elif que in no:
  		inp.write("0 %s\n" % flatstring)
			

for que in range(20,40):
        d =data1[var1[que][0]]
        print var1[que][0]
        cwin,cy = Window(d,250)
        print 'length of the list: ',len(cwin)
        flatten=[x for sublist in cwin for x in sublist]
        print flatten
        print 'length of the list: ',len(flatten)
        flatstring = ', '.join(map(str, flatten))
        if que<30:
		inp.write("1 %s\n" % flatstring)
	else:
  		inp.write("0 %s\n" % flatstring)

for que in range(20,40):
	d =data2[var2[que][0]]
	print var2[que][0]
	cwin,cy = Window(d,250)
	print 'length of the list: ',len(cwin)	
	flatten=[x for sublist in cwin for x in sublist]
	print flatten
	print 'length of the list: ',len(flatten)
	flatstring = ', '.join(map(str, flatten))
	if que<30:
                inp.write("1 %s\n" % flatstring)
        else:
                inp.write("0 %s\n" % flatstring)

for que in range(20,40):
        d =data8[var8[que][0]]
        print var8[que][0]
        cwin,cy = Window(d,250)
        print 'length of the list: ',len(cwin)
        flatten=[x for sublist in cwin for x in sublist]
        print flatten
        print 'length of the list: ',len(flatten)
        flatstring = ', '.join(map(str, flatten))
	if que<30:
                inp.write("1 %s\n" % flatstring)
        else:
                inp.write("0 %s\n" % flatstring)

for que in range(20,40):
        d =data10[var10[que][0]]
        print var10[que][0]
        cwin,cy = Window(d,250)
        print 'length of the list: ',len(cwin)
        flatten=[x for sublist in cwin for x in sublist]
        print flatten
        print 'length of the list: ',len(flatten)
        flatstring = ', '.join(map(str, flatten))
	if que<30:
                inp.write("1 %s\n" % flatstring)
        else:
                inp.write("0 %s\n" % flatstring)

for que in range(18,36):
	d =data3[var3[que][0]]
	print var3[que][0]
	cwin,cy = Window(d,250)
	print 'length of the list: ',len(cwin)	
	flatten=[x for sublist in cwin for x in sublist]
	print flatten
	print 'length of the list: ',len(flatten)
	flatstring = ', '.join(map(str, flatten))
	if que in yes1:
                inp.write("1 %s\n" % flatstring)
        elif que in no1:
                inp.write("0 %s\n" % flatstring)
	
	
for que in range(20,40):
	d =data5[var5[que][0]]
	print var5[que][0]
	cwin,cy = Window(d,250)
	print 'length of the list: ',len(cwin)	
	flatten=[x for sublist in cwin for x in sublist]
	print flatten
	print 'length of the list: ',len(flatten)
	flatstring = ', '.join(map(str, flatten))
	if que<30:
                inp.write("1 %s\n" % flatstring)
        else:
                inp.write("0 %s\n" % flatstring)

for que in range(20,40):
	d =data4[var4[que][0]]
	print var4[que][0]
	cwin,cy = Window(d,250)
	print 'length of the list: ',len(cwin)	
	flatten=[x for sublist in cwin for x in sublist]
	print flatten
	print 'length of the list: ',len(flatten)
	flatstring = ', '.join(map(str, flatten))
  	inp.write("%s\n" % flatstring)
  	
for que in range(20,40):
        d =data7[var7[que][0]]
        print var7[que][0]
        cwin,cy = Window(d,250)
        print 'length of the list: ',len(cwin)
        flatten=[x for sublist in cwin for x in sublist]
        print flatten
        print 'length of the list: ',len(flatten)
        flatstring = ', '.join(map(str, flatten))
	if que<30:
                inp.write("1 %s\n" % flatstring)
        else:
                inp.write("0 %s\n" % flatstring)

for que in range(20,40):
        d =data9[var9[que][0]]
        print var9[que][0]
        cwin,cy = Window(d,250)
        print 'length of the list: ',len(cwin)
        flatten=[x for sublist in cwin for x in sublist]
        print flatten
        print 'length of the list: ',len(flatten)
        flatstring = ', '.join(map(str, flatten))
	if que<30:
                inp.write("1 %s\n" % flatstring)
        else:
                inp.write("0 %s\n" % flatstring)

inp.close()
	





