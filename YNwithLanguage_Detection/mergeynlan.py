import random
inp = open('inputvalrand_all.txt', 'w')
out = open('outputvalrand_all.txt', 'w')
Arr=[]
with open('inputval_all.txt') as f:
	lines=f.readlines()
	length=len(lines)
	#for i in range(length/2):
	#	yline=lines[i]
	#	nline=lines[(length/2)+i]
	#	inp.write("1 %s" % yline)
	#	inp.write("0 %s" % nline)
	randlist = random.sample(range(0,length),length)
	for i in randlist:
		line=lines[i]
		if (i*2<length):
			if line[0]=="1":
				add="1,0,0,0"
			else:
				add="0,1,0,0"
		else:
			if line[0]=="1":
				add="0,0,1,0"
			else:
				add="0,0,0,1"
		#cout=line[0]
		#cin=line[2:]
		inp.write("%s" % line[2:])
		out.write("%s\n" % add)

inp.close()
out.close()
