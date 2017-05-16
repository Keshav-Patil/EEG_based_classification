import random
inp = open('inputvalrand_dec.txt', 'w')
out = open('outputvalrand_dec.txt', 'w')
Arr=[]
with open('inputval_dec.txt') as f:
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
			add="1"
		else:
			add="0"
		#cout=line[0]
		#cin=line[2:]
		inp.write("%s" % line)
		out.write("%s\n" % add)

inp.close()
out.close()
