import random
if __name__ == '__main__':
	data1 = []
	data2 = []
	data3 = []
	data4 = []
	f = open("train.txt","r")
	for l in f:
		temp = l.strip().split("\t")
		if temp[4] == "others":
			data1.append(temp)
		elif temp[4] == "angry":
			data2.append(temp)
		elif temp[4] == "happy":
			data3.append(temp)
		else:
			data4.append(temp)
	print(len(data1),len(data2),len(data3),len(data4))
	data1 = random.sample(data1,500)
	data2 = random.sample(data2,500)
	data3 = random.sample(data3,500)
	data4 = random.sample(data4,500)
	print(len(data1),len(data2),len(data3),len(data4))
	f = open("new_data.txt","w+")
	for l in data1:
		for l1 in l[:-1]:
			f.write(l1+"\t")
		f.write(l[-1]+"\n")
	for l in data2:
		for l1 in l[:-1]:
			f.write(l1+"\t")
		f.write(l[-1]+"\n")
	for l in data3:
		for l1 in l[:-1]:
			f.write(l1+"\t")
		f.write(l[-1]+"\n")
	for l in data4:
		for l1 in l[:-1]:
			f.write(l1+"\t")
		f.write(l[-1]+"\n")
	f.close()
