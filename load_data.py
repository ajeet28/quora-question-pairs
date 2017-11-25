import re
import csv
from nltk import word_tokenize
import numpy as np

trainfile="data/train.csv"
testfile="data/test_withlabels.csv"

def process_sentence(strs):

	nstr = re.sub(r'[?|$|.|!]:',r'',strs)
	nestr = re.sub(r'[^a-zA-Z0-9 ]',r'',nstr)

	nestr = re.sub('[^a-zA-Z0-9\n\.]', ' ', strs)
	sentence = word_tokenize(nestr)	
	return [word.lower()  for word in sentence]


def convert_to_num(q_list, word_2_idx, max_len):

	retVal = []

	for item in q_list:
		tmp = []
	
		for jitem in item:
			tmp.append( word_2_idx[jitem])	
	
		tmp += [0]*(max_len - len(tmp))
		retVal.append(tmp)

	retVal = np.array(retVal,dtype=np.int32)

	assert len(retVal.shape) == 2

	return retVal


def load():
	#reading data file
	fp = csv.reader(open(trainfile,"r"))
	q1_train = []
	q2_train = []
	label_train = []

	count = 0 
	for row in fp:
		if count == 0:
			count = 1
			continue
		q1_train.append(process_sentence(row[3]))			#training data for base network 1
		q2_train.append(process_sentence(row[4]))			#training data for base network 2
		label_train.append(process_sentence(row[5]))			#training labels

	fp = csv.reader(open(testfile,"r"))
	q1_test = []
	q2_test = []
	label_test = []

	count = 0 
	for row in fp:
		if count == 0:
			count = 1
			continue
		q1_test.append(process_sentence(row[3]))			#testing data for base network 1
		q2_test.append(process_sentence(row[4]))			#testing data for base network 2
		label_test.append(process_sentence(row[5]))			#testing labels

	#creating dictionary of words
	vocab = set()
	max_len = -1

	for item in [q1_train, q2_train, q1_test, q2_test]:	
		for kitem in item:
			max_len = np.max([max_len, len(kitem)])
			for jitem in kitem:
				vocab.add(jitem)
	#converting words to indices
	word_2_idx = {}
	idx_2_word = {}

	count = 0
	for word in vocab:
		count += 1
		word_2_idx[word] = count
		idx_2_word[count + 1] = word
	
	print "Num words in vocab: ", len(vocab)
	print "Max question len: ", max_len

	nq1_train 	= convert_to_num(q1_train, word_2_idx, max_len)
	nq2_train	= convert_to_num(q2_train, word_2_idx, max_len)
	nq1_test 	= convert_to_num(q1_test, word_2_idx, max_len)
	nq2_test 	= convert_to_num(q2_test, word_2_idx, max_len)

	label_train 	= np.array(label_train,dtype=np.int32)
	label_test 	= np.array(label_test,dtype=np.int32)
	
	return nq1_train, nq2_train, nq1_test, nq2_test, label_train,label_test, max_len, vocab
