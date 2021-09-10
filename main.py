import pandas as pd
import numpy as np
import random
import Node

#Dictionary that contains pair of index-attribute. It has to be modified for every different data set
attribute_index = {1:"a1", 2:"a2", 3:"a3", 4:"a4", 5:"a5", 6:"a6", 7:"a7", 8:"a8", 9:"a9", 10:"a10", 11:"a11",
12:"a12", 13:"a13", 14:"a14", 15:"a15", 16:"a16"}

#first row must be attribute names
#loads data and "cleans" it (only deletes rows with missing values)  
def load_data(filename):
	data = pd.read_csv(filename, header=0, na_values = "?")
	clean_data = data.dropna(how="any")
	return clean_data

#loads data but instead of discard missing values for them
def load_data_missing_values(filename):
	#aux contains the dataframe without missing values so we use it to calculate mean or searching the most common item etc...
	data = pd.read_csv(filename, header=0, na_values = "?")
	for column in data:
		data[column] = data[column].fillna(data[column].mode()[0])
	return data

#discretize in quartiles (4) all continuous columns based on given indexs
def discretizeContValues(dataframe, cont_columns):
	data = dataframe
	for index in cont_columns:
		data[attribute_index[index]] = pd.qcut(data[attribute_index[index]].rank(method="first"), 4)
	return data

#calculates the entropy
def entropy(selected_column):

	entropy = 0

	#value contains every different value and value_count the number of times every different value is repeated
	value, value_count = np.unique(selected_column,return_counts = True) 

	for i in range(len(value_count)):
		entropy += (-value_count[i]/np.sum(value_count))*np.log2(value_count[i]/np.sum(value_count))

	return entropy

#calculate the gain (ID3 algotithm)
def gain(data, target_attribute, selected_attribute):

	# S_entropy is the entropy of the whole dataset (S is the name given in thery pdfs), SA_entropy is Entropy(S,A)
	S_entropy = entropy(data[target_attribute])
	SA_entropy = 0

	#we calculate the different values and counts of the selected_attribute to calculate the gain
	value, value_count = np.unique(data[selected_attribute], return_counts=True)

	#we calculate the Entropy(S,A)
	for i in range(len(value_count)):
		SA_entropy += (value_count[i]/np.sum(value_count)) * entropy(data.where(data[selected_attribute] == value[i]).dropna()[target_attribute])

	return S_entropy - SA_entropy

#calculates the gain ratio (C4.5 algorithm)
def gainRatio(data, target_attribute, selected_attribute):

	#first we get the gain
	gain_ = gain(data, target_attribute, selected_attribute)

	#second we calculate the split info
	value, value_count = np.unique(data[selected_attribute], return_counts=True)
	split_info = 0

	for i in range(len(value_count)):
		split_info += (value_count[i]/np.sum(value_count)) * np.log2((value_count[i]/np.sum(value_count)))

	if split_info == 0:
		return 0
	else:
		return -(gain_/split_info)

#calculates the gini value of the target (whole dataset)
def gini(data, target_attribute):

	#get the different values of the target attribute and its counts
	value, value_count = np.unique(data[target_attribute], return_counts=True)
	
	#now we calculate gini's value
	gini = 0
	for i in range(len(value_count)):
		gini += ((value_count[i]/np.sum(value_count))*(value_count[i]/np.sum(value_count)))

	return (1 - gini)

#calculates the gini gain for an attribute
def giniGain(data, target_attribute, selected_attribute):

	#first we get the gini of the target
	S_gini = gini(data, target_attribute)

	#get the different values of the selected attribute
	value, value_count = np.unique(data[selected_attribute], return_counts=True)
	SA_gini = 0

	for i in range(len(value_count)):
		SA_gini += ((value_count[i]/np.sum(value_count)) * gini(data.where(data[selected_attribute] == value[i]).dropna(), target_attribute))

	return S_gini - SA_gini

#generates the tree
def tree_growing(data, target_attribute, selection_criteria, depth):
	
	tree = Node.Node(np.unique(df[target_attribute]))
	a, a_counts = np.unique(df[target_attribute], return_counts=True)

	#if first iteration, this node is root
	if depth==0:
		tree.root = True
	
	#updates Node.p with new probs 
	for a_v in data.iloc[:,15].value_counts().index: 
		tree.prbs[a_v] += data.iloc[:,15].value_counts()[a_v]

	#checks the 3 stop criterions we have if we stop, this node is leave and we give the class atrib
	if len(np.unique(data[target_attribute]))==1 or len(data)==0 or depth==len(data.columns)-1:
		tree.leave = True
		tree.name = target_attribute
		tree._class = a[np.argmax(a_counts)] 

	#we can proceed
	else:
		attributes = data.columns[:-1]
		values = []
		
		#select criteria
		for attr in attributes:

			if selection_criteria == 'Gini':
				values.append(giniGain(data, target_attribute, attr))
			
			elif selection_criteria == 'C45':
				values.append(gainRatio(data, target_attribute, attr))

			elif selection_criteria == 'ID3':
				values.append(gain(data, target_attribute, attr))

		best_attribute = attributes[np.argmax(values)]
		depth += 1
		tree.name = best_attribute

		for val in data[best_attribute].unique():
			subtree = tree_growing(data[data[best_attribute] == val], target_attribute, selection_criteria, depth)
			tree.children.append(subtree)
			subtree.parent = tree
			subtree.attr_value = val

	return tree

#predicts the class of the given test_data based on the tree
def predict(tree,testing_data):
	
	prediction=[]

	for entry in range(len(testing_data)):

		#search for probs
		p=tree.search(testing_data.iloc[entry])

		#calculate the highest prob and append to predictions array
		aux_max = 0
		result=None

		for ps in p:

			if p[ps] > aux_max:

				result = ps
				aux_max = p[ps]

		prediction.append(result)

	return prediction

#k-fold cross-validation method
def cross_validation(data, k, selection_criteria):
	#we get the int part of the offset we have to apply
	offset = int(len(data)/k)
	acc = []
	prec = []
	rec = []
	TP = 0
	TN = 0
	FP = 0
	FN = 0

	for i in range(k):
		
		#in the first case we get 	
		if i == 0:
			test_data = data.head(n=offset)
			train_data = data.tail(n=(len(data)-offset))
		
		#last the e 
		elif i == k-1:
			test_data = data.iloc[i*offset:]
			train_data = data.iloc[:i*offset]

		#all the other cases
		else:
			test_data = data.iloc[i*offset:(i*offset)+offset]
			
			train_data_1 = data.iloc[:i*offset]
			train_data_2 = data.iloc[(i*offset)+offset:]
			train_data_aux = [train_data_1, train_data_2]

			train_data = pd.concat(train_data_aux)

		tree = tree_growing(train_data, "a16", selection_criteria, 0)

		predictions = predict(tree, test_data)

		ground_truth = []
		for j in test_data.iloc[:,15]:
			ground_truth.append(j)

		count = 0
		for e in range(len(predictions)):
			if predictions[e] == ground_truth[e]:
				count += 1

		for e in range(len(predictions)):
			if predictions[e] == ground_truth[e] and predictions[e] == '+':
				TP += 1
			elif predictions[e] == ground_truth[e] and predictions[e] == '-':
				TN += 1
			elif predictions[e] != ground_truth[e] and predictions[e] == '+':
				FP += 1
			elif predictions[e] != ground_truth[e] and predictions[e] == '-':
				FN += 1
		
		acc.append((TP+TN)/(TP+TN+FP+FN))
		prec.append((TP)/(TP+FP))
		rec.append(TP/(TP+FN))

	print("*-*-*-*-*-*-*-*-*-* Results *-*-*-*-*-*-*-*-*-*")
	print("The cross-fold mean accuracy for k=", k, "is:", np.mean(acc))
	print("The cross-fold mean precision for k=", k, "is:", np.mean(prec))
	print("The cross-fold mean recall for k=", k, "is:", np.mean(rec))

#Holdout validation method. I made it with the common 80/20 train/test ratio 
def holdout(data, selection_criteria):

	train_len = int(len(data)*0.8)
	test_len = int(len(data)*0.2)

	TP = 0
	TN = 0
	FP = 0
	FN = 0
	
	#if len(data) is odd always train_len+test_len < len(data) so we have to
	#add 1 to train data to equilibrate things
	if len(data)%2 == 1:
		train_len += 1
	
	training_data = pd.DataFrame()
	testing_data = pd.DataFrame()

	#assign every example randomly to test or train data until they full
	for i in range(len(data)):
		
		test_train = random.randint(0,1)
		
		#0 priorizes train but if full it goes to test
		if test_train == 0:
			#if training data not full
			if len(training_data) < train_len:
				#append example to training_data
				training_data = training_data.append(data.iloc[i], sort=False)
			else:
				#append example to test_data
				testing_data = testing_data.append(data.iloc[i], sort=False)
		
		#1 priorizes test but if full it goes to train
		else:
			if len(testing_data) < test_len:
				testing_data = testing_data.append(data.iloc[i], sort = False)
			else:
				training_data = training_data.append(data.iloc[i], sort= False)

	#for some reason pd.DataFrame.append() sorts even boolean is in false so we reorder columns
	training_data = training_data.reindex(data.columns, axis=1)
	testing_data = testing_data.reindex(data.columns, axis=1)

	tree = tree_growing(training_data, "a16", selection_criteria, 0)
	predictions = predict(tree, testing_data)
	
	ground_truth = []
	for j in testing_data.iloc[:,15]:
		ground_truth.append(j)
	
	count = 0
	for e in range(len(predictions)):
		if predictions[e] == ground_truth[e] and predictions[e] == '+':
			TP += 1
		elif predictions[e] == ground_truth[e] and predictions[e] == '-':
			TN += 1
		elif predictions[e] != ground_truth[e] and predictions[e] == '+':
			FP += 1
		elif predictions[e] != ground_truth[e] and predictions[e] == '-':
			FN += 1
	
	print("The accuracy with the holdout method is:", (TP+TN)/(TP+TN+FP+FN))
	print("The precision with the holdout method is:", (TP)/(TP+FP))
	print("The recall with the holdout method is:", TP/(TP+FN))
	
	


df = load_data_missing_values("data.csv")
df = discretizeContValues(df, [2,3,8,11,14,15])
holdout(df, "ID3")
holdout(df, "C45")
holdout(df, "Gini")
