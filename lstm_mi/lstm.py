"""Train and test LSTM classifier"""
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras import metrics
import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report,accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
import collections
import math
import pandas as pd
from scipy import interp
from datetime import datetime
from StringIO import StringIO
from urllib import urlopen
from zipfile import ZipFile
import cPickle as pickle
import os
import random
import tldextract
import csv


def get_data(): 
	"""Read data from file (Traning, testing and validation) to process"""
	data= []
	with open("traindga5.csv", "r") as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row) 
	return data

def build_binary_model(max_features, maxlen):
    """Build LSTM model for two-class classification"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop')

    return model

def build_multiclass_model(max_features, maxlen):
    """Build multiclass LSTM model for multiclass classification"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(38))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop')

    return model


def create_class_weight(labels_dict,mu):
    """Create weight based on the number of domain name in the dataset"""
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.pow(total/float(labels_dict[key]),mu)
	class_weight[key] = score	


    return class_weight

def classifaction_report_csv(report,precision,recall,f1_score,fold):
    """Generate the report to data processing"""
    with open('classification_report_cost.csv', 'a') as f:
        report_data = []
        lines = report.split('\n')
        row = {}
        row['class'] =  "fold %u" % (fold+1)
        report_data.append(row)
        for line in lines[2:44]:
            row = {}
            line = " ".join(line.split())
            row_data = line.split(' ')
            if(len(row_data)>2):
                if(row_data[0]!='avg'):
                    row['class'] = row_data[0]
                    row['precision'] = float(row_data[1])
                    row['recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['support'] = row_data[4]
                    report_data.append(row)
                else:
                    row['class'] = row_data[0]+row_data[1]+row_data[2]
                    row['precision'] = float(row_data[3])
                    row['recall'] = float(row_data[4])
                    row['f1_score'] = float(row_data[5])
                    row['support'] = row_data[6]
                    report_data.append(row)
        row = {}
        row['class'] = 'macro'
        row['precision'] = float(precision)
        row['recall'] = float(recall)
        row['f1_score'] = float(f1_score)
        row['support'] = 0
        report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(f, index = False)

def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""

    #Begin preprocessing stage
    #Read data to process
    indata = get_data()

    # Extract data and labels
    binary_labels = [x[0] for x in indata]
    X = [x[2] for x in indata]
    labels = [x[1] for x in indata]
    # Generate a dictionary of valid characters
    valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
    max_features = len(valid_chars) + 1
    maxlen = np.max([len(x) for x in X])

    # Convert characters to int and pad
    X = [[valid_chars[y] for y in x] for x in X]
    X = sequence.pad_sequences(X, maxlen=maxlen)
   
	# Convert labels to 0-1 for binary class
    y_binary = np.array([0 if x == 'legit' else 1 for x in binary_labels])
    # Convert labels to 0-37 for multi class
    valid_class = {i:indx for indx, i in enumerate(set(labels))}
    y = [valid_class[x] for x in labels]
    y = np.array(y)
    #End preprocessing stage


    #Begin two-class classification stage
    #Divide the dataset into training + holdout (80%) and testing (20%) dataset
    sss = StratifiedShuffleSplit(n_splits=nfolds, test_size=0.2, random_state=0)
    fold =0
    for train, test in sss.split(X,y_binary,y):
        print "fold %u/%u" % (fold+1, nfolds)
        fold = fold+1
        X_train, X_test, y_train, y_test, y_dga_train, y_dga_test = X[train], X[test], y_binary[train], y_binary[test], y[train], y[test]
        y_dga =[]
        X_dga =[]
        for i in range(len(y_dga_train)):
        	if y_dga_train[i]!= 20:
        		y_dga.append(y_dga_train[i])
        		X_dga.append(X_train[i])
        X_dga =np.array(X_dga)
        y_dga =np.array(y_dga)       

        #Build the model for two-class classification stage
        model = build_binary_model(max_features, maxlen)
        
        print "Training the model for two-class classification stage..."
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
        for train, test in sss1.split(X_train, y_train):
            X_train, X_holdout, y_train, y_holdout = X_train[train], X_train[test], y_train[train], y_train[test]
        
        #Create weight for two-class classification stage
        labels_dict=collections.Counter(y_train)
        class_weight = create_class_weight(labels_dict,0.1)
        best_auc = 0.0
        #20
        for ep in range(20):
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, class_weight=class_weight)
            t_probs = model.predict_proba(X_holdout)
            t_result = [0 if(x<=0.5) else 1 for x in t_probs]
            t_acc = accuracy_score(y_holdout, t_result)
            #Get the model with highest accuracy
            if t_acc > best_auc:
                best_model = model
                best_auc = t_acc
		#Save the model for two-class classification     
        #Serialize model to JSON
        model_json = best_model.to_json()
        name_file = "model_binary.json"
        name_file2 = "model_binary.h5"
        with open(name_file, "w") as json_file:
            json_file.write(model_json)
        #Serialize weights to HDF5
            best_model.save_weights(name_file2)
        print("Saved two-class model to disk")
        #End of two-class classification stage

        #Begin multiclass classification stage
        #Build the model for multiclass classification stage
        model_dga = build_multiclass_model(max_features, maxlen)
        print "Training the model in multiclass classification..."
        
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
        for train, test in sss2.split(X_dga, y_dga):
            X_train, X_holdout, y_train, y_holdout = X_dga[train], X_dga[test], y_dga[train], y_dga[test]
        
        # Caculate class weight 
        labels_dict=collections.Counter(y_train)
        class_weight = create_class_weight(labels_dict,0.3)
        print 'Class weight for multiclass classification:'
        print class_weight
        best_acc = 0.0
        #20
        for ep in range(20):
            model_dga.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, class_weight=class_weight)
            y_pred= model_dga.predict_proba(X_holdout)
            y_result = [np.argmax(x) for x in y_pred]  
            t_acc= accuracy_score(y_holdout, y_result)
            if t_acc > best_acc:
                best_model_dga = model_dga
                best_acc = t_acc
        #Save the model for multiclass classification stage
        model_json = best_model_dga.to_json()
        name_file = "model_multi.json"
        name_file2 = "model_multi.h5"
        with open(name_file, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
            best_model_dga.save_weights(name_file2)
        print("Saved multiclass model to disk")

        y_pred = best_model.predict_proba(X_test)
        y_result = [20 if(x<=0.5) else 0 for x in y_pred]
        
        X_dga_test =[]
        y_dga_test_labels =[]
        for i in range(len(y_result)):
            if y_result[i] == 0:
                X_dga_test.append(X_test[i])
                y_dga_test_labels.append(y_dga_test[i])
        X_dga_test = np.array(X_dga_test)
        y_dga_test_labels = np.array(y_dga_test_labels)
        y_pred_dga = best_model_dga.predict_proba(X_dga_test)
        y_result_dga = [np.argmax(x) for x in y_pred_dga]
        #End of multiclass classification stage

        j = 0
        for i in range(len(y_result)):
            if y_result[i] != 20:
                y_result[i] = y_result_dga[j]
                j = j+1

        #Calculate the final result
        score = f1_score(y_dga_test, y_result,average="macro")
        precision = precision_score(y_dga_test, y_result,average="macro")
        recall = recall_score(y_dga_test, y_result,average="macro")
        report = classification_report(y_dga_test,y_result,digits=4)
        acc= accuracy_score(y_dga_test, y_result)
        classifaction_report_csv(report,precision,recall,score,fold)
        print '\n clasification report:\n', report
        print 'F1 score:', score
        print 'Recall:', recall
        print 'Precision:', precision
        print 'Acc:', acc
             
   
if __name__ == "__main__":
    run()
