The whole code has been written as one part in Jupyter notebook

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import fnmatch
import os
import numpy as np
import matplotlib.cm as cm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
import glob

Training_Digit_Path ="/home/suma/Desktop/digits/trainingDigits/"
Test_Digit_Path = "/home/suma/Desktop/digits/testDigits/"

l=0
k =0
p =0
count =0
training_vector = np.zeros((1934,1024))
test_vector = np.zeros((946,1024))
fig = plt.figure(figsize = (10,10))
rows = 2
columns = 5
i = 1

# Training Set Vector Formation
files_1 = sorted(os.listdir(Training_Digit_Path))
for file in files_1:
    if fnmatch.fnmatch(file, '*.txt'):
        with open(Training_Digit_Path + file, 'r') as f:
            for l in range(0,32):
                data = f.readline()
                for k in range(0,32):
                    training_vector[count, 32*l + k] = int(data[k]) 
        #Displaying the training vector as a binary digit
        if fnmatch.fnmatch(file, '*_0.txt'):
                fig.add_subplot(rows,columns,i)
                plt.imshow(np.array(training_vector[count]).reshape(32,32), cmap = cm.gray)
                plt.tight_layout()
                i += 1
        count += 1

# Labeling the Training Data
training_label = np.zeros((len(files_1),1))
p=0
for file in files_1:
    training_label[p,0]=int(files_1[p].split('_')[0])
    p+=1
training_label=training_label.ravel()

#Testing set vector formation
files2= glob.glob(Test_Digit_Path + '*.txt')
l=0
k=0
count=0
for file in files2:
    with open(file) as f:
        for l in range(0,32):
            data=f.readline()
            for k in range(0,32):
                test_vector[count, 32*l + k]=int(data[k])
        count+=1

#Labeling the testing data
files_2 = os.listdir(Test_Digit_Path)
testing_label = np.zeros((len(files_2),1))
p=0
for file in files_2:
    testing_label[p,0]=int(files_2[p].split('_')[0])
    p+=1
    
testing_label = testing_label.ravel()

#########################  Question 2 #########################

################## Bernoulli Naive Bayes Classifier #########

NBC_Model = BernoulliNB()
training_prediction = NBC_Model.fit(training_vector, training_label).predict(training_vector)

error_train_1 = (training_label != training_prediction).sum()
errorrate_train_1 = error_train_1 / 1934

print('Training Data: ')
print('No. of errors = %d' %(error_train_1))
print('Error Rate = %f' %(errorrate_train_1))
print('\n')

test_prediction = NBC_Model.fit(training_vector, training_label).predict(test_vector)
error_test_1 = (testing_label != test_prediction).sum()
errorrate_test_1 = error_test_1 / 946

print('Testing Data: ')
print('No. of errors = %d' %(error_test_1))
print('Error Rate = %f' %(errorrate_test_1))
print('\n')

##############  Gaussian Naive Bayes classifier ###########

NBC_Model = GaussianNB()
training_prediction = NBC_Model.fit(training_vector, training_label).predict(training_vector)

error_train_1 = (training_label != training_prediction).sum()
errorrate_train_1 = error_train_1 / 1934

print ('Gaussian Naive Bayes classifier')
print('\n')

print('Training Data: ')
print('No. of errors = %d' %(error_train_1))
print('Error Rate = %f' %(errorrate_train_1))
print('\n')

test_prediction = NBC_Model.fit(training_vector, training_label).predict(test_vector)
error_test_1 = (testing_label != test_prediction).sum()
errorrate_test_1 = error_test_1 / 946

print('Testing Data: ')
print('No. of errors = %d' %(error_test_1))
print('Error Rate = %f' %(errorrate_test_1))
print('\n')

##############  Multinomial Naive Bayes classifier ###########

NBC_Model = MultinomialNB()
training_prediction = NBC_Model.fit(training_vector, training_label).predict(training_vector)

error_train_1 = (training_label != training_prediction).sum()
errorrate_train_1 = error_train_1 / 1934

print ('Multinomial Naive Bayes classifier')
print('\n')

print('Training Data: ')
print('No. of errors = %d' %(error_train_1))
print('Error Rate = %f' %(errorrate_train_1))
print('\n')

test_prediction = NBC_Model.fit(training_vector, training_label).predict(test_vector)
error_test_1 = (testing_label != test_prediction).sum()
errorrate_test_1 = error_test_1 / 946

print('Testing Data: ')
print('No. of errors = %d' %(error_test_1))
print('Error Rate = %f' %(errorrate_test_1))
print('\n')

################################  Question 3 ######################
##################### K nearest neighbor classifier ##############

xaxis = range(1,11)
yaxis = []

for m in range(1,11):
    KNN_Model = KNeighborsClassifier(n_neighbors = m)
    knn_testing_prediction = KNN_Model.fit(training_vector, training_label).predict(test_vector)
    
    error_test_2 = (testing_label != knn_testing_prediction).sum()
    errorrate_test_2 = error_test_2 / 946
    
    print('%d: Error Rate = %f' %(m, errorrate_test_2))
    yaxis.append(errorrate_test_2)
yaxis = np.array(yaxis)

plt.figure()
plt.xlabel("k-neighbors")
plt.ylabel("error rate") 
plt.plot(xaxis, yaxis)

num = np.linspace(100,1800,18)
sample_digit = []

for s in range(0,10):
    count =0
    for number in training_label:
        if number == s:
            count += 1
    sample_digit.append(count)
    
print('\n')
print ('Counting the number of samples of each digit in the Training set')
print(sample_digit)
print ('\n')

Bayes_training_error = []
Bayes_testing_error = []
KNN_training_error_1 = []
KNN_testing_error_1 = []

for training_samples in num:
    
    x_training = []
    y_training = []
    num_training_samples = int(training_samples/10)
    
    # Digit 0
    slice = training_vector[0:num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[0:num_training_samples]
    y_training.append(slice_label)
    
    # Digit 1
    slice = training_vector[188:188 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[188: 188 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 2
    slice = training_vector[387:387 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[387: 387 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 3
    slice = training_vector[582:582 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[582: 582 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 4
    slice = training_vector[781:781 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[781: 781 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 5
    slice = training_vector[967:967 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[967:967 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 6
    slice = training_vector[1154:1154 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[1184: 1184 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 7
    slice = training_vector[1355:1355 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[1355:1355 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 8
    slice = training_vector[1535:1535 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[1535: 1535 + num_training_samples]
    y_training.append(slice_label)
    
    # Digit 9
    slice = training_vector[1739:1739 + num_training_samples]
    for i in range(0, num_training_samples):
        x_training.append(slice[i])
    slice_label = training_label[1739: 1739 + num_training_samples]
    y_training.append(slice_label)
    
    x_training = np.array(x_training)    
    y_training = np.array(y_training)
    
    y_training = y_training.reshape(1, 10*num_training_samples)
    y_training = y_training.flatten()
 
    NBC_Model = GaussianNB()
    train_pred = NBC_Model.fit(x_training, y_training).predict(x_training)
    train_error = (y_training != train_pred).sum()/x_training.shape[0]
    Bayes_training_error.append(train_error)
    
    NBC_Model = GaussianNB()
    test_pred = NBC_Model.fit(x_training, y_training).predict(test_vector)
    test_error = (testing_label != test_pred).sum()/test_vector.shape[0]
    Bayes_testing_error.append(test_error)
    
    KNN_Model_1 = KNeighborsClassifier(n_neighbors = 1)
    training_knn_pred = KNN_Model_1.fit(x_training, y_training).predict(x_training)
    error = (y_training != training_knn_pred).sum()/len(y_training)
    KNN_training_error_1.append(error)
    
    test_knn_pred = KNN_Model_1.fit(x_training, y_training).predict(test_vector)
    error = (testing_label != test_knn_pred).sum()/len(test_vector)
    KNN_testing_error_1.append(error)
    
    
plt.figure(figsize = (8,4))
plt.plot(num, Bayes_training_error, "b", label = 'Bayes training error')
plt.plot(num, Bayes_testing_error, "r", label = 'Bayes testing error')
plt.xlabel("number of training points")
plt.ylabel("error rate")
plt.legend()
plt.plot(xaxis, yaxis)

plt.figure(figsize = (8,4))
plt.plot(num, KNN_training_error_1, "b", label = '1NN training error')
plt.plot(num, KNN_testing_error_1, "r", label = '1NN test error')
plt.xlabel("number of training points")
plt.ylabel("error rate")
plt.legend()
plt.plot(xaxis, yaxis)
