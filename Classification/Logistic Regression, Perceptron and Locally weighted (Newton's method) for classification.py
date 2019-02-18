# Logistic Regression code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as prep

Training_path = '/home/suma/bclass/bclass-train'
Testing_path = '/home/suma/bclass/bclass-test'

def read_data(filename):
    
    with open(filename) as f:
        data = f.readlines()
        labels=[]
        features=[]
        for line in data:
            line = line[:-1]  #Removes the last character i.e. \n
            feature = line.split("\t")
            label = int(feature[0]) #1st column is label
            feature = feature[1:] #2nd column onwards it's features
            features += [feature]
            labels += [label]
        labels = np.array(labels).astype(int)
        features = np.array(features).astype(float)
            
    return features,labels

def sigmoid(t):
    return 1.0/ (1 + np.exp(-t))

def update_weights(x, labels, weights, learning_rate):
    
    scores = np.matmul(x, weights)
    predictions = sigmoid(scores)
    output_error = predictions - labels
    gradient = np.matmul(x.T, output_error)/ (len(x))
    weights = weights - learning_rate*gradient
    
    return weights


def training (x, labels, weights, learning_rate, iterations, testing_data, testing_labels, num):
    error_rate = []
    for i in range(iterations):
        weights = update_weights(x, labels, weights, learning_rate)
        predictions = sigmoid(np.matmul(testing_data, weights))
        predicted_labels = [1 if p>= 0.5 else 0 for p in predictions]
        wrong = (predicted_labels != testing_labels)
        rate = np.sum(wrong)/len(testing_labels)
        error_rate.append(rate)

    plt.figure(num)
    plt.plot(error_rate)
    plt.xlabel('Iterations')
    plt.ylabel('Error Rate')    

def logistic_plot(x, y, testing_data, testing_labels, num):
     
    weights = np.zeros((x.shape[1]))
    training(x, y, weights, 0.01, 2500,testing_data,testing_labels, num)
     
    
def logistic_regression():
    
    training_data, training_labels = read_data(Training_path)
    testing_data, testing_labels = read_data(Testing_path)

    training_labels = training_labels.clip(min=0)   
    testing_labels=testing_labels.clip(min=0)
    training_samples = len(training_labels)
    testing_samples  = len(testing_labels)  
    
    tr_data = np.insert(training_data, [0], np.ones((training_samples,1)), axis=1)
    tt_data = np.insert(testing_data, [0], np.ones((testing_samples,1)), axis=1)

    l2_tr_data = prep.normalize(tr_data, 'l2')
    l2_tt_data = prep.normalize(tt_data, 'l2')
    
    logistic_plot(tr_data, training_labels, tr_data, training_labels,1)
    logistic_plot(l2_tr_data, training_labels, l2_tr_data, training_labels, 2)
    logistic_plot(tr_data, training_labels, tt_data, testing_labels, 3)
    logistic_plot(l2_tr_data, training_labels, l2_tt_data, testing_labels, 4)
    
logistic_regression()   

# Perceptron:

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as prep

Training_path = '/home/suma/bclass/bclass-train'
Testing_path = '/home/suma/bclass/bclass-test'

def read_data(filename):
    
    with open(filename) as f:
        data = f.readlines()
        labels=[]
        features=[]
        for line in data:
            line = line[:-1]  #Removes the last character i.e. \n
            feature = line.split("\t")
            label = int(feature[0]) #1st column is label
            feature = feature[1:] #2nd column onwards it's features
            features += [feature]
            labels += [label]
        labels = np.array(labels).astype(int)
        features = np.array(features).astype(float)
            
    return features,labels

def sigmoid(t):
    return 1.0/ (1 + np.exp(-t))

def update_weights(x, y, iterations):
                 
    num_data_points = x.shape[0]
    weight = np.zeros((x.shape[1]))
    weights = []
    weights.append(weight)
    i = 0
    
    for iterr in range(iterations):
        j = 0
        mislabeled = False 
        for j in range(num_data_points):
            i = (i + 1) %num_data_points
            if(y[i] * np.inner(x[i],weight) <=0): #y*<wi,xi> < 0
                weight = weight + y[i] * x[i] #update weight, assuming no bias
                mislabeled = True
                break
        weights.append(weight)
        
        if(mislabeled == False):
            return weight, weights
    return weight, weights

def perceptron_plot(tr_data, training_labels, tt_data, testing_labels, iterations):
    
    weight, weights = update_weights(tr_data, training_labels, iterations)
    testing_data_points  = len(testing_labels)
    
    result = testing_labels * np.matmul(tt_data, weight)
    index_error = np.where(result <= 0)
    #Returns the indices of the elements that are non zero
    print('Regular perceptron error rate =', (len(index_error[0])/testing_data_points))

    weighted_average = np.sum(weights, axis=0)
    result = testing_labels * np.matmul(tt_data, weighted_average)
    index_error = np.where(result <= 0)
    print('Weighted average perceptron error rate=', (len(index_error[0])/testing_data_points))
    
    errors = []
    for w_iter in weights:
        result = testing_labels * np.matmul(tt_data, w_iter)
        index_error = np.where(result <= 0)
        errors.append(len(index_error[0]))
        
    plt.plot(errors)
    plt.ylabel("Error Count")
    plt.xlabel("iterations")
    plt.show()
    
def perceptron():
    
    training_data, training_labels = read_data(Training_path)
    testing_data, testing_labels = read_data(Testing_path)
    
    training_samples = len(training_labels)
    testing_samples  = len(testing_labels)
    
    #raw data
    tr_data = np.insert(training_data, [0], np.ones((training_samples,1)), axis=1)
    tt_data = np.insert(testing_data, [0], np.ones((testing_samples,1)), axis=1)
        
    #data that has been normalized to have unit l2 norm
    l2_tr_data = prep.normalize(tr_data, 'l2') 
    l2_tt_data = prep.normalize(tt_data, 'l2')
    
    perceptron_plot(tr_data, training_labels, tr_data, training_labels, 2500)
    perceptron_plot(l2_tr_data, training_labels, l2_tr_data, training_labels, 2500)
    perceptron_plot(tr_data, training_labels, tt_data, testing_labels, 2500)
    perceptron_plot(l2_tr_data, training_labels, l2_tt_data, testing_labels, 2500)

perceptron()

# Newton’s 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

Training_path = '/home/suma/bclass/bclass-train'
Testing_path = '/home/suma/bclass/bclass-test'

def read_data(filename):
    
    with open(filename) as f:
        data = f.readlines()
        labels=[]
        features=[]
        for line in data:
            line = line[:-1]  #Removes the last character i.e. \n
            feature = line.split("\t")
            label = int(feature[0]) #1st column is label
            feature = feature[1:] #2nd column onwards it's features
            features += [feature]
            labels += [label]
        labels = np.array(labels).astype(int)
        features = np.array(features).astype(float)
            
    return features,labels

def predict(features, weights):
    t = np.matmul(features, weights)
    return 1.0/ (1 + np.exp(-t))

def cost_function(features, yi, weights, sample_weights, given_val):
    
    num_labels = len(yi)
    h_xi = predict(features, weights)
    cost_1 = yi * np.log(h_xi)
    cost_2 = (1-yi)*np.log( 1 - h_xi)
    cost = cost_1 + cost_2
    cost = sample_weights * cost
    log_likelihood = cost.sum()/num_labels
    
    weights_1 = np.copy(weights)
    weights_1[0] = 0
    regularisation = given_val * np.matmul(weights_1.T, weights_1) 
    log_likelihood = log_likelihood - regularisation

    return log_likelihood

def normalize(data, norm=2): 
    norm = np.linalg.norm(data, norm, axis=0)
    Z = np.copy(data)
    
    for i in range(data.shape[1]):
        if norm[i] != 0:
            Z[:, i] = Z[:,i]/norm[i]
    
    return Z, norm

def classifyNewton(predicted_values):
  output = 1 if predicted_values >=0.5 else 0
  return output

def classify(predicted_values):
    output = [1 if x >=0.5 else 0 for x in predicted_values]
    return output

def updateLabels(labels):
    new_labels = np.zeros_like(labels)
    
    for i in range(len(labels)):
        if labels[i] == 1:
            new_labels[i] = 1
    return new_labels

def hessian(x, y, weights):
    val_1 = predict(x, weights)
    val_2 = predict(-x, weights)
    b = np.diagflat(val_1 * val_2)
    h = np.matmul(np.matmul(x.T, b), x)
    return h


def compute_weights(data_point, x, tau):
    samples = x.shape[0]
    sample_weights = np.zeros(samples)
    i = 0
    
    while (i < samples):
        val = np.linalg.norm(data_point - x[i])**2
        sample_weights[i] = np.exp(val*(-1/(2*tau**2)))
        i += 1
        
    return sample_weights

def newton_update(features, labels, weights, sample_weights, given_val):
    N = features.shape[1]
    identity_mat = np.eye(N)
    identity_mat[0,0] = 0
    weight = np.copy(weights)
    weight[0] = 0

    yi = predict(features, weights) 
    g =  ((-1 * np.matmul(features.T, sample_weights *(yi - labels))) - (2*given_val*weight))     
    h = ((-1 * hessian(features, labels, weights)) - (2*given_val*identity_mat))
    weights = weights - (np.matmul(np.linalg.pinv(h), g)) #Newton's update

    return weights

def newtons_algorithm(features, labels, weights, iterations, test_point, given_val, tau):
    costHistory = []
    weightHistory = []
    
    sample_weights = compute_weights(test_point, features, tau)

    for i in range(iterations):
        weights = newton_update(features, labels, weights, sample_weights, given_val)
        cost = cost_function(features, labels, weights, sample_weights, given_val)
        costHistory.append(cost)
        w = np.copy(weights)
        weightHistory.append(w)
                    
    return costHistory, weightHistory

def calculate_error_rate(r,s):
    diff = r - s
    return (float(np.count_nonzero(diff)) / len(diff))

def loclogregnewt(x,y,x_test,y_test, iterations, given_val, tau):
    w = np.zeros((x.shape[1]))
    num_test_samples = len(y_test)
    final_result = np.ones(num_test_samples)
    
    for i in range(num_test_samples):
        w = np.zeros((x.shape[1]))
        costHistory, weightHistory = newtons_algorithm(x, updateLabels(y), w, iterations, x_test[i], given_val, tau)
        result = classifyNewton(predict(x_test[i], w))
        final_result[i] = result
        
        errorArray = []
        for w_iter in weightHistory:
            training_result = classify(predict(x, w_iter))
            err = calculate_error_rate(training_result, updateLabels(y))
            errorArray.append(err)
    
        errorRate = calculate_error_rate(final_result, updateLabels(y_test))
    return errorRate, errorArray


def local_logistic():  
    
    training_data, training_labels = read_data(Training_path)
    testing_data, testing_labels = read_data(Testing_path)
    
    training_samples = len(training_labels)
    testing_samples  = len(testing_labels)
    tr_data = np.insert(training_data, [0], np.ones((training_samples,1)), axis=1)
    tt_data = np.insert(testing_data, [0], np.ones((testing_samples,1)), axis=1)
    
    l2_tr_data = preprocessing.normalize(tr_data, 'l2') 
    l2_tt_data = preprocessing.normalize(tt_data, 'l2')
    
    iterations =100
    given_val = 0.001
    tau_array = [0.01, 0.05, 0.1, 0.5, 1, 5]
    testing_error = np.zeros(6)
    training_error = np.zeros(6)
    
    for i in range(len(tau_array)):
        training_error[i], TrainErrorArray = loclogregnewt(tr_data, training_labels, tr_data, training_labels, iterations, given_val, tau_array[i]) #Train
        testing_error[i], TestErrorArray = loclogregnewt(tr_data, training_labels, tt_data, testing_labels, iterations, given_val, tau_array[i]) #Test
        print('tau=%f: Training Error:%f, Testing Error:%f' %(tau_array[i], training_error[i], testing_error[i]))
    
    plt.plot(training_error)
    plt.plot(testing_error)
    plt.ylabel('error_rate raw data')
    plt.xlabel('tau')
    plt.show()
    
    print('Best tau=%f for raw data' %(tau_array[np.argmin(testing_error)]))
     
local_logistic()

