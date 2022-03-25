import os    
os.environ['MPLCONFIGDIR'] = "matplot-temp"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def loadAndProcess():
    print('Loading...')
    # load the data...

    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    featureNames = cancer.feature_names
    labelNames = cancer.target_names

    # we have an issue because range of features values is very wide
    # this is a problem for most machine learning algorithms, but
    # it is a major proboem for SVM
    plt.boxplot( X, showfliers = False )
    plt.yscale( "symlog" )
    plt.xlabel( "Feature index" )
    plt.ylabel ( "Feature magnitude" )
    plt.savefig(r"feature_magnitude1.png",bbox_inches='tight')
    
    return labelNames, featureNames, X, y
    
def buildTrainAndTest(X, y):
    print('Building train and test sets...')
    # create the train and test sets for X and y
    # traning has 67% of the rows and test has 33% of the rows...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 12)

    # print shape of data sets...
    print('Entire set shape= %s' % str(X.shape))
    print('Training set shape= %s' % str(X_train.shape))
    print('Test set shape= %s' % str(X_test.shape))
    
    return X_train, X_test, y_train, y_test
    
def train(X_train, y_train):
    # train the algorithm...
    print('Training...')
    svc = SVC(kernel='rbf', C=1, gamma='auto') # change c to 10 for better fit
    svc.fit(X_train, y_train)
    return svc, svc.score(X_train, y_train)
    
def test(svc, X_test , y_test):
    # test the algorithm...
    print('Testing...')
    return svc.score(X_test , y_test)

def main():
    print("Running Main...")
    labelNames, featureNames, X, y = loadAndProcess()

    # print shape...
    print('X shape: %s' % str(X.shape))
    print('y shape: %s' % str(y.shape))

    #print features and labels...
    print("Features %s" % featureNames)
    print("Labels %s" % labelNames)

    # print data...
    print('first five rows of X= \n%s' % X[0:6, :])
    print('first 150 rows of y= \n%s' % y[0:150]) 

    X_train, X_test, y_train, y_test = buildTrainAndTest(X, y)
    print("X_train = %s\n" % X_train)
    print("X_test = %s\n" % X_test)
    print("y_train = %s\n" % y_train)
    print("y_test = %s\n" % y_test)
  
    svm, score = train(X_train, y_train)
    print("Score on train data %s\n" % score)

    score = test(svm, X_test , y_test)
    print("Score on test data %s\n" % score)
    