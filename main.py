from __future__ import division
import numpy as np
import os
import optimize_assignments as oa
import copy
import csv
import pandas as pd
import sys

def read_features(network):
    #Create feature vector
    feats = open("{}.feat".format(str(network)))
    node_list = []
    featMatrix = None
    for node in feats.readlines():
        info = node.split()
        for ind, x in enumerate(info):
            if ind == 0:
                node_list.append(int(x))
                featVec = np.ones(1)
                continue
            featVec = np.append(featVec, int(x))
        if featMatrix is None:
            featMatrix = featVec
        else:
            featMatrix = np.c_[featVec, featMatrix]

    #organize the matrix
    node_list = np.array(node_list)
    indices = np.argsort(node_list)
    node_list = node_list[indices]
    for i in range(featMatrix.shape[0]):
        featMatrix[i] = featMatrix[i][indices]

    featMatrix = np.transpose(featMatrix)
    return node_list, featMatrix


## Get the ground truth assignment of circles
def read_circles(network, node_list):
    mapping = dict()
    for i, x in enumerate(node_list):
        mapping[x] = i
    circ = open("{}.circles".format(str(network)))
    circles = None
    for line in circ.readlines():
        memb = line.split()
        circlist = np.zeros(len(node_list))
        for ind, x in enumerate(memb):
            if ind != 0:
                circlist[mapping[int(x)]] = 1
        if circles is None:
            circles = circlist
        else:
            circles = np.c_[circles, circlist]
    return circles.T

def run_algorithm(network, kval, enc):
    print 'Running Algorithm: network = {}, K = {}, Encoding = {}'.format(network, kval, enc)
    node_list, featMatrix = read_features(network)
    circles = read_circles(network, node_list)
    n = len(node_list)
    
    ## Create reference vector, the keys are the edges
    ## the values are the position in the vector storing edges
    reference = {}
    l = 0
    for i in range(n):
        for j in range(i+1, n):
            reference[(i,j)] = l
            l += 1

    # Create edge matrix
    f = open("{}.edges".format(str(network)))
    edges = np.zeros(int(n * (n - 1) / 2))
    for edge in f.readlines():
       a, b = edge.split()
       a = np.where(node_list==int(a))[0][0]
       b = np.where(node_list==int(b))[0][0]
       if a < b:
           edges[reference[(a, b)]] = 1
       else:
           edges[reference[(b, a)]] = 1
    
    #create feature vector
    phi, featMatrix = oa.create_feature_vector(featMatrix, enc)
    ####################################################
    #### Run the Algorithm #############################
    ####################################################
   
    #Set Hyperparameters
    i = 0
    K = kval
    lmda = 100
    gradientReps = 50
    iterations = 25

    ## Collect Data 
    data = {}
    data['Iteration'] = []
    data['LL'] = []
    data['BER'] = []
    for k in range(K):
        data['Assn_{}'.format(k)] = []
    data['Alpha'] = []
    data['Theta'] = []

    #initialize parameter vector
    theta, alpha = oa.create_params(K, featMatrix)
    m = featMatrix.shape[1]
    clusters = np.random.randint(2, size=(K, n))
    prevResult = np.ones((K, n))
    prevLogLik = -10**98
    logLik = -10**97
    edgeClust = np.zeros((K, int(n*(n-1)/2)))

    for k in range(K):
        edgeClust[k] = oa.nodes_to_edges(clusters[k]) 
    
    data['Iteration'].append(i)
    for k, val in enumerate(np.sum(clusters, axis = 1)):
        data['Assn_{}'.format(k)].append(val)
    data['Alpha'].append(np.mean(alpha))
    data['Theta'].append(np.mean(theta))
    data['LL'].append(0)
    data['BER'].append(0)

    ## Use iterations to set an upper limit on learning,
    ## Typically converges far before this limit is reached
    for i in range(1, iterations + 1):

        print 'ITERATION: {}'.format(i)
        prevResult = clusters
        
        #Unpack results and turn them into the circles
        clusters, edgeClust = oa.update_circles(edges, theta, alpha, phi, edgeClust, K, n)
        #If assignments are unchanged, we have converged
        if np.array_equal(clusters, prevResult):
            break
        data['Iteration'].append(i)
        
        for k, val in enumerate(np.sum(clusters, axis = 1)):
            data['Assn_{}'.format(k)].append(val)
            #reinitialize circle k if it has everyone or noone
            if val == 0 or val == n:
                thetak, alphak, Ck = oa.initialize(m, n)
                theta[k] = thetak
                alpha[k] = alphak
                clusters[k] = Ck
        if np.sum(edgeClust) == 0:
            print 'RESTARTING DEGENERATE'
            theta, alpha = oa.create_params(K, featMatrix)

        oldLogLik = oa.compute_log_likelihood(edges, theta, alpha, phi, edgeClust, K)
        
        #With assignments, perform gradient ascent
        for gr in range(gradientReps):
            #store parameters
            alphaCopy = copy.copy(alpha)
            thetaCopy = copy.copy(theta)

            #Update alpha
            alpha = oa.update_alpha(alpha, edgeClust, theta, edges, phi, K, n)

            #Update theta
            theta = oa.update_theta(alphaCopy, edgeClust, theta, edges, phi, K, lmda, n)
            logLik = oa.compute_log_likelihood(edges, theta, alpha, phi, edgeClust, K)

            ## If our objective decreases, undo and stop
            if oldLogLik > logLik:
                theta = thetaCopy
                alpha = alphaCopy
                logLik = oldLogLik
                break
            oldLogLik = logLik

        data['LL'].append(logLik)
        data['Alpha'].append(np.mean(alpha))
        data['Theta'].append(np.mean(theta))
        data['BER'].append(oa.evaluate_BER(clusters, circles))

        print 'Log Likelihood: {}'.format(logLik)
        ## If our log likelihood was better in previous iteration,
        ## undo and stop
        if prevLogLik > logLik:
            break
        prevLogLik = logLik
    
    oa.evaluate_BER(prevResult, circles)
    BIC = -2*prevLogLik + (np.sum(abs(theta)) + np.sum(abs(alpha))) * np.log(np.sum(edges))
    data['BIC'] = [BIC for i in range(len(data['LL']))]
    return data

if __name__ == "__main__":
    network = int(sys.argv[1])
    k = int(sys.argv[2])
    enc = sys.argv[3]

    os.chdir(os.getcwd() + '/data/')

    network_list = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    encodings = ['phi1', 'w1', 'w2']

    if enc not in encodings or network not in network_list:
        print 'ERROR: Check arguments'
        sys.exit()

    data = run_algorithm(network, k, enc)
    df = pd.DataFrame(data)
    df.to_csv('../results/results_n{}_k{}_enc{}.csv'.format(network, k, enc), index = False)
