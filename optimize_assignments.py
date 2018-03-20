from __future__ import division
import numpy as np
from pyqpbo import alpha_expansion_general_graph
from munkres import Munkres
import util

def create_feature_vector(matrix, enc):
    #create an array representing the feature vector
    #m + 1 features
    #n(n-1)/2 edges
    if enc == 'w1' or enc == 'w2':
        if enc ==  'w1':
            thresh = 0.4
        else:
            thresh = 0.96
        ## Subtract the mean for each feature
        ## Shorten matrix to include only non zero columns
        means = np.mean(matrix, axis = 0)
        matrix = matrix - means
        matrix = np.where(matrix > thresh, 1, 0)
        zeros = np.nonzero(np.sum(matrix, axis = 0))
        print zeros
        matrix = matrix[:, zeros[0]]
    
    n, m = matrix.shape
    phi = np.zeros((int(n * (n - 1) / 2), m))
    l = 0
    for i in range(n):
        for j in range(i+1, n):
            phi[l] = - abs(matrix[i] - matrix[j])
            phi[l][0] = 1
            l += 1

    phi = phi.T

    return phi, matrix

def create_params(K, featMatrix):
    n, m = featMatrix.shape
    theta = np.zeros((K,m)).astype(np.float64)
    ## for each k, randomly set one feature to 1
    ## as initialization, set theta_0 to 1 as well
    for k in range(K):
        ind = np.random.randint(m)
        theta[k, ind] = 1
        theta[k, 0] = 1
    alpha = np.ones(K)
    return theta, alpha

def initialize(m, n):
    thetak = np.zeros(m).astype(np.float64)
    thetak[np.random.randint(m)] = 1
    thetak[0] = 1
    alphak = 1
    Ck = np.random.randint(2, size = (n,))
    return thetak, alphak, Ck 
    
def dot_product(theta, phi):
    m, n_edges = phi.shape
    k, m = theta.shape
    dp = np.matmul(theta, phi)
    return dp

def nodes_to_edges(nodes):
    n = len(nodes)
    l = 0
    edges = np.zeros((int(n * (n - 1) / 2)))
    for i in range(n):
        for j in range(i+1, n):
            edges[l] = nodes[i] * nodes[j]
            l += 1
    return edges

def get_dk(edgeClust, alpha, K):
    dk = edgeClust - (alpha.T * (1 - edgeClust).T).T
    return dk

def get_potential(K, dk, dp):
    potential = dk*dp
    potential = np.sum(potential, axis=0)
    return potential

def update_alpha(alpha, edgeClust, theta, edges, phi, K, n):

    LEARNING_RATE = 1. / float(n**2)
    dp = dot_product(theta, phi)
    dk = get_dk(edgeClust, alpha, K)
    potential = get_potential(K, dk, dp)
    prod = dp * (1 - edgeClust)

    # Get the gradient and update alpha
    grad = np.sum(prod*util.expoverexp(potential),axis=1)\
            - np.sum(edges*prod, axis = 1)
    alpha += LEARNING_RATE * grad

    return alpha
    
def update_theta(alpha, edgeClust, theta, edges, phi, K, lmda, n):
        
    LEARNING_RATE = 1. /(float(n**2))
    dp = dot_product(theta, phi)
    dk = get_dk(edgeClust, alpha, K)
    potential = get_potential(K, dk, dp)

    # Get the gradient and update theta
    grad = lmda * - np.sign(theta)

    dkpot = dk*util.expoverexp(potential)
    dkedge = dk*edges
    for k in range(K):
        decr = np.sum(dkpot[k] * phi, axis = 1)
        grad[k] += -decr
        incr = np.sum(dkedge[k] * phi, axis = 1)
        grad[k] += incr
    theta += LEARNING_RATE * grad

    return theta

def update_circles(edges, theta, alpha, phi, edgeClust, K, n):
   
    kt, n_edges = edgeClust.shape
    edge_labels = np.zeros((n_edges, 2))
    l = 0
    ## Generate edge labels for QPBO
    for i in range(n):
        for j in range(i + 1, n):
            edge_labels[l, 0] = i
            edge_labels[l, 1] = j
            l += 1
    ## Coerce to integer for Pyqpbo
    edge_labels = edge_labels.astype(np.int32)

    ## We have no node costs in this setting, only edge related costs
    unary_cost = np.zeros((int(np.max(edge_labels))+1, 2)).astype(np.int32)
    clusters = np.zeros((K, n))
    dp = dot_product(theta, phi)

    for k in np.random.permutation(K):
        dk = get_dk(edgeClust, alpha, K)
        dkdp = dk*dp
        ok = np.sum(dkdp, axis = 0) - dkdp[k]
        ## Calculate the Energies
        exp11 = ok + dp[k]
        exp00 = ok - (alpha[k] * dp[k])
        
        ## Energy matrix if they are edges in the graph
        eEdge00 = -exp00 + util.logoneplusexp(exp00)
        eEdge01 = eEdge00
        eEdge10 = eEdge00
        eEdge11 = -exp11 + util.logoneplusexp(exp11)

        ## Energy matrix if they are not edges in the graph
        nEdge00 = util.logoneplusexp(exp00)
        nEdge01 = nEdge00
        nEdge10 = nEdge00
        nEdge11 = util.logoneplusexp(exp11)

        energy00 = edges*eEdge00 + (1-edges)*nEdge00
        energy01 = edges*eEdge01 + (1-edges)*nEdge01
        energy10 = edges*eEdge10 + (1-edges)*nEdge10
        energy11 = edges*eEdge11 + (1-edges)*nEdge11
        
        edges_cost = np.zeros((n_edges, 2, 2))
        edges_cost[:, 0, 0] = energy00
        edges_cost[:, 0, 1] = energy01
        edges_cost[:, 1, 0] = energy10
        edges_cost[:, 1, 1] = energy11

        ## Costs must be integers in Pyqpbo, multiply by a large
        ## constant to not lose precision
        edges_cost *= 10000
        edges_cost = edges_cost.astype(np.int32)
        clusters[k] = alpha_expansion_general_graph(edge_labels, unary_cost, edges_cost, n_iter = 5)
        edgeClust[k] = nodes_to_edges(clusters[k])
    return clusters, edgeClust

def compute_log_likelihood(edges, theta, alpha, phi, edgeClust, K):
    dp = dot_product(theta, phi)
    dk = get_dk(edgeClust, alpha, K)
    potential = get_potential(K, dk, dp)
    Z = util.logoneplusexp(potential)
    return np.sum(potential*edges) - np.sum(Z)

def evaluate_BER(predicted, truth):
    C1, n1 = predicted.shape
    C2, n2 = truth.shape
    matrix = []
    ## If any predicted circle as all nodes, or no node
    ## We cannot calculate BER
    if 0 in np.sum(predicted, axis = 1) or 0 in np.sum(1 - predicted, axis = 1):
        return 0
    for c1 in range(C1):
        circList = []
        for c2 in range(C2):
            FN = np.where(predicted[c1]-truth[c2]==1, 1, 0)
            FP = np.where(truth[c2]-predicted[c1]==1, 1, 0)
            FNR = np.sum(FN) / np.sum(predicted[c1])
            FPR = np.sum(FP) / np.sum(1 - predicted[c1])
            circList.append(0.5*(FNR + FPR))
        matrix.append(circList)
    m = Munkres()
    ## Get minimum cost assignment
    indexes = m.compute(matrix)
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value / min(C2, C1)
    print 'BER: {}'.format(1 - total)
    return 1 - total
