import numpy as np
import ast
from scipy.sparse import csr_matrix

def create_x(subject, session):
    file = 'data/subject_' + str(subject) + '/session_' + str(session) + '/fmri.data'
    x = ast.literal_eval(open(file,'r').read())
    return np.array(x, dtype='float64').squeeze()

categories = ['scissors','face','cat','shoe','house','scrambledpix','bottle','chair']

def encode(cat):
    return categories.index(cat)

def oneHotEncode(cat):
    res = np.zeros(len(categories))
    res[encode(cat)] = 1
    return res

def create_y(subject, session, one_hot_encode=False):
    file = 'data/subject_' + str(subject) + '/session_' + str(session) + '/y.data'
    y = ast.literal_eval(open(file,'r').read())
    if one_hot_encode:
        return np.array([oneHotEncode(it[0]) for it in y], dtype='uint8')
    return np.array([encode(it[0]) for it in y], dtype='uint8')

def x_train(subject):
    res = np.array([create_x(subject, session) for session in range(10)])
    a, b, c = res.shape
    return res.reshape((a*b,c))

def x_test(subject):
    res = np.array([create_x(subject, session) for session in range(10,12)])
    a, b, c = res.shape
    return res.reshape((a*b,c))

def y_train(subject):
    res = np.array([create_y(subject, session) for session in range(10)])
    if len(res.shape) == 3:
        a, b, c = res.shape
        return res.reshape((a*b,c))
    a, b = res.shape
    return res.reshape((a*b))

def y_test(subject):
    res = np.array([create_y(subject, session) for session in range(10,12)])
    if len(res.shape) == 3:
        a, b, c = res.shape
        return res.reshape((a*b,c))
    a, b = res.shape
    return res.reshape((a*b))

def load_dataset(subject):    
    return (x_train(subject), y_train(subject)), (x_test(subject), y_test(subject))

def load_graph(path):
    lines = open(path, 'r').readlines()
    n = int(lines[0].strip())
    res = csr_matrix((n,n), dtype='float32')
    for i in range(n):
        neib = [int(v) for v in lines[i+1].strip().split(' ')]
        res[i,neib] = 1.
        res[neib,i] = 1.
    return res