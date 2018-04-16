'''Train CIFAR10 using ChebNet.'''
from __future__ import print_function

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision

from torch.autograd import Variable
from torch.autograd import Function
from torchvision.transforms import ToTensor

import os
import argparse

from utils import progress_bar

parser = argparse.ArgumentParser(description='CIFAR10 ChebNet Training')
parser.add_argument('--graph', '-g', default='cifar10_cov_4closest_symmetrized', help='path of graph file')
parser.add_argument('--lr', '-l', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch', '-b', default=32, type=int, help='batch size')
parser.add_argument('--epochs', '-e', default=150, type=int, help='epochs to train')
parser.add_argument('--clip', '-c', default=-1.0, type=float, help='gradient clipping value if > 0')
parser.add_argument('--k', '-k', default=5, type=int, help='polynomial orders')
parser.add_argument('--adam', '-a', action='store_true', help='ude Adam instead of SGD')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

class my_sparse_mm(Function):
    
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y
    
    def backward(self, grad_output):
        W, x = self.saved_tensors 
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t()) 
        grad_input_dL_dx = torch.mm(W.t(), grad_input )
        return grad_input_dL_dW, grad_input_dL_dx

def rescale_L(L, lmax=2):
    """Rescale Laplacian eigenvalues to [-1,1]"""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax * 2
    L -= I
    return L 

def lmax_L(L):
    """Compute largest Laplacian eigenvalue"""
    return eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]

class ChebConv(nn.Module):

    def __init__(self, Fin, Fout, L, K, bias=False):
        super(ChebConv, self).__init__()
        self.L = L
        self.Fin = Fin
        self.Fout = Fout
        self.K = K
        self.bias = bias
        
        self.cl = nn.Linear(K*Fin, Fout, bias)
        scale = np.sqrt( 2.0/ (Fin+Fout) )
        self.cl.weight.data.uniform_(-scale, scale)
        if self.bias:
            self.cl.bias.data.fill_(0.0)

    def forward(self, x):
        #(x, cl, L, lmax, Fout, K):

        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size

        x = x.permute(0,2,1).contiguous()
        B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin) 

        # rescale Laplacian
        lmax = lmax_L(self.L)
        L = rescale_L(self.L, lmax) 
        
        # convert scipy sparse matric L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T 
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data) 
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = Variable( L , requires_grad=False)
        if torch.cuda.is_available():
            L = L.cuda()
        
        # transform to Chebyshev basis
        x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B
        
        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B  
             
        if self.K > 1: 
            x1 = my_sparse_mm()(L,x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)),0)  # 2 x V x Fin*B
        for k in range(2, self.K):
            x2 = 2 * my_sparse_mm()(L,x1) - x0  
            x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B
            x0, x1 = x1, x2  
        
        x = x.view([self.K, V, Fin, B])           # K x V x Fin x B     
        x = x.permute(3,1,2,0).contiguous()       # B x V x Fin x K
        x = x.view([B*V, Fin*self.K])             # B*V x Fin*K
        
        # Compose linearly Fin features to get Fout features
        x = self.cl(x)                            # B*V x Fout  
        x = x.view([B, V, self.Fout])             # B x V x Fout
        x = x.permute(0, 2, 1).contiguous()
        
        return x

def read_graph(filename):
    f = open(filename,"r")
    lines = f.readlines()
    for index,line in enumerate(lines):
        if index == 0:
            N = int(line)
            A = np.zeros((N,N))
            print(line)
        else:
            splitted = line.replace("\n","").split(" ")
            for value in splitted:
                value = int(value)
                A[index-1,value] = 1
    A = scipy.sparse.csr_matrix(A)
    return A

class ChebNet(nn.Module):
    def __init__(self, graphPath, K):
        super(ChebNet, self).__init__()
        self.graph = graphPath
        self.L = read_graph(graphPath)
        self.K = K
        self.n = 32*32
        self.c = 10
        self.inference = nn.Sequential(
                ChebConv(3, 96, self.L, self.K),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                
                ChebConv(96, 96, self.L, self.K),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                
                ChebConv(96, 96, self.L, self.K),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                ChebConv(96, 192, self.L, self.K),
                nn.BatchNorm1d(192),
                nn.ReLU(inplace=True),
                
                ChebConv(192, 192, self.L, self.K),
                nn.BatchNorm1d(192),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                ChebConv(192, 192, self.L, self.K),
                nn.BatchNorm1d(192),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                
                ChebConv(192, 96, self.L, self.K),
                nn.BatchNorm1d(96),
                nn.ReLU(inplace=True),
                nn.Dropout()
            )
        self.classifier = nn.Linear(self.n * 96, self.c)

    def forward(self, x):
        x = x.view((x.size(0), 3, self.n))
        x = self.inference(x)
        x = x.view((x.size(0), -1))
        x = self.classifier(x)
        return x

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
perm = None
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = ChebNet(args.graph, args.k)

# Data
print('==> Preparing data..')
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
if args.adam:
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        #inputs = reorder(perm, inputs)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    #if acc > best_acc:
    if True:
        #print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
