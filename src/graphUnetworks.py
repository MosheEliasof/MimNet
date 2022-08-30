import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src import utils

def conv2(X, Kernel):
    return F.conv2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1(X, Kernel):
    return F.conv1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv1T(X, Kernel):
    return F.conv_transpose1d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def conv2T(X, Kernel):
    return F.conv_transpose2d(X, Kernel, padding=int((Kernel.shape[-1] - 1) / 2))


def tv_norm(X, eps=1e-3):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


def restrict(A):
    Ac = (A[0:-1:2, 0:-1:2] + A[1::2, 0:-1:2] + A[0:-1:2, 1::2] + A[1::2, 1::2]) / 4.0
    return Ac


def getDistMat(X):
    D = torch.sum(torch.pow(X, 2), dim=0, keepdim=True) + torch.sum(torch.pow(X, 2), dim=0,
                                                                    keepdim=True).t() - 2 * X.t() @ X

    return torch.sqrt(torch.relu(D))


def getGraphLap(X, M=torch.ones(1), sig=10):
    X = X.squeeze(0)
    M = M.squeeze()
    # normalize the data
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / (torch.std(X, dim=1, keepdim=True) + 1e-3)

    W = getDistMat(X)
    We = torch.exp(-W / sig)
    D = torch.diag(torch.sum(We, dim=0))
    L = D - We

    if torch.numel(M) > 1:
        MM = torch.ger(M, M)
        L = MM * L
        II = torch.diag(1 - M)
        L = L + II

    Dh = torch.diag(1 / torch.diag(D))

    L = Dh @ L @ Dh

    L = 0.5 * (L + L.t())
    return L


class GraphUnet(nn.Module):
    """ VNet """

    def __init__(self, nLevels, nIn, nsmooth):
        super(GraphUnet, self).__init__()
        K = self.init_weights(nLevels, nIn, nsmooth)
        self.K = K

    def init_weights(self, nL, nIn, nsmooth):
        print('Initializing network  ')

        stencil_size = 9
        K = nn.ParameterList([])
        npar = 0
        cnt = 1
        k = nIn
        stdv = 1e-3

        for i in range(nL):
            Ki = torch.zeros(k, k, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
            cnt += 1
            npar += np.prod(Ki.shape)
            K.append(Ki)

            Ki = torch.zeros(2 * k, k, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
            cnt += 1
            npar += np.prod(Ki.shape)
            K.append(Ki)
            k = 2 * k

        for i in range(nsmooth):
            Ki = torch.zeros(k, k, stencil_size)
            Ki.data.uniform_(-stdv, stdv)
            Ki = nn.Parameter(Ki)
            print('layer number', cnt, 'layer size', Ki.shape[0], Ki.shape[1], Ki.shape[2])
            cnt += 1
            npar += np.prod(Ki.shape)
            K.append(Ki)

        print('Number of parameters  ', npar)
        return K

    def forward(self, x, X, m=torch.ones(1), idxs=None, vals=None, gradFlag=False):
        """ Forward propagation through the network """

        nL = len(self.K)
        idxs_clone = idxs
        vals_clone = vals
        xS = []
        mS = [m]
        Xs = [X]
        L = getGraphLap(X)
        Ls = [L]
        for i in range(nL):
            coarsen = self.K[i].shape[0] != self.K[i].shape[1]
            if coarsen:
                xS.append(x)

            if idxs is None and gradFlag:
                grad = utils.graphGrad(x, X, k=10)
                grad, _ = torch.max(grad, dim=3)
            elif gradFlag:

                curr_idx = idxs[0]
                curr_val = vals[0]
                grad = utils.graphGrad(x, X, k=10, vals=curr_val, idx=curr_idx)
                grad, _ = torch.max(grad, dim=3)
                if coarsen:
                    idxs = idxs[1:]
                    vals = vals[1:]

            z = mS[-1] * conv1(x + x @ Ls[-1], self.K[i])
            z = F.instance_norm(z)
            x_input = x.clone()

            if coarsen:
                x = F.relu(z)
                x = F.avg_pool1d(x, 3, stride=2, padding=1)
                m = F.avg_pool1d(m, 3, stride=2, padding=1)
                X = F.avg_pool1d(X, 3, stride=2, padding=1)
                L = getGraphLap(X)
                mS.append(m)
                Ls.append(L)
                Xs.append(X)

            else:
                x = z
                x = x + x_input

        n_scales = len(xS)

        if idxs_clone is not None:
            idxs = idxs_clone
            vals = vals_clone
        for i in reversed(range(nL)):

            refine = self.K[i].shape[0] != self.K[i].shape[1]
            if refine:
                n_scales -= 1
                x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
                mS = mS[:-1]
                Ls = Ls[:-1]
                Xs = Xs[:-1]

            if idxs is None and gradFlag:
                grad = utils.graphGrad(x, Xs[-1], k=10)
                grad, _ = torch.max(grad, dim=3)
            elif gradFlag:
                if refine:
                    vals = vals[:-1]
                    idxs = idxs[:-1]
                curr_idx = idxs[-1]
                curr_val = vals[-1]
                grad = utils.graphGrad(x, X, k=10, vals=curr_val, idx=curr_idx)
                grad, _ = torch.max(grad, dim=3)

            z = mS[-1] * conv1T(x + x @ Ls[-1], self.K[i])

            z = F.instance_norm(z)
            x_input = x.clone()
            x = F.relu(z)
            if refine:
                x = x + xS[n_scales]
            else:
                x = x_input + x



        return F.relu(x)


#
##### END UNET ###########################

class stackedGraphUnet(nn.Module):

    def __init__(self, nLevels, nsmooth, nin, nopen, nLayers, nout, h=0.1):
        super(stackedGraphUnet, self).__init__()
        Unets, Kopen, Kclose = self.init_weights(nLevels, nsmooth, nin, nopen, nLayers, nout)
        self.Unets = nn.ModuleList(Unets)
        self.h = h
        self.Kopen = Kopen
        self.Kclose = Kclose
        self.nLevels = nLevels

    def init_weights(self, nLevels, nsmooth, nin, nopen, nLayers, nout):

        print('Initializing network  ')
        Kopen = nn.Parameter(torch.rand(nopen, nin) * 1e-3)
        Kclose = nn.Parameter(torch.rand(nout, nopen) * 1e-3)

        Unets = []
        total_params = 0
        for i in range(nLayers):
            Unet = GraphUnet(nLevels, nopen, nsmooth)
            Unets.append(Unet)
            total_params += sum(p.numel() for p in Unet.parameters())

        print('Total Number of parameters ', total_params)
        return Unets, Kopen, Kclose

    def forward(self, x, m=torch.tensor([1.0])):

        nL = len(self.Unets)
        x = self.Kopen @ x
        xold = x

        idxs = []
        vals = []
        X = self.Kclose @ x

        for i in range(nL):
            temp = x
            if i % 2 == 0:
                X = self.Kclose @ x
            x = x - self.h * self.Unets[i](x, X, m, idxs=idxs, vals=vals)
            xold = temp
        x = self.Kclose @ x
        xold = self.Kclose @ xold
        return x, xold

    def backProp(self, x, m=torch.tensor([1.0])):

        nL = len(self.Unets)
        x = self.Kclose.t() @ x
        xold = x

        Xs = []
        idxs = []
        vals = []
        X = self.Kclose @ x
        Xs.append(X)

        for i in reversed(range(nL)):
            temp = x
            if i % 2 == 0:
                X = self.Kclose @ x
            x = x - self.h * self.Unets[i](x, X, m, idxs=idxs, vals=vals)

            xold = temp

        x = self.Kopen.t() @ x
        xold = self.Kopen.t() @ xold
        return x, xold

    def NNReg(self):

        uWeights = list(unet.K for unet in self.Unets)
        w0 = (list(zip(*uWeights))[0])
        w0 = torch.stack(w0)
        w1 = (list(zip(*uWeights))[1])
        w1 = torch.stack(w1)
        w2 = (list(zip(*uWeights))[2])
        w2 = torch.stack(w2)
        w3 = (list(zip(*uWeights))[3])
        w3 = torch.stack(w3)
        w4 = (list(zip(*uWeights))[4])
        w4 = torch.stack(w4)

        dW0dt = w0[1:] - w0[:-1]
        dW0dt = torch.sum(torch.abs(dW0dt)) / dW0dt.numel()

        dW1dt = w1[1:] - w1[:-1]
        dW1dt = torch.sum(torch.abs(dW1dt)) / dW1dt.numel()

        dW2dt = w2[1:] - w2[:-1]
        dW2dt = torch.sum(torch.abs(dW2dt)) / dW2dt.numel()

        dW3dt = w3[1:] - w3[:-1]
        dW3dt = torch.sum(torch.abs(dW3dt)) / dW3dt.numel()

        dW4dt = w4[1:] - w4[:-1]
        dW4dt = torch.sum(torch.abs(dW4dt)) / dW4dt.numel()

        RW = dW0dt + dW1dt + dW2dt + dW3dt + dW4dt
        RKo = torch.norm(self.Kopen) ** 2 / 2 / self.Kopen.numel()
        RKc = torch.norm(self.Kclose) ** 2 / 2 / self.Kclose.numel()
        return RW + RKo + RKc
