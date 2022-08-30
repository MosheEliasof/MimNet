from scipy import interpolate
import torch
import numpy as np
import torch.nn.functional as F




def getDistMat(X, msk=torch.tensor([1.0]), train_mode=False):
    D = torch.sum(torch.pow(X, 2), dim=0, keepdim=True) + torch.sum(torch.pow(X, 2), dim=0,
                                                                    keepdim=True).t() - 2 * X.t() @ X

    dev = X.device
    msk = msk.to(dev)

    mm = torch.ger(msk, msk)
    if train_mode:
        return mm * torch.relu(D)
    return mm * torch.sqrt(torch.relu(D))


def getNormMat(N, msk=torch.tensor([1.0])):
    N = N / torch.sqrt(torch.sum(N ** 2, dim=0, keepdim=True) + 1e-9)
    D = N.t() @ N
    mm = torch.ger(msk, msk)
    return mm * D


def orgProtData(x, normals, s, msk, sigma=1.0):
    n = s.shape[1]
    D = getDistMat(x, msk)
    D = torch.exp(-sigma * D)
    N = getNormMat(normals, msk)
    XX = torch.zeros(20, 20, n, n)
    NN = torch.zeros(20, 20, n, n)
    mm = torch.ger(msk, msk)
    mm = mm.view(-1)

    for i in range(20):
        for j in range(20):
            sij = 0.5 * (torch.ger(s[i, :], s[j, :]) + torch.ger(s[j, :], s[i, :]))
            XX[i, j, :, :] = sij * D
            NN[i, j, :, :] = sij * N

    XX = XX.reshape((400, -1))
    NN = NN.reshape((400, -1))
    return XX, NN


def getAdjMatrix(X, st=13):
    # normalize the data
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) / X.shape[1] + 1e-4)
    W = getDistMat(X)
    n = W.shape[0]
    # Position matrix
    P = torch.diag(torch.ones(n)) + \
        torch.diag(torch.ones(n - 1), -1) + \
        torch.diag(torch.ones(n - 2), -2) + \
        torch.diag(torch.ones(n - 3), -3)
    P = P + P.t()
    for jj in range(n):
        if jj < W.shape[0] - 1:
            W[jj, jj + 1] = 0
        if jj > 0 - 1:
            W[jj, jj - 1] = 0
        wj = W[jj, :]
        ws, id = torch.sort(wj, descending=False)
        idb = id[:st]
        wjs = wj * 0
        wjs[idb] = wj[idb] * 0 + 1
        W[jj, :] = wjs
    W = (W + W.t()) / 2.0 + P
    W[W != 0] = 1.0
    return W


def getGraphLapBin(X, st=13):
    W = getAdjMatrix(X, st=st)
    D = torch.diag(torch.sum(W, dim=0))
    L = D - W
    L = 0.5 * (L + L.t())

    return L, W


def getWeights(X, sig=10):
    Xa = X
    W = getDistMat(Xa)
    W = torch.exp(-W / sig)
    return W


def getGraphLap(X, sig=10, p_delete=0.0, pos_encoding=False):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / (torch.std(X, dim=1, keepdim=True) + 1e-3)

    if pos_encoding:
        pos = torch.linspace(-1, 1, X.shape[1]).unsqueeze(0).cuda()
        Xa = torch.cat((X, pos), dim=0)
    else:
        Xa = X

    W = getDistMat(Xa)
    if p_delete > 0.0:
        diags = ~(torch.eye(W.shape[1], W.shape[1], dtype=torch.bool)).cuda()
        probs = torch.rand(W.shape).cuda()
        probs = (probs > p_delete) & diags
        W[probs] = 100

    W = torch.exp(-W / sig)
    D = torch.diag(torch.sum(W, dim=0))
    L = D - W
    Dh = torch.diag(1 / torch.sqrt(torch.diag(D)))
    L = Dh @ L @ Dh

    L = 0.5 * (L + L.t())
    return L, W


def getGraphLap_grad(X, sig=10):
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / (torch.std(X, dim=1, keepdim=True) + 1e-3)
    pos = torch.linspace(-0.5, 0.5, X.shape[1]).unsqueeze(0)
    dev = X.device
    pos = pos.to(dev)
    Xa = torch.cat((X, 5e1 * pos), dim=0)
    W = getDistMat(Xa)
    We = torch.exp(-W / sig)
    D = torch.diag(torch.sum(We, dim=0))
    L = D - We
    Dh = torch.diag(1 / torch.sqrt(torch.diag(D)))
    L = Dh @ L @ Dh

    L = 0.5 * (L + L.t())

    return L, W




def linearInterp1D(X, M):
    n = X.shape[1]
    ti = np.arange(0, n)
    t = ti[M != 0]
    f = interpolate.interp1d(t, X[:, M != 0], kind='slinear', axis=-1, copy=True, bounds_error=None,
                             fill_value='extrapolate')
    Xnew = f(ti)

    return Xnew


def distConstraint(X, dc=0.375):
    n = X.shape[1]
    dX = X[:, 1:] - X[:, :-1]
    d = torch.sqrt(torch.sum(dX ** 2, dim=0, keepdim=True))
    dX = (dX / d) * dc

    Xh = torch.zeros(3, n, device=X.device)
    Xh[:, 0] = X[:, 0]
    Xh[:, 1:] = X[:, 0].unsqueeze(1) + torch.cumsum(dX, dim=1)

    return Xh


def kl_div(p, q, weight=False):
    n = p.shape[1]
    p = torch.log_softmax(p, dim=0)
    KLD = F.kl_div(p.unsqueeze(0), q.unsqueeze(0), reduction='none').squeeze(0)
    if weight:
        r = torch.sum(q, dim=1)
    else:
        r = torch.ones(q.shape[0])

    r = r / r.sum()
    KLD = torch.diag(1 - r) @ KLD
    return KLD.sum() / KLD.shape[1]


def distances(x):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    return pairwise_distance


def distPenality(D, dc=0.379, M=torch.ones(1), mean=True):
    U = torch.triu(D, 2)
    p2 = torch.norm(M * torch.relu(2 * dc - U)) ** 2
    return p2 / D.shape[1]


def getIterData_nopad(S, Aind, Yobs, MSK, i, device='cpu', return_a=False):
    scale = 1e-2
    PSSM = S[i].t()
    n = PSSM.shape[1]
    M = MSK[i][:n]
    a = Aind[i]

    X = Yobs[i].t()
    X = linearInterp1D(X, M)
    X = torch.tensor(X)

    X = X - torch.mean(X, dim=1, keepdim=True)
    U, Lam, V = torch.svd(X)

    Coords = scale * torch.diag(Lam) @ V.t()
    Coords = Coords.type('torch.FloatTensor')

    PSSM = PSSM.type(torch.float32)

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))
    Seq = Seq.to(device=device, non_blocking=True)

    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)
    if return_a:
        return Seq, Coords, M, a
    return Seq, Coords, M


def dRMSD(X, Xobs, M, cutoff=3.9 * 100, k=None, training=False):
    X = torch.squeeze(X)
    Xobs = torch.squeeze(Xobs)
    M = torch.squeeze(M)

    D = torch.sum(torch.pow(X, 2), dim=0, keepdim=True) + torch.sum(torch.pow(X, 2), dim=0,
                                                                    keepdim=True).t() - 2 * X.t() @ X
    D = torch.sqrt(torch.relu(D))
    Dobs = torch.sum(torch.pow(Xobs, 2), dim=0, keepdim=True) + torch.sum(torch.pow(Xobs, 2), dim=0,
                                                                          keepdim=True).t() - 2 * Xobs.t() @ Xobs
    # Dobs = torch.sqrt(torch.relu(Dobs))
    # Filter non-physical ones
    n = X.shape[-1]
    Xl = torch.zeros(3, n, device=X.device)
    Xl[0, :] = 3.9 * torch.arange(0, n)
    Dl = torch.sum(Xl ** 2, dim=0, keepdim=True) + torch.sum(Xl ** 2, dim=0, keepdim=True).t() - 2 * Xl.t() @ Xl
    Dl = torch.sqrt(torch.relu(Dl))
    ML = (M * Dl - M * torch.sqrt(torch.relu(Dobs))) > 0
    MS = torch.sqrt(torch.relu(Dobs)) < cutoff  # 7 * 3.9

    if k is not None and k > 0:
        Dobs = Dobs.unsqueeze(0).unsqueeze(0).float()
        D = D.unsqueeze(0).unsqueeze(0).float()
        M = M.unsqueeze(0).unsqueeze(0).float()
        ML = ML.unsqueeze(0).unsqueeze(0).float()
        MS = MS.unsqueeze(0).unsqueeze(0).float()
        for i in torch.range(0, k):
            out_shape = (int(round(Dobs.shape[2] / 2)), int(round(Dobs.shape[3] / 2)))
            Dobs = torch.nn.functional.adaptive_avg_pool2d(Dobs, output_size=out_shape)
            D = torch.nn.functional.adaptive_avg_pool2d(D, output_size=out_shape)
            M = torch.nn.functional.adaptive_max_pool2d(M, output_size=out_shape)
            ML = torch.nn.functional.adaptive_max_pool2d(ML, output_size=out_shape)
            MS = torch.nn.functional.adaptive_max_pool2d(MS, output_size=out_shape)

        Dobs = torch.squeeze(Dobs)
        D = torch.squeeze(D)
        M = torch.squeeze(M).int()
        MS = torch.squeeze(MS).int()
        ML = torch.squeeze(ML).int()

    Morig = M.clone()
    M = M > 0
    if not training:
        if k is None:
            M = (M & MS & ML) * 1.0
        else:
            M = (M * MS * ML) * 1.0
    else:
        M = (M & ML) * 1.0
    if k is None:
        R = torch.triu(D - torch.sqrt(torch.relu(Dobs)), 1)
        M = torch.triu(M, 1)
    else:
        R = D - torch.sqrt(torch.relu(Dobs))
    loss = torch.norm(M * R) ** 2 / torch.sum(M)

    return loss


def dRMSD_pointclouds(X, Xobs, M):
    X = torch.squeeze(X)
    Xobs = torch.squeeze(Xobs)
    M = torch.squeeze(M)

    D = torch.sum(torch.pow(X, 2), dim=0, keepdim=True) + torch.sum(torch.pow(X, 2), dim=0,
                                                                    keepdim=True).t() - 2 * X.t() @ X
    D = torch.sqrt(torch.relu(D))
    Dobs = torch.sum(torch.pow(Xobs, 2), dim=0, keepdim=True) + torch.sum(torch.pow(Xobs, 2), dim=0,
                                                                          keepdim=True).t() - 2 * Xobs.t() @ Xobs
    # Dobs = torch.sqrt(torch.relu(Dobs))
    # Filter non-physical ones
    n = X.shape[-1]
    Xl = torch.zeros(3, n, device=X.device)
    Xl[0, :] = 3.8 * torch.arange(0, n)
    Dl = torch.sum(Xl ** 2, dim=0, keepdim=True) + torch.sum(Xl ** 2, dim=0, keepdim=True).t() - 2 * Xl.t() @ Xl
    Dl = torch.sqrt(torch.relu(Dl))
    ML = (M * Dl - M * torch.sqrt(torch.relu(Dobs))) > 0

    M = M > 0
    D = torch.exp(-0.5 * D)
    Dobs = torch.sqrt(torch.relu(Dobs))
    Dobs = torch.exp(-0.5 * Dobs)
    R = torch.triu(D - Dobs, 2)
    M = torch.triu(M, 1)
    loss = torch.norm(M * R) ** 2 / torch.sum(M)

    return loss


