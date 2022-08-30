import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import datetime
import os
import sys
sys.path.insert(0, os.getcwd())
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ZtoC = False
CtoZ = True
classify = True
useBothPSSMandOneHot = True
base_path = '/yourpath/'
from src import utils
from src import graphUnetworks as gunts
caspver = "casp11"
reload = False
# load training data
Aind = torch.load(base_path + caspver + '/AminoAcidIdx.pt')
Yobs = torch.load(base_path + caspver + '/RCalpha.pt')
MSK = torch.load(base_path + caspver + '/Masks.pt')
S = torch.load(base_path + caspver + '/PSSM.pt')
n_data_total = len(S)


def getIterData(S, Aind, Yobs, MSK, i, device='cpu', pad=0):
    scale = 1e-2  # picometre -> e-9 , angstrom e-10
    PSSM = S[i].t()
    n = PSSM.shape[1]
    M = MSK[i][:n]
    a = Aind[i]

    X = Yobs[i].t()
    # X = utils.linearInterp1D(X, M)
    X = torch.tensor(X)

    X = X - torch.mean(X, dim=1, keepdim=True)
    U, Lam, V = torch.svd(X)

    Coords = scale * torch.diag(Lam) @ V.t()
    Coords = Coords.type('torch.FloatTensor')

    PSSM = PSSM.type(torch.float32)

    A = torch.zeros(20, n)
    A[a, torch.arange(0, n)] = 1.0
    Seq = torch.cat((PSSM, A))

    if pad > 0:
        L = Coords.shape[1]
        k = 2 ** torch.tensor(L, dtype=torch.float64).log2().ceil().int()
        k = k.item()
        CoordsPad = torch.zeros(3, k)
        CoordsPad[:, :Coords.shape[1]] = Coords
        SeqPad = torch.zeros(Seq.shape[0], k)
        SeqPad[:, :Seq.shape[1]] = Seq
        Mpad = torch.zeros(k)
        MM = torch.zeros(k)
        Mpad[:M.shape[0]] = torch.ones(M.shape[0], device=M.device)
        M = Mpad
        MM[:M.shape[0]] = M
        M = MM
        Seq = SeqPad
        num_residues = Coords.shape[1]
        Coords = CoordsPad

    Seq = Seq.to(device=device, non_blocking=True)
    Coords = Coords.to(device=device, non_blocking=True)
    M = M.type('torch.FloatTensor')
    M = M.to(device=device, non_blocking=True)
    Mpad = Mpad.type('torch.FloatTensor')
    Mpad = Mpad.to(device=device, non_blocking=True)

    return Seq, Coords, M, Mpad


# Unet Architecture
nLevels = 2  # 3
nin = 40
nsmooth = 2
nopen = 256
nLayers = 12
nout = 3
h = 0.1
model = gunts.stackedGraphUnet(nLevels, nsmooth, nin, nopen, nLayers, nout, h)
model.to(device)
now = datetime.datetime.now()

model = gunts.stackedGraphUnet(nLevels, nsmooth, nin, nopen, nLayers, nout, h)
model.to(device)

lrO = 1e-4
lrC = 1e-4
lrU = 1e-4

optimizer = optim.Adam([{'params': model.Kopen, 'lr': lrO},
                        {'params': model.Kclose, 'lr': lrC},
                        {'params': model.Unets.parameters(), 'lr': lrU}])

alossBest = 1e6
epochs = 1000
sig = 0.3
ndata = n_data_total
bestModel = model
hist = torch.zeros(epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
best_coord = 1e6
best_kl = 1e6

l = ['TBM', 'TBM-hard', 'FM', 'Unclassified']
typeToIdx = dict([(y, x) for x, y in enumerate(sorted(set(l)))])


def test(epoch):
    # load Testing data
    AindTest = torch.load(base_path + caspver + '/AminoAcidIdxTesting.pt')
    YobsTest = torch.load(base_path + caspver + '/RCalphaTesting.pt')
    MSKTest = torch.load(base_path + caspver + '/MasksTesting.pt')
    STest = torch.load(base_path + caspver + '/PSSMTesting.pt')
    TYPETest = torch.load(base_path + caspver + '/TYPES_Testing.pt')

    loss_per_type = torch.zeros(4, 1).to(device)  # 0 TBM, 1TBM Hard, 2 FM, 3 Unclassified
    counter_per_type = torch.zeros(4, 1).to(device)
    all_losses = []
    nVal = len(STest)
    Couts = dict()
    Coordss = dict()
    dRMSDs = dict()
    accs = []
    # Validation on 0-th data
    with torch.no_grad():
        totalMisfitOneHot = 0.0
        totalMisfitPSSM = 0.0
        misVal = 0.0
        misbVal = 0.0
        misAQval = 0.0
        AQdisTEST = 0.0
        nVal = len(STest)
        for jj in range(nVal):
            Z, Coords, M, Mpad = getIterData(STest, AindTest, YobsTest, MSKTest, jj,
                                             device=device, pad=1)
            prot_type = TYPETest[jj]
            Coords = Coords.unsqueeze(0).to(device)
            Z = Z.unsqueeze(0).to(device)
            M = M.unsqueeze(0).unsqueeze(0).to(device)
            M_verts = M.clone()
            Mpad = Mpad.unsqueeze(0).unsqueeze(0).to(device)

            Cout, CoutOld = model(Z, Mpad)
            Couts.update({str(jj): Cout.clone().detach().cpu().numpy()})
            Coordss.update({str(jj): Coords.clone().detach().cpu().numpy()})
            Zout, ZOld = model.backProp(Coords, Mpad)
            if not useBothPSSMandOneHot:
                PSSMpred = F.softshrink(Zout[:20, :].abs(), Zout.abs().mean().item() / 5)
                misfit = utils.kl_div(PSSMpred, Z[:20, :], weight=True)
            else:
                labels = Z.squeeze()[20:, :]
                PSSMpred = Zout.squeeze()[20:, :]
                misfit = utils.kl_div(PSSMpred, labels, weight=True)  # F.nll_loss(PSSMpred, labels)
                misfitOneHot = misfit

                pssm = torch.softmax(Zout.squeeze()[:20, :].clone(), dim=0).cpu().numpy()
                PSSMpred = F.softshrink(Zout.squeeze()[:20, :].abs(), Zout.abs().mean().item() / 5)
                misfitPSSM = utils.kl_div(PSSMpred, Z.squeeze()[:20, :], weight=True)

                totalMisfitPSSM += misfitPSSM
                totalMisfitOneHot += misfitOneHot
            all_losses.append(misfitPSSM)
            misVal += misfit

            M = torch.ger(M.squeeze(), M.squeeze())
            misfitBackward = utils.dRMSD(Cout, Coords, M, cutoff=6 * 3.8)
            AQloss = torch.sqrt(misfitBackward)
            misbVal += misfitBackward
            AQdisTEST += AQloss
            dRMSDs.update({str(jj): AQloss.clone().detach().cpu().numpy()})
            loss_per_type[typeToIdx[str(prot_type)]] += torch.sqrt(misfitBackward)
            counter_per_type[typeToIdx[str(prot_type)]] += 1
            misAQval += 0
            output = torch.softmax(Zout.squeeze()[20:, :], dim=0)
            _, predicted = output.max(dim=0)
            _, gt = (Z.squeeze()[20:, M_verts.squeeze() == 1]).max(dim=0)
            num_correct = float((predicted.squeeze()[M_verts.squeeze() == 1] == gt).sum())
            num_total = float((M_verts.squeeze() == 1).sum())
            accs.append(num_correct / num_total)

        print("TEST : %2d       %10.3E   %10.3E   %10.3E    %10.3E" % (
            epoch, misVal / nVal, misbVal / nVal, AQdisTEST / nVal, misAQval / nVal), flush=True)
        print("TEST, MisfitOneHot:", totalMisfitOneHot / nVal, flush=True)
        print("TEST, MisfitPSSM:", totalMisfitPSSM / nVal, flush=True)
        print("Test accuracy:", accs, np.array(accs).mean(), flush=True)


for j in range(epochs):
    # Prepare the data
    totalMisfitOneHot = 0.0
    totalMisfitPSSM = 0.0
    aloss = 0.0
    amis = 0.0
    amisb = 0.0
    amisAQ = 0.0
    AQdis = 0.0
    Reg = 0.0
    lDist = 0.0
    model.train()
    perm = torch.randperm(ndata)
    model.zero_grad()
    optimizer.zero_grad()

    for i in range(ndata):
        jj = perm[i]
        Z, Coords, M, Mpad = getIterData(S, Aind, Yobs, MSK, jj, device=device, pad=1)
        Z = Z.unsqueeze(0).to(device)
        Coords = Coords.unsqueeze(0).to(device)
        M = M.unsqueeze(0).unsqueeze(0).to(device)
        M_verts = M.clone()
        Mpad = Mpad.unsqueeze(0).unsqueeze(0).to(device)
        if ZtoC:
            Cout, CoutOld = model(Z, Mpad)
            M = torch.ger(M.squeeze(), M.squeeze())
            misfitBackward = utils.dRMSD(Cout, Coords, M, cutoff=7 * 3.9)
            AQloss = torch.sqrt(misfitBackward)
            amisb += misfitBackward
            C0 = torch.norm(Cout - CoutOld) ** 2 / torch.numel(Z)
            lossForward = AQloss
            model.zero_grad()
            optimizer.zero_grad()
            lossForward.backward()
            if not CtoZ:
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
        if CtoZ:
            Zout, Zold = model.backProp(Coords, Mpad)

            if useBothPSSMandOneHot:
                labels = Z.squeeze()[20:, :]
                PSSMpred = Zout.squeeze()[20:, :]
                misfit = utils.kl_div(PSSMpred, labels, weight=True)
                misfitOneHot = misfit

                PSSMpred = F.softshrink(Zout.squeeze()[:20, :].abs(), Zout.abs().mean().item() / 5)
                misfitPSSM = utils.kl_div(PSSMpred, Z.squeeze()[:20, :], weight=True)

                totalMisfitPSSM += misfitPSSM
                totalMisfitOneHot += misfitOneHot
                misfit += misfitPSSM

            else:
                if classify:
                    labels = Z.squeeze()[20:, :]
                    PSSMpred = Zout.squeeze()[20:, :]
                    misfit = utils.kl_div(PSSMpred, labels, weight=True)  # F.nll_loss(PSSMpred, labels)
                    totalMisfitOneHot += misfit

                else:
                    PSSMpred = F.softshrink(Zout.squeeze()[:20, :].abs(), Zout.abs().mean().item() / 5)
                    misfit = utils.kl_div(PSSMpred, Z.squeeze()[:20, :], weight=True)
                    totalMisfitPSSM += misfit

            Z0 = torch.norm(Zout - Zold) ** 2 / torch.numel(Z)

            lossBackward = misfit + 10 * Z0
            lossBackward.backward()
            if not ZtoC:
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
            # optimizer.step()
            lossDist = 0

        if ZtoC and CtoZ:
            optimizer.step()
            aloss += lossBackward.detach() + lossBackward.detach()
            amis += misfit.detach().item()
            amisb += misfitBackward.detach().item()
            lDist += lossDist

            AQdis += AQloss
            amisAQ += 0

        elif ZtoC:
            aloss += lossForward.detach()
            amis += 0
            amisb += misfitBackward.detach().item()
            lDist += 0
            AQdis += AQloss
            amisAQ += 0

        elif CtoZ:
            aloss += lossBackward.detach()
            amis += misfit.detach().item()
            amisb += 0
            lDist += 0
            AQdis += 0
            amisAQ += 0

        nprnt = 10
        if (i + 1) % nprnt == 0:
            amis = amis / nprnt
            amisb = amisb / nprnt
            amisAQ = amisAQ / nprnt
            AQdis = AQdis / nprnt
            Reg = Reg / nprnt
            lDist = lDist / nprnt
            print("e , i, dmis, cmis ,ldist,   emet,     aqmet")
            print("%2d.%1d   %10.3E  %10.3E   %10.3E   %10.3E   %10.3E   " %
                  (j, i, amis, amisb, lDist, AQdis, amisAQ), flush=True)
            print("TRAIN, MisfitOneHot:", totalMisfitOneHot / nprnt, flush=True)
            print("TRAIN, MisfitPSSM:", totalMisfitPSSM / nprnt, flush=True)
            amis = 0.0
            amisb = 0.0
            amisAQ = 0.0
            AQdis = 0.0
            Reg = 0.0
            lDist = 0.0
            totalMisfitOneHot = 0.0
            totalMisfitPSSM = 0.0
    # test(epoch=j, iter=-1, save=tosave)
    # scheduler.step()
