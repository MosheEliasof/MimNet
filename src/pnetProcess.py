import numpy as np
from scipy import interpolate
from numpy.linalg import norm
import re
import os
import torch
import sys
import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# import os.path as osp
# from torch.utils.data import Dataset
# from os import listdir
# from os.path import isfile, join
# from Bio.PDB import PDBParser
from dataloader_utils import AA_DICT, MASK_DICT, DSSP_DICT, NUM_DIMENSIONS, AA_PAD_VALUE, MASK_PAD_VALUE, \
    DSSP_PAD_VALUE, SeqFlip, PSSM_PAD_VALUE, ENTROPY_PAD_VALUE, COORDS_PAD_VALUE, ListToNumpy


# Routines to read the file
def separate_coords(full_coords, pos):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    res = []
    for i in range(len(full_coords[0])):
        if i % 3 == pos:
            res.append([full_coords[j][i] for j in range(3)])

    return res


class switch(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5
            self.fall = True
            return True
        else:
            return False


def letter_to_num(string, dict_):
    """ Convert string of letters to list of ints """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num


def flip_multidimensional_list(list_in):  # pos can be either 0(n_term), 1(calpha), 2(cterm)
    list_out = []
    ld = len(list_in)
    for i in range(len(list_in[0])):
        list_out.append([list_in[j][i] for j in range(ld)])
    return list_out


class list2np(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        args_array = ()
        for arg in args:
            args_array += (np.asarray(arg),)
        return args_array

    def __repr__(self):
        return self.__class__.__name__ + '()'


def read_record(file_, num_evo_entries):
    """ Read all protein records from pnet file. """

    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []
    protType = []

    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                curr_line = file_.readline()[:-1]
                print("Line is:", curr_line)
                t, name = curr_line.split("#")
                print("type:", t)
                print("name:", name)
                id.append(name)
                protType.append(t)

            elif case('[PRIMARY]' + '\n'):
                seq.append(letter_to_num(file_.readline()[:-1], AA_DICT))
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries):
                    evolutionary.append([float(step) for step in file_.readline().split()])
                pssm.append(evolutionary)
                entropy.append([float(step) for step in file_.readline().split()])
            elif case('[SECONDARY]' + '\n'):
                dssp.append(letter_to_num(file_.readline()[:-1], DSSP_DICT))
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord) for coord in file_.readline().split()])
                coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask.append(letter_to_num(file_.readline()[:-1], MASK_DICT))
            elif case(''):
                return protType, id, seq, pssm, entropy, coord, mask

def read_record_train(file_, num_evo_entries):
    """ Read all protein records from pnet file. """

    id = []
    seq = []
    pssm = []
    entropy = []
    dssp = []
    coord = []
    mask = []
    protType = []

    while True:
        next_line = file_.readline()
        for case in switch(next_line):
            if case('[ID]' + '\n'):
                id.append(file_.readline()[:-1])
            elif case('[PRIMARY]' + '\n'):
                seq.append(letter_to_num(file_.readline()[:-1], AA_DICT))
            elif case('[EVOLUTIONARY]' + '\n'):
                evolutionary = []
                for residue in range(num_evo_entries):
                    evolutionary.append([float(step) for step in file_.readline().split()])
                pssm.append(evolutionary)
                entropy.append([float(step) for step in file_.readline().split()])
            elif case('[SECONDARY]' + '\n'):
                dssp.append(letter_to_num(file_.readline()[:-1], DSSP_DICT))
            elif case('[TERTIARY]' + '\n'):
                tertiary = []
                for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord) for coord in file_.readline().split()])
                coord.append(tertiary)
            elif case('[MASK]' + '\n'):
                mask.append(letter_to_num(file_.readline()[:-1], MASK_DICT))
            elif case(''):
                return id, seq, pssm, entropy, coord, mask


def parse_pnet(file):
    with open(file, 'r') as f:
        type, id, seq, pssm, entropy, coords, mask = read_record(f, 20)
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        for i in range(len(pssm)):  # We transform each of these, since they are inconveniently stored
            pssm2.append(flip_multidimensional_list(pssm[i]))
            r1.append(separate_coords(coords[i], 0))
            r2.append(separate_coords(coords[i], 1))
            r3.append(separate_coords(coords[i], 2))

    return type, id, seq, pssm2, entropy, r1, r2, r3, mask

def parse_pnet_train(file):
    with open(file, 'r') as f:
        id, seq, pssm, entropy, coords, mask = read_record_train(f, 20)
        r1 = []
        r2 = []
        r3 = []
        pssm2 = []
        for i in range(len(pssm)):  # We transform each of these, since they are inconveniently stored
            pssm2.append(flip_multidimensional_list(pssm[i]))
            r1.append(separate_coords(coords[i], 0))
            r2.append(separate_coords(coords[i], 1))
            r3.append(separate_coords(coords[i], 2))

    return id, seq, pssm2, entropy, r1, r2, r3, mask




################# Processing the data to maps

def ang2plain(v1, v2, v3, v4):
    nA = torch.cross(v1, v2)
    nA = nA / torch.norm(nA)
    nB = torch.cross(v3, v4)
    nB = nB / torch.norm(nB)

    cosPsi = torch.dot(nA, nB)
    # Psi    = torch.acos(cosPsi)
    return cosPsi


def crossProdMat(V1, V2):
    Vcp = torch.zeros(V1.shape)
    Vcp[0, :, :] = V1[1, :, :] * V2[2, :, :] - V1[2, :, :] * V2[1, :, :]
    Vcp[1, :, :] = -V1[0, :, :] * V2[2, :, :] + V1[2, :, :] * V2[0, :, :]
    Vcp[2, :, :] = V1[0, :, :] * V2[1, :, :] - V1[1, :, :] * V2[0, :, :]
    return Vcp


def dotProdMat(V1, V2):
    Vdp = torch.sum(V1 * V2, dim=0)
    return Vdp


def ang2plainMat(v1, v2, v3, v4):
    nA = crossProdMat(v1, v2)
    nB = crossProdMat(v3, v4)
    nA = nA / (torch.sqrt(torch.sum(nA ** 2, axis=0)).unsqueeze(0))
    nB = nB / (torch.sqrt(torch.sum(nB ** 2, axis=0)).unsqueeze(0))

    cosPsi = torch.sum(nA * nB, axis=0)
    Psi = torch.acos(cosPsi)
    indnan = torch.isnan(Psi)
    Psi[indnan] = 0
    return Psi


def convertCoordToDistMaps(rN, rCa, rCb, mask=None):
    # Central coordinate
    # rM = (rN + rCa + rCb)/3
    M = mask.unsqueeze(1) @ mask.unsqueeze(0)
    # Get Daa
    D = torch.sum(rCa ** 2, dim=1).unsqueeze(1) + torch.sum(rCa ** 2, dim=1).unsqueeze(0) - 2 * (rCa @ rCa.t())
    Daa = torch.sqrt(torch.relu(M * D))

    # Get Dab
    D = torch.sum(rCa ** 2, dim=1).unsqueeze(1) + torch.sum(rCb ** 2, dim=1).unsqueeze(0) - 2 * (rCa @ rCb.t())
    Dab = torch.sqrt(torch.relu(M * D))

    # Get DaN
    D = torch.sum(rCa ** 2, dim=1).unsqueeze(1) + torch.sum(rN ** 2, dim=1).unsqueeze(0) - 2 * (rCa @ rN.t())
    DaN = torch.sqrt(torch.relu(M * D))

    # Get DbN
    D = torch.sum(rCb ** 2, dim=1).unsqueeze(1) + torch.sum(rN ** 2, dim=1).unsqueeze(0) - 2 * (rCb @ rN.t())
    DbN = torch.sqrt(torch.relu(M * D))

    return Daa, Dab, DaN, DbN, M, rCa


def torsionAngleIJ(r1, r2, r3, r4, M=1.0):
    pi = 3.1415926535
    a = r2 - r1
    b = r3 - r2
    c = r4 - r3
    norma = torch.norm(a)
    normb = torch.norm(b)
    normc = torch.norm(c)
    if norma != 0:
        a = a / norma
    if normb != 0:
        b = b / normb
    if normc != 0:
        c = c / normc

    x = torch.dot(b, torch.cross(torch.cross(c, b), torch.cross(a, b)))
    y = torch.norm(b) * torch.dot(torch.cross(a, b), torch.cross(c, b))
    ang = torch.atan2(x, y)

    return ang


def convertCoordToAngles(rN, rCa, rCb, mask=1.0):
    # OMEGA = torsionAngle(rCa, rCb, rCb, rCa) # Omega: Ca, Cb, Cb, Ca
    # THETA = torsionAngle(rN, rCa, rCb, rCb) # N, Ca, Cb, Cb
    # PHI   = torsionAngle(rCb, rCb, rCa, rN) # Cb, Cb, Ca, N
    n = rN.shape[0]
    OMEGA = torch.zeros(n, n, dtype=rN.dtype)
    THETA = torch.zeros(n, n, dtype=rN.dtype)
    PHI = torch.zeros(n, n, dtype=rN.dtype)
    for i in range(n):
        for j in range(n):
            p1 = rCa[i, :]
            p2 = rCb[i, :]
            p3 = rCb[j, :]
            p4 = rCa[j, :]
            OMEGA[i, j] = torsionAngleIJ(p1, p2, p3, p4)

            p1 = rN[i, :]
            p2 = rCa[i, :]
            p3 = rCb[i, :]
            p4 = rCb[j, :]
            THETA[i, j] = torsionAngleIJ(p1, p2, p3, p4)

            p1 = rCb[i, :]
            p2 = rCb[j, :]
            p3 = rCa[j, :]
            p4 = rN[j, :]
            PHI[i, j] = torsionAngleIJ(p1, p2, p3, p4)

    return mask * OMEGA, mask * THETA, mask * PHI


def interpolateRes(Z, M):
    M = torch.tensor(M)
    Z = torch.tensor(Z, dtype=torch.float64)
    n = M.shape[0]
    x = torch.linspace(0, n - 1, n)
    x = x[M != 0]
    xs = torch.linspace(0, n - 1, n)
    y = Z
    y = y[M != 0, :]

    f = interpolate.interp1d(x.numpy(), y.numpy(), kind='linear', axis=0, copy=True, bounds_error=False,
                             fill_value="extrapolate")
    ys = f(xs)
    ys = torch.tensor(ys, dtype=torch.float64)
    ii = torch.isnan(ys[:, 0])
    msk = 1 - 1 * ii
    Z = ys  # torch.tensor(ys,dtype=torch.float64)
    return Z, msk


# Use the codes to process the data and get X and Y
def getProteinData(seq, pssm2, entropy, RN, RCa, RCb, mask, idx, ncourse):
    L2np = list2np()

    Sh, S, E, rN, rCa, rCb, msk = L2np(seq[idx], pssm2[idx], entropy[idx],
                                       RN[idx], RCa[idx], RCb[idx], mask[idx])

    S = torch.tensor(S)
    Sh = torch.tensor(Sh)
    E = torch.tensor(E)

    nc = ncourse
    kp = S.shape[0]
    k = (2 ** nc) * (kp // (2 ** nc) + 1)
    X = torch.zeros(20, 20, k, k)
    for i in range(20):
        for j in range(20):
            X[i, j, :kp, :kp] = S[:, i].unsqueeze(1) @ S[:, j].unsqueeze(1).t()
    X = X.reshape((400, k, k))
    rN, m = interpolateRes(rN, msk)  # torch.tensor(rN)
    rCa, m = interpolateRes(rCa, msk)  # torch.tensor(rCa)
    rCb, m = interpolateRes(rCb, msk)  # torch.tensor(rCb)
    msk = torch.tensor(m)
    # msk[:] = True
    # Define model and data Y
    # D, OMEGA, PHI, THETA, M = convertCoordToDistAnglesVec(rN,rCa,rCb,mask=msk)
    Dmm, Dma, Dmb, DmN, M, rM = convertCoordToDistMaps(rN, rCa, rCb, mask=msk)

    Yobs = torch.zeros(4, k, k)
    Yobs[0, :kp, :kp] = Dmm
    Yobs[1, :kp, :kp] = Dma
    Yobs[2, :kp, :kp] = Dmb
    Yobs[3, :kp, :kp] = DmN

    Yobs = Yobs.unsqueeze(0)

    Mpad = torch.zeros(1, k, k)
    Mpad[0, :kp, :kp] = M

    return X, Yobs, Mpad, Sh, S, rM


# Use the codes to process the data and get X and Y
def getProteinDataLinear(seq, pssm2, entropy, RN, RCa, RCb, mask, idx, ncourse, inter=True):
    L2np = list2np()

    Sh, S, E, rN, rCa, rCb, msk = L2np(seq[idx], pssm2[idx], entropy[idx],
                                       RN[idx], RCa[idx], RCb[idx], mask[idx])

    S = torch.tensor(S)
    Sh = torch.tensor(Sh)
    E = torch.tensor(E)

    nc = ncourse
    kp = S.shape[0]
    k = (2 ** nc) * (kp // (2 ** nc) + 1)
    X = torch.zeros(21, k)
    X[:-1, :kp] = S.t()
    X[-1, :kp] = E
    X = X.unsqueeze(0)

    if inter:
        rN, m = interpolateRes(rN, msk)  # torch.tensor(rN)
        rCa, m = interpolateRes(rCa, msk)  # torch.tensor(rCa)
        rCb, m = interpolateRes(rCb, msk)  # torch.tensor(rCb)
    else:
        m = torch.tensor(msk)
        rN = torch.tensor(rN)
        rCa = torch.tensor(rCa)
        rCb = torch.tensor(rCb)

    Yobs = torch.zeros(1, 3, 3, k)
    Yobs[0, 0, :, :kp] = rN.t()
    Yobs[0, 1, :, :kp] = rCa.t()
    Yobs[0, 2, :, :kp] = rCb.t()

    msk = torch.zeros(k)
    msk[:kp] = m
    return X, Yobs, msk, S


def plotProteinData(Y, j):
    plt.figure(j)
    plt.subplot(2, 2, 1)
    plt.imshow(Y[0, 0, :, :])
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(Y[0, 1, :, :])
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(Y[0, 2, :, :])
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(Y[0, 3, :, :])
    plt.colorbar()


def convertCoordToAnglesVec(rN, rCa, rCb, mask=1.0):
    # Vectorized operations to compute angles
    # OMEGA : Ca, Cb, Cb, Ca
    # THETA : N, Ca, Cb, Cb
    # PHI   : Cb, Cb, Ca, N
    nat = rCa.shape[0]
    # Phi: Cb, Cb, Ca, N
    V1 = torch.zeros(3, nat, nat, dtype=rN.dtype)
    V2 = torch.zeros(3, nat, nat, dtype=rN.dtype)
    V3 = torch.zeros(3, nat, nat, dtype=rN.dtype)
    V1[0, :, :] = rCb[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(0)
    V1[1, :, :] = rCb[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(0)
    V1[2, :, :] = rCb[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(0)
    V2[0, :, :] = rCb[:, 0].unsqueeze(1) - rCa[:, 0].unsqueeze(1).repeat(1, nat)
    V2[1, :, :] = rCb[:, 1].unsqueeze(1) - rCa[:, 1].unsqueeze(1).repeat(1, nat)
    V2[2, :, :] = rCb[:, 2].unsqueeze(1) - rCa[:, 2].unsqueeze(1).repeat(1, nat)
    V3[0, :, :] = rCa[:, 0].unsqueeze(1) - rN[:, 0].unsqueeze(1).repeat(1, nat)
    V3[1, :, :] = rCa[:, 1].unsqueeze(1) - rN[:, 1].unsqueeze(1).repeat(1, nat)
    V3[2, :, :] = rCa[:, 2].unsqueeze(1) - rN[:, 2].unsqueeze(1).repeat(1, nat)
    # Normalize them
    V1n = torch.sqrt(torch.sum(V1 ** 2, dim=0))
    V1 = V1 / V1n.unsqueeze(0)
    V2n = torch.sqrt(torch.sum(V2 ** 2, dim=0))
    V2 = V2 / V2n.unsqueeze(0)
    V3n = torch.sqrt(torch.sum(V3 ** 2, dim=0))
    V3 = V3 / V3n.unsqueeze(0)
    PHI = mask * ang2plainMat(V1, V2, V2, V3)
    indnan = torch.isnan(PHI)
    PHI[indnan] = 0.0

    # Omega
    nat = rCa.shape[0]
    V1 = torch.zeros(3, nat, nat)
    V2 = torch.zeros(3, nat, nat)
    V3 = torch.zeros(3, nat, nat)
    # Ca1 - Cb1
    V1[0, :, :] = (rCa[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1)).repeat((1, nat))
    V1[1, :, :] = (rCa[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1)).repeat((1, nat))
    V1[2, :, :] = (rCa[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1)).repeat((1, nat))
    # Cb1 - Cb2
    V2[0, :, :] = rCb[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1).t()
    V2[1, :, :] = rCb[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1).t()
    V2[2, :, :] = rCb[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1).t()
    # Cb2 - Ca2
    V3[0, :, :] = (rCb[:, 0].unsqueeze(0) - rCa[:, 0].unsqueeze(0)).repeat((nat, 1))
    V3[1, :, :] = (rCb[:, 1].unsqueeze(0) - rCa[:, 1].unsqueeze(0)).repeat((nat, 1))
    V3[2, :, :] = (rCb[:, 2].unsqueeze(0) - rCa[:, 2].unsqueeze(0)).repeat((nat, 1))

    OMEGA = mask * ang2plainMat(V1, V2, V2, V3)
    indnan = torch.isnan(OMEGA)
    OMEGA[indnan] = 0.0

    # Theta
    V1 = torch.zeros(3, nat, nat)
    V2 = torch.zeros(3, nat, nat)
    V3 = torch.zeros(3, nat, nat)
    # N - Ca
    V1[0, :, :] = (rN[:, 0].unsqueeze(1) - rCa[:, 0].unsqueeze(1)).repeat((1, nat))
    V1[0, :, :] = (rN[:, 1].unsqueeze(1) - rCa[:, 1].unsqueeze(1)).repeat((1, nat))
    V1[2, :, :] = (rN[:, 2].unsqueeze(1) - rCa[:, 2].unsqueeze(1)).repeat((1, nat))
    V2[0, :, :] = (rCa[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1)).repeat((1, nat))
    V2[1, :, :] = (rCa[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1)).repeat((1, nat))
    V2[2, :, :] = (rCa[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1)).repeat((1, nat))
    V3[0, :, :] = rCb[:, 0].unsqueeze(1) - rCb[:, 0].unsqueeze(1).t()
    V3[1, :, :] = rCb[:, 1].unsqueeze(1) - rCb[:, 1].unsqueeze(1).t()
    V3[2, :, :] = rCb[:, 2].unsqueeze(1) - rCb[:, 2].unsqueeze(1).t()

    THETA = mask * ang2plainMat(V1, V2, V2, V3)
    indnan = torch.isnan(THETA)
    THETA[indnan] = 0.0

    return OMEGA, PHI, THETA
## To generate the .pt data files uncomment the following and run the file:
#     base_path = '/yourdatapath/'
# for caspIdx in np.arange(10,11):
#     caspver = "/casp" + str(caspIdx) + "/"
#     print("Doing casp:", caspver, "-----------------------------------------------")
#     import matplotlib.pyplot as plt
#     ##########VAL###################
#     if 1 == 0:
#         protType, id, seq, pssm2, entropy, rN, rCa, rCb, mask = parse_pnet(base_path + caspver + '/validation')
#
#         numprot = len(id)
#         Aind = []
#         PSSM = []
#         DSSP = []
#         E = []
#         ra = []
#         rb = []
#         rn = []
#         M = []
#         L2np = list2np()
#         bins = range(0, 2000, 50)
#         lengths = []
#         mask_ratios = []
#         for idx in range(numprot):
#             Ai, PSSMi, Ei, rNi, rCai, rCbi, mski = L2np(seq[idx], pssm2[idx], entropy[idx],
#                                                         rN[idx], rCa[idx], rCb[idx], mask[idx])
#
#             if idx % 100 == 0:
#                 print(idx)
#             lengths.append(len(Ei))
#             mask_ratios.append(sum(mski > 0) / sum(mski))
#             if len(Ei) <= 1000:
#                 # if (sum(mski > 0) / sum(mski)) < 0.85:
#                 #    print("idx:", idx, " has perc.:", sum(mski > 0) / sum(mski), flush=True)
#                 #    continue
#                 # continue
#                 Aind.append(torch.tensor(Ai))
#                 PSSM.append(torch.tensor(PSSMi))
#                 E.append(torch.tensor(Ei))
#                 ra.append(torch.tensor(rCai))
#                 M.append(torch.tensor(mski))
#                 rb.append(torch.tensor(rCbi))
#                 rn.append(torch.tensor(rNi))
#
#         # len_hist = np.histogram(lengths, bins=bins)
#         # rat_hist = np.histogram(mask_ratios)
#         # _ = plt.hist(lengths, bins=bins)  # arguments are passed to np.histogram
#         # plt.title('Val. Seq. length hist.')
#         # plt.show()
#         # plt.savefig("./plots/val_length_hist.jpg")
#         # plt.close()
#         #
#         # _ = plt.hist(mask_ratios, bins='auto')  # arguments are passed to np.histogram
#         # plt.title('Mask ratio hist.')
#         # plt.show()
#         # plt.savefig("./plots/val_ratio_hist.jpg")
#         # plt.close()
#         torch.save(Aind, base_path + caspver + '/AminoAcidIdxVal.pt')
#         torch.save(PSSM, base_path + caspver + '/PSSMVal.pt')
#         torch.save(E, base_path + caspver + '/entropyVal.pt')
#         torch.save(ra, base_path + caspver + '/RCalphaVal.pt')
#         torch.save(M, base_path + caspver + '/MasksVal.pt')
#
#         torch.save(ra, base_path + caspver + '/CoordAlphaVal.pt')
#         torch.save(rb, base_path + caspver + '/CoordBetaVal.pt')
#         torch.save(rn, base_path + caspver + '/CoordNVal.pt')
#
#     ##########TEST################
#     # id, seq, pssm2, entropy, dssp, rN, rCa, rCb, mask = parse_pnet(base_path + caspver + '/testing')
#     protType, id, seq, pssm2, entropy, rN, rCa, rCb, mask = parse_pnet(base_path + caspver + '/testing')
#     print("id :", id)
#     numprot = len(id)
#     Aind = []
#     PSSM = []
#     E = []
#     ra = []
#     M = []
#     L2np = list2np()
#     IDS = []
#     protTYPES = []
#     lengths = []
#     mask_ratios = []
#
#     for idx in range(numprot):
#         Ai, PSSMi, Ei, rNi, rCai, rCbi, mski = L2np(seq[idx], pssm2[idx], entropy[idx],
#                                                     rN[idx], rCa[idx], rCb[idx], mask[idx])
#         print("protType[idx] :", protType[idx])
#         lengths.append(len(Ei))
#         mask_ratios.append(sum(mski > 0) / sum(mski))
#         if idx % 100 == 0:
#             print(idx)
#         if len(Ei) <= 1000:
#             # if (sum(mski > 0) / sum(mski)) < 0.85:
#             #    print("idx:", idx, " has perc.:", sum(mski > 0) / sum(mski), flush=True)
#             #    continue
#             # continue
#             Aind.append(torch.tensor(Ai))
#             PSSM.append(torch.tensor(PSSMi))
#             E.append(torch.tensor(Ei))
#             ra.append(torch.tensor(rCai))
#             M.append(torch.tensor(mski))
#             IDS.append(id[idx])
#             protTYPES.append(protType[idx])
#             print("id:", id[idx])
#
#     # _ = plt.hist(lengths, bins=bins)  # arguments are passed to np.histogram
#     # plt.title('Test Seq. length hist.')
#     # plt.show()
#     # plt.savefig("./plots/test_length_hist.jpg")
#     # plt.close()
#     #
#     # _ = plt.hist(mask_ratios, bins='auto')  # arguments are passed to np.histogram
#     # plt.title('Mask ratio hist.')
#     # plt.show()
#     # plt.savefig("./plots/test_ratio_hist.jpg")
#     # plt.close()
#     torch.save(Aind, base_path + caspver + '/AminoAcidIdxTesting.pt')
#     torch.save(PSSM, base_path + caspver + '/PSSMTesting.pt')
#     torch.save(E, base_path + caspver + '/entropyTesting.pt')
#     torch.save(ra, base_path + caspver + '/RCalphaTesting.pt')
#     torch.save(M, base_path + caspver + '/MasksTesting.pt')
#     torch.save(IDS, base_path + caspver + '/IDS_Testing.pt')
#     torch.save(protTYPES, base_path + caspver + '/TYPES_Testing.pt')
#     ###############TRAIN############################
#
#     #id, seq, pssm2, entropy, dssp, rN, rCa, rCb, mask = parse_pnet_train(base_path + caspver + '/training_90')
#
#     id, seq, pssm2, entropy, rN, rCa, rCb, mask = parse_pnet_train(base_path + caspver + '/training_90')
#
#     numprot = len(id)
#     print("numprot:", numprot)
#     Aind = []
#     PSSM = []
#     E = []
#     ra = []
#     M = []
#     L2np = list2np()
#
#     lengths = []
#     mask_ratios = []
#
#     for idx in range(numprot):
#         Ai, PSSMi, Ei, rNi, rCai, rCbi, mski = L2np(seq[idx], pssm2[idx], entropy[idx],
#                                                     rN[idx], rCa[idx], rCb[idx], mask[idx])
#         if idx % 100 == 0:
#             print(idx)
#
#         lengths.append(len(Ei))
#         mask_ratios.append(sum(mski > 0) / sum(mski))
#         if len(Ei) <= 1000:
#             # print("(sum(mski > 0) / sum(mski)):", (sum(mski > 0) / sum(mski)))
#             # if (sum(mski > 0) / sum(mski)) < 0.85:
#             #    print("idx:", idx, " has perc.:", sum(mski > 0) / sum(mski), flush=True)
#             #    continue
#             # continue
#             Aind.append(torch.tensor(Ai))
#             PSSM.append(torch.tensor(PSSMi))
#             E.append(torch.tensor(Ei))
#             ra.append(torch.tensor(rCai))
#             M.append(torch.tensor(mski))
#
#     # _ = plt.hist(lengths, bins=bins)  # arguments are passed to np.histogram
#     # plt.title('Train Seq. length hist.')
#     # plt.show()
#     # plt.savefig("./plots/train_length_hist.jpg")
#     # plt.close()
#     #
#     # _ = plt.hist(mask_ratios, bins='auto')  # arguments are passed to np.histogram
#     # plt.title('Mask ratio hist.')
#     # plt.show()
#     # plt.savefig("./plots/train_ratio_hist.jpg"); plt.close()
#
#
#     torch.save(Aind, base_path + caspver + '/AminoAcidIdx.pt')
#     torch.save(PSSM, base_path + caspver + '/PSSM.pt')
#     torch.save(E, base_path + caspver + '/entropy.pt')
#     torch.save(ra, base_path + caspver + '/RCalpha.pt')
#     torch.save(M, base_path + caspver + '/Masks.pt')
#
#     exit()