import numpy as np
import pysptk as sptk
import pyworld as pw
from scipy.io import wavfile
import os
import sys
import time
import argparse
from train import *

# load trained model
dim = 25
n_units = 128

model = VCDNN(dim,n_units)
_ = model.load_state_dict(torch.load("model/vcmodel.model"))

# load eval data
datalist = []
with open("conf/eval.list","r") as f:
    for line in f:
        line = line.rstrip()
        datalist.append(line)

# constants
fs = 16000
fftlen = 512
alpha = 0.42

for i in range(0,len(datalist)):
    outfile = "result/wav/{}_both.wav".format(datalist[i])
    with open("data/SF/mgc/{}.mgc".format(datalist[i]), "rb") as f:
        mgc = np.fromfile(f, dtype="<f8", sep="")
        mgc = mgc.reshape(len(mgc)//dim, dim)
        mgc = model.get_predata(mgc) # convert mgc
    with open("data/SF-TF/f0/{}.f0".format(datalist[i]),"rb") as f:
        f0 = np.fromfile(f, dtype="<f8", sep="")
    with open("data/SF/ap/{}.ap".format(datalist[i]),"rb") as f:
        ap = np.fromfile(f, dtype="<f8", sep="")
        ap = ap.reshape(len(ap)//(fftlen+1),fftlen+1)
    mgc = mgc.astype(np.float64)
    sp = sptk.mc2sp(mgc, alpha, fftlen*2)
    owav = pw.synthesize(f0, sp, ap, fs)
    owav = np.clip(owav, -32768, 32767)
    wavfile.write(outfile, fs, owav.astype(np.int16))
