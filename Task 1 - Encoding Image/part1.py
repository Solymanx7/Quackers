# QPIE method ( Quantum Probability Image Encoding )

# Importing the libraries
from itertools import chain
import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, BasicAer, transpile, assemble
from qiskit.execute_function import execute
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from PIL import Image
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.metrics import mean_squared_error
import pickle
import json
import matplotlib.pyplot as plt
import time

# GLOBAL VARIABLES
DIM = 28
NUM_QUBITS = 10


###########################
##### Functions #####
###########################

def encode_img(image, register_num):
    ''' encoding of image using QPIE method '''
    img = [j for sub in image for j in sub]
    # img = list(chain(*image))
    pix_val = img

    # normalize
    pix_norm = np.linalg.norm(pix_val)
    arr_norm = np.array(pix_val)/pix_norm
    arr_norm = arr_norm.tolist()

    # Encode onto the quantum register
    qc = QuantumCircuit(register_num)
    test = arr_norm + np.zeros((2**register_num)-DIM**2).tolist()
    qc.initialize(test, qc.qubits)
    return qc


def encode(image):
    ''' final wrapper function (for submission) '''
    return encode_img(255*255*image, register_num=NUM_QUBITS)


def decode_img(histogram):
    ''' decoding (written by prathu) '''
    pixelnums = list(range(DIM**2))
    for pix in pixelnums:
        if pix not in histogram.keys():
            # grayscale pixel value is 0
            histogram.update({pix: 0})

    histnew = dict(sorted(histogram.items()))

    histdata = []
    # for i in enumerate(histnew):
    for i in range(len(histnew)):
        histdata.append(histnew[i])
    histdata = np.array(histdata)
    histarr = np.reshape(histdata, (DIM, DIM))

    return histarr


def decode(histogram):
    ''' final wrapper function (for submission) '''
    return decode_img(histogram)


################
# Helpers
################


def simulate(circuit: QuantumCircuit) -> Dict:
    """Simulate circuit given state vector"""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector(circuit)

    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population

    return histogram


def histogram_to_cat(histogram):
    assert abs(sum(histogram.values()) - 1) < 1e-8
    positive = 0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1] == '0':
            positive += histogram[key]

    return positive


def image_mse(img1, img2):
    return mean_squared_error(img1, img2)


def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """ finds num of gate operations with each num of qubits """
    return Counter([len(gate[1]) for gate in circuit.data])


###########################
##### Main #####
###########################

# load data
images = np.load('../data/images.npy')
labels = np.load('../data/labels.npy')


# test part1
def run_part1(img):
    q_circuit = encode(img)
    histogram = simulate(q_circuit)
    image_re = decode(histogram)
    return q_circuit, image_re


start = 0
stop = 5
images = images[start:stop]
length = len(images)
mse = 0
gatecount = 0
start = time.time()
for image in images:
    # encode
    circuit, image_reconstructed = run_part1(image)
    # count num of 2qubit gates
    gatecount += count_gates(circuit)[2]
    # calculate mse
    mse += image_mse(image, image_reconstructed)

end = time.time()

print('runtime: ', end-start)

# fidelity of reconstruction
f = 1-mse/length
gatecount = gatecount/length

# score
score = f*(0.999**gatecount)

print('fidelity: ', f)
print('gatecount: ', gatecount)
print('score: ', score)
