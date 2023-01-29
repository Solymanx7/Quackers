
# QPIE method ( Quantum Probability Image Encoding )


# Importing the libraries
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
from itertools import chain
import matplotlib.pyplot as plt

# GLOBAL VARIABLES
DIM = 28


###########################
##### Functions #####
###########################

# encoding


def encode_img(image, register_num):
    img = list(chain(*image))
    pix_val = img

    # normalize
    pix_norm = np.linalg.norm(pix_val)
    pix_val = np.array(pix_val)
    arr_norm = pix_val/pix_norm
    arr_norm = arr_norm.tolist()

    # Encode onto the quantum register
    qc = QuantumCircuit(register_num)
    # test = arr_norm.append(np.zeros(2**10-arr_norm.shape))
    test = arr_norm + np.zeros(2**register_num-DIM**2).tolist()
    qc.initialize(test, qc.qubits)
    return qc


# decoding (written by prathu)
def decode_img(histogram):
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

# apply qft


def apply_qft(circuit, register_num):
    circuit.append(
        QFT(register_num, do_swaps=False).to_gate(), circuit.qubits)
    return circuit


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

start = 0
stop = 1
n = 10
# encoding
circuit = encode_img(255*255*images[550], n)
# apply classifier
# circuit = apply_qft(circuit, n)
histogram = simulate(circuit)
plot_histogram(histogram)
# qft_label = histogram_to_cat(histogram)
# print(qft_label)
# decoding
fig, ax = plt.subplots(1)
ax.set_aspect('equal')
# this is the array you want to plot
array_plot = decode_img(histogram)
plt.imshow(array_plot, interpolation='nearest', cmap=plt.cm.hsv)
plt.gray()
u = plt.colorbar(fraction=0.046, pad=0.04)
plt.clim(0, np.max(array_plot))


plt.xlim([0, DIM])
plt.ylim([0, DIM])

# set fig size
fig.set_size_inches(5, 5)
fig.set_dpi(100)
# and ticks size
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.gca().invert_yaxis()
plt.show()
# state vector: keys are the pixel position and the values are the normalized grayscale value. pixels with 0 grayscale are left out
# for image in images:
#     while start < stop:
#         circuit = encode_img(255*255*image, 10)
#         histogram = simulate(circuit)
#         print(histogram)
#     start += 1
