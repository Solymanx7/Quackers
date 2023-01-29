
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


###########################
##### Functions #####
###########################

# encoding
def encode_img(image, register_num):
    img = Image.open(image, 'r')
    img = img.convert("L")  # grayscale

    pix_val = list(img.getdata())
    arr_pix_val = np.array(255*255*pix_val)

    # normalize
    pix_norm = np.linalg.norm(arr_pix_val)
    arr_norm = arr_pix_val/pix_norm

    # Eoncde onto the quantum register
    qc = QuantumCircuit(register_num)
    qc.initialize(arr_norm.data, qc.qubits)

    return qc

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
dataset_labels = np.load('data/labels.npy')
print(dataset_labels)
