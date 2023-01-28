
# QPIE method ( Quantum Probability Image Encoding )


# Importing the libraries
import numpy as np
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, transpile, assemble
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from PIL import Image


################
# Functions
################

# encoding
def encode_img(image, n):
    img = Image.open(image, 'r')
    img = img.convert("L")  # grayscale

    pix_val = list(img.getdata())
    arr_pix_val = np.array(255*255*pix_val)

    # normalize
    pix_norm = np.linalg.norm(arr_pix_val)
    arr_norm = arr_pix_val/pix_norm

    # Eoncde onto the quantum register
    qc = QuantumCircuit(n)
    qc.initialize(arr_norm.data, qc.qubits)

    return qc
