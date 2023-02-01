class CustomFeatureMap(FeatureMap):
    """Mapping data with a custom feature map."""
    
    def __init__(self, feature_dimension, depth=2, entangler_map=None):
        """
        Args:
            feature_dimension (int): number of features
            depth (int): the number of repeated circuits
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.        
        """
        self._support_parameterized_circuit = False
        self._feature_dimension = feature_dimension
        self._num_qubits = self._feature_dimension = feature_dimension
        self._depth = depth
        self._entangler_map = None
        if self._entangler_map is None:
            self._entangler_map = [[i, j] for i in range(self._feature_dimension) for j in range(i + 1, self._feature_dimension)]

    def apply_qft(circuit, register_num):
        circuit.append(
            QFT(register_num, do_swaps=False).to_gate(), circuit.qubits)
        return circuit

    def encode_img(image, register_num):
        ''' encoding of image using QPIE method '''
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

    def encode(image):
        ''' final wrapper function (for submission) '''
        return encode_img(255*255*image, register_num=NUM_QUBITS)
#         return apply_qft(encode_img(255*255*image, register_num=NUM_QUBITS), register_num=NUM_QUBITS)

    def construct_circuit(self, x, qr, inverse=False):
            """Construct the feature map circuit.
            
            Args:
                x (numpy.ndarray): 1-D to-be-transformed data.
                qr (QauntumRegister): the QuantumRegister object for the circuit.
                inverse (bool): whether or not to invert the circuit.
                
            Returns:
                QuantumCircuit: a quantum circuit transforming data x.
            """
            qc = encode(x)
                        
            if inverse:
                qc.inverse()
            return qc