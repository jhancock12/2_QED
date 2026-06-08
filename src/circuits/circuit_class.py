# Standard libraries
from copy import copy
from dataclasses import dataclass

# Local modules
from lattice_class import Lattice

# Third-party libraries
import qiskit
import numpy as np

@dataclass
class CircuitNoLattice: # WORK IN PROGRESS
    n_qubits : int = 1
    n_layers : int = 1

    def __post_init__(self):
        if not isinstance(self.n_qubits, int): raise TypeError(f"The n_qubits must be an integer, you have entered a {type(self.n_qubits)}")
        if not isinstance(self.n_layers, int): raise TypeError(f"The n_thetas must be an integer, you have entered a {type(self.n_layers)}")

        if self.n_qubits <= 0: raise ValueError(f"Circuit must have positive number of qubits, you entered n_fermion_layers = {self.n_qubits}")
        if self.n_layers < 0: raise ValueError(f"Circuit must have positive number of parameters, you entered n_fermion_layers = {self.n_layers}")

        self.n_total_thetas = self.n_layers * self.n_qubits
        self.circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)
        self.thetas = qiskit.circuit.ParameterVector('x', self.n_total_thetas)
    
        self.build_circuit()        

    def r_y_layer(self, thetas_slice : list | np.ndarray):
        if not isinstance(thetas_slice, list) and not isinstance(thetas_slice, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_slice)}")
        
        if not self.n_qubits == len(thetas_slice): raise ValueError(f"Parameters for each layer must have the same number ({len(thetas_slice)}) as the number of qubits in the circuit ({len(self.n_qubits)})")

        for qubit in range(self.n_qubits):
            self.circuit.ry(thetas_slice[qubit], qubit)

    def cnot_layer(self):
        for qubit in range(self.n_qubits - 1):
            self.circuit.cx(qubit, qubit + 1)

    def build_circuit(self):
        for j in range(self.n_layers):
            thetas = self.thetas[self.n_qubits * j : self.n_qubits * (j + 1)]
            self.r_y_layer(thetas)
            self.cnot_layer()
            self.circuit.barrier()

    def print_circuit(self):
        print(self.circuit.draw())

    def get_circuit(self):
        return copy(self.circuit)

@dataclass
class CircuitForLattice:
    lattice : Lattice  
    n_fermion_layers : int = 1

    def __post_init__(self):
        if not isinstance(self.lattice, Lattice): raise TypeError(f"The lattice must be a lattice_class.Lattice, you have entered a {type(self.lattice)}")
        if not isinstance(self.n_fermion_layers, int): raise TypeError(f"The n_fermion_layers must be an integer, you have entered a {type(self.n_fermion_layers)}")

        if self.n_fermion_layers <= 0: raise ValueError(f"Circuit must have positive number of n_fermion_layers, you entered n_fermion_layers = {self.n_fermion_layers}")

        self.circuit = qiskit.QuantumCircuit(self.lattice.n_qubits, self.lattice.n_qubits)
        self.n_slice = self.iSwap_block_calculate_qed()
        self.thetas_per_gauge = {2: 2, 
                                 3: 4}
        self.n_gauge_thetas = self.thetas_per_gauge[self.lattice.n_g] * self.lattice.n_dynamical_links
        self.n_fermion_thetas = self.n_slice * self.n_fermion_layers
        self.n_total_thetas = self.n_fermion_thetas + self.n_gauge_thetas
        self.thetas = qiskit.circuit.ParameterVector('x', self.n_total_thetas)
        self.fermion_thetas = self.thetas[:self.n_fermion_thetas]
        self.gauge_thetas = self.thetas[self.n_fermion_thetas:self.n_fermion_thetas + self.n_gauge_thetas]
        self.gauge_gates = {2: self.gauge_gate_2,
                            3: self.gauge_gate_3}
        
        self.build_circuit()

    def gauge_gate_2(self, thetas: list | np.ndarray, start_qubit):
        if not isinstance(thetas, list) and not isinstance(thetas, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_slice)}")
        
        if not len(thetas) == 2: raise ValueError(f"Gauge gates with n_g = 2 require only 2 parameters, you entered {len(thetas)}")

        self.circuit.ry(thetas[0], start_qubit)
        self.circuit.cry(thetas[1], start_qubit, start_qubit+1)
        return self
    
    def gauge_gate_3(self, thetas: list | np.ndarray, start_qubit):
        if not isinstance(thetas, list) and not isinstance(thetas, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_slice)}")
        
        if not len(thetas) == 2: raise ValueError(f"Gauge gates with n_g = 3 require only 4 parameters, you entered {len(thetas)}")
        self.circuit.ry(thetas[0], start_qubit)
        self.circuit.ry(thetas[1], start_qubit+1)
        self.circuit.cry(thetas[2], start_qubit+1, start_qubit+2)
        self.circuit.cry(thetas[3], start_qubit, start_qubit+2)
        return self

    def iSwap(self, theta, j : int, k : int):
        if not isinstance(j, int): raise TypeError(f"j must be an integer, you have entered a {type(j)}")
        if not isinstance(k, int): raise TypeError(f"k must be an integer, you have entered a {type(k)}")

        if not 0 <= j < self.lattice.n_qubits: raise ValueError(f"j must be positive and on the circuit, you entered with j = {j}, self.lattice.n_qubits = {self.lattice.n_qubits}")
        if not 0 <= k < self.lattice.n_qubits: raise ValueError(f"k must be positive and on the circuit, you entered with k = {k}, self.lattice.n_qubits = {self.lattice.n_qubits}")
        if j == k: raise ValueError(f"j and k must be different qubits, you entered j = {j}, k = {k}")

        self.circuit.ryy(theta/2, j, k)
        self.circuit.rxx(theta/2, j, k)
        return self
        
    def iSwap_block_calculate_qed(self):
        possible_pairs = []

        for i in range(self.lattice.n_fermion_qubits):
            for j in range(i):
                if i != j:
                    indices_i = self.lattice.labels[i]
                    indices_j = self.lattice.labels[j]

                    dx = abs(indices_i[0] - indices_j[0])
                    dy = abs(indices_i[1] - indices_j[1])

                    if dx + dy == 1:
                        possible_pairs.append([j,i])

        return len(possible_pairs)
    
    def iSwap_block_qed(self, thetas_slice: list | np.ndarray):
        if not isinstance(thetas_slice, list) and not isinstance(thetas_slice, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_slice)}")

        if not len(thetas_slice) == self.n_slice: raise ValueError(f"{self.n_slice} parameters are required for the iSWAP blocks, you entered {len(thetas_slice)}")

        possible_pairs = []

        for i in range(self.lattice.n_fermion_qubits):
            for j in range(i):
                if i != j:
                    indices_i = self.lattice.labels[i]
                    indices_j = self.lattice.labels[j]

                    dx = abs(indices_i[0] - indices_j[0])
                    dy = abs(indices_i[1] - indices_j[1])

                    if dx + dy == 1:
                        possible_pairs.append([j,i])

        counter = 0
        for pair in possible_pairs:
            self.iSwap(thetas_slice[counter], self.lattice.n_dynamical_gauge_qubits + pair[0], self.lattice.n_dynamical_gauge_qubits + pair[1])
            counter += 1
        self.circuit.barrier()
        return self
        
    def initialize_fermions(self):        
        for x in range(self.lattice.L_x):
            for y in range(self.lattice.L_y):
                if (x + y) % 2 == 1:
                    self.circuit.x(self.lattice.n_dynamical_gauge_qubits + self.lattice.reverse_labels[(x, y)])

    def initialize_gauge(self):
        for n in range(self.lattice.n_dynamical_links):
            self.circuit.x(self.lattice.n_g * n + 1)
        self.circuit.barrier()

    def parametrize_fermions(self):
        for j in range(self.n_fermion_layers):
            self.iSwap_block_qed(self.fermion_thetas[self.n_slice * j : self.n_slice * (j + 1)])

    def parametrize_gauge(self):       
        for j in range(int(self.lattice.n_dynamical_gauge_qubits / self.lattice.n_g)):
            thetas = self.gauge_thetas[self.thetas_per_gauge[self.lattice.n_g] * j : self.thetas_per_gauge[self.lattice.n_g] * j + (self.thetas_per_gauge[self.lattice.n_g])]
            self.gauge_gates[self.lattice.n_g](thetas, self.lattice.n_g*j)
        self.circuit.barrier()
        return self
    
    def build_circuit(self):
        self.initialize_fermions()
        self.parametrize_gauge()
        self.parametrize_fermions()

    def print_circuit(self):
        print(self.circuit.draw())

    def get_circuit(self):
        return copy(self.circuit)
     

