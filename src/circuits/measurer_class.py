# Standard libraries
from dataclasses import dataclass
from copy import copy

# Local modules
from circuits.circuit_class import CircuitNoLattice, CircuitForLattice
from hamiltonian_class import Hamiltonian

# Third-party libraries
import numpy as np
import qiskit
import qiskit_aer

@dataclass
class CircuitMeasurer:
    circuit_class : CircuitNoLattice | CircuitForLattice
    hamiltonian : Hamiltonian
    simulator : qiskit_aer.AerSimulator
    shots : int = 1024
    
    def __post_init__(self):
        if not isinstance(self.circuit_class, CircuitForLattice) and not isinstance(self.circuit_class, CircuitNoLattice): raise TypeError(f"The circuit must be a circuit_class.CircuitNoLattice or circuit_class.CircuitForLattice, you have entered a {type(self.circuit_class)}")
        if not isinstance(self.hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian for measurement must be a hamiltonian_class.Hamiltonian, you have entered a {type(self.hamiltonian)}")
        if not isinstance(self.simulator, qiskit_aer.AerSimulator): raise TypeError(f"The simulator must be a qiskit_aer.AerSimulator, you have entered a {type(self.simulator)}")
        if not isinstance(self.shots, int): raise TypeError(f"The shots must be an integer, you have entered a {type(self.shots)}")
        
        if self.shots <= 0: raise ValueError(f"Circuit must have positive number of shots, you entered shots = {self.shots}")

        self.n_qubits = self.hamiltonian.n_qubits
        self.MEM = False
        self.MEM_matrix = None
        self.SV = False
        self.SV_allowed_states = None
    
    def self_transpile(self, circuit_copy : qiskit.QuantumCircuit):
        if not isinstance(circuit_copy, qiskit.QuantumCircuit): raise TypeError(f"The circuit must be a qiskit.QuantumCircuit, you have entered a {type(circuit_copy)}")

        return qiskit.transpile(circuit_copy, backend = self.simulator, optimization_level = 1, layout_method = "sabre", routing_method = "sabre", seed_transpiler = 1)
    
    def build_MEM(self):
        calibration_circuits = []
        labels = []
        for i in range(2**self.n_qubits):
            bitstr = format(i, f'0{self.n_qubits}b')
            circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)
            for q, b in enumerate(bitstr):
                if b == '1':
                    circuit.x(q)
            for q in range(self.n_qubits):
                circuit.measure(q, q)
            calibration_circuits.append(circuit)
            labels.append(bitstr)

        transpiled = [self.self_transpile(circuit) for circuit in calibration_circuits]
        job = self.simulator.run(transpiled, shots = self.shots)
        results = job.result()

        matrix = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))
        for j, _ in enumerate(labels):
            counts = results.get_counts(j)
            total = sum(counts.values())
            for k, v in counts.items():
                idx = int(k, 2)
                matrix[idx, j] = v / total
        self.MEM_matrix = matrix
        self.MEM = True

    def add_SV(self, allowed_states : list[str]):
        if not isinstance(allowed_states, list): raise TypeError(f"The allowed states must be given as a list of the strings of the state, you have entered a {type(allowed_states)}")
        for state in allowed_states:
            if not len(state) == self.n_qubits: raise ValueError(f"Allowed states must be over the same number (currently {len(state)}) of qubits as the circuit ({self.n_qubits})")

        self.SV_allowed_states = allowed_states
        self.SV = True

    def global_fold_circuit(self, scale_factor : int):
        if not isinstance(scale_factor, int): raise TypeError(f"The scale_factor must be an integer, you have entered a {type(scale_factor)}")

        if scale_factor < 0: raise ValueError(f"The scale_factor must be greater than or equal to 1, you entered scale_factor = {scale_factor}")
        
        circuit_copy = copy(self.parametrized_circuit)
        if scale_factor == 0:
            return circuit_copy
        folded_circuit = circuit_copy.copy()
        inverse_circuit = circuit_copy.inverse()

        for _ in range(scale_factor):
            folded_circuit.compose(inverse_circuit, inplace = True)
            folded_circuit.compose(circuit_copy, inplace = True)

        return folded_circuit

    def apply_MEM_filter(self, counts : dict):
        if not isinstance(counts, dict): raise TypeError(f"The counts must be a dict, you have entered a {type(counts)}")

        n = 2**self.n_qubits
        p_meas = np.zeros(n)
        for bit, c in counts.items():
            idx = int(bit, 2)
            p_meas[idx] = c
        p_meas /= p_meas.sum()

        try:
            p_true = np.linalg.solve(self.MEM_matrix, p_meas)
        except np.linalg.LinAlgError:
            p_true = np.linalg.lstsq(self.MEM_matrix, p_meas, rcond = None)[0]

        corrected_counts = {}
        total_shots = sum(counts.values())
        for i, p in enumerate(p_true):
            corrected_counts[format(i, f'0{self.n_qubits}b')] = max(p * total_shots, 0.0)
        return corrected_counts
    
    def apply_SV_filter(self, counts : dict):
        if not isinstance(counts, dict): raise TypeError(f"The counts must be a dict, you have entered a {type(counts)}")
        corrected_counts = {}

        for state in counts:
            if state in self.SV_allowed_states:
                corrected_counts[state] = counts[state]
        return corrected_counts

    def change_hamiltonian(self, new_hamiltonian : Hamiltonian):
        if not isinstance(self.hamiltonian, Hamiltonian): raise TypeError(f"The Hamiltonian for measurement must be a hamiltonian_class.Hamiltonian, you have entered a {type(self.hamiltonian)}")

        if not new_hamiltonian.n_qubits == self.n_qubits: raise ValueError(f"New Hamiltonian must have the same number of qubits (currently {new_hamiltonian.n_qubits}) as the circuit ({self.n_qubits})")

        self.hamiltonian = new_hamiltonian        

    def bind_values(self, thetas_values :  list | np.ndarray ):
        if not isinstance(thetas_values, list) and not isinstance(thetas_values, np.ndarray): raise TypeError(f"The values of theta must be given as a list or np.array, you have entered a {type(thetas_values)}")
        
        if not self.circuit_class.n_total_thetas == len(thetas_values): raise ValueError(f"Parameters for binding must be of the same length ({len(thetas_values)}) as the number of parameters in the circuit ({len(self.circuit_class.thetas)})")

        param_dict = dict(zip(self.circuit_class.thetas, thetas_values))

        self.parametrized_circuit = self.circuit_class.circuit.assign_parameters(param_dict)

    def expected_value_from_counts(self, counts : dict, term : str):
        if not isinstance(counts, dict): raise TypeError(f"The counts must be a dict, you have entered a {type(counts)}")
        if not isinstance(term, str): raise TypeError(f"The term must be given as a string, you have entered a {type(term)}")
    
        total_shots = sum(counts.values())
        if total_shots == 0: raise ValueError(f"The counts dictionary has zero total counts")
    
        expected_value = 0
        for bitstring, count in counts.items():
            parity = 1
            for qubit, pauli_term in enumerate(term):
                if pauli_term in ['X', 'Y', 'Z']:
                    if bitstring[self.n_qubits - 1 - qubit] == '1':
                        parity *= -1
            expected_value += parity * count / total_shots
    
        return expected_value

    def measure_value_pauli_term(self, term : str):
        if not isinstance(term, str): raise TypeError(f"The term must be given as a string, you have entered a {type(term)}")
    
        if not len(term) == self.n_qubits: raise ValueError(f"The term must be of the same length (currently {len(term)}) as the number of qubits in the circuit ({self.n_qubits})")
        self.hamiltonian.valid_term_check(term)
    
        circuit_copy = self.parametrized_circuit.copy()
        for qubit, pauli_term in enumerate(term):
            if pauli_term == 'X':
                circuit_copy.h(qubit)
            elif pauli_term == 'Y':
                circuit_copy.sdg(qubit)
                circuit_copy.h(qubit)
    
        if self.SV:
            for qubit in range(self.n_qubits):
                circuit_copy.measure(qubit, qubit)
        else:
            for qubit, pauli_term in enumerate(term):
                if pauli_term in ['X', 'Y', 'Z']:   
                    circuit_copy.measure(qubit, qubit)
    
        transpiled = self.self_transpile(circuit_copy)
        job = self.simulator.run(transpiled, shots = self.shots)
        return job.result().get_counts(transpiled)
    
    def expected_value_pauli_term(self, term : str):
        if not isinstance(term, str): raise TypeError(f"The term must be given as a string, you have entered a {type(term)}")
    
        if not len(term) == self.n_qubits: raise ValueError(f"The term must be of the same length (currently {len(term)}) as the number of qubits in the circuit ({self.n_qubits})")
        self.hamiltonian.valid_term_check(term)
    
        counts = self.measure_value_pauli_term(term)
    
        if self.MEM:
            counts = self.apply_MEM_filter(counts)
        if self.SV:
            counts = self.apply_SV_filter(counts)
    
        return self.expected_value_from_counts(counts, term)
    
    def expected_value_hamiltonian(self):
        expected_value = 0
        for term in self.hamiltonian.terms:
            if term == 'I'*self.n_qubits:
                expected_value += self.hamiltonian.terms[term]
            else:
                expected_value += self.hamiltonian.terms[term] * self.expected_value_pauli_term(term)
        return expected_value
    
    def ZNE_expected_value_hamiltonian(self, max_scale_factor : int):
        if not isinstance(max_scale_factor, int): raise TypeError(f"The scale_factor must be an integer, you have entered a {type(scale_factor)}")

        if max_scale_factor < 0: raise ValueError(f"The scale_factor must be greater than or equal to 1, you entered scale_factor = {max_scale_factor}")
        
        if max_scale_factor == 0: 
            print(f"Warning! You have requested no folding, the original circuit will be returned")
            return self.expected_value_hamiltonian()
        circuit_copy = copy(self.parametrized_circuit)
        scale_factors = list(range(max_scale_factor))
        values = []
        for lam in scale_factors:
            self.parametrized_circuit = circuit_copy
            self.parametrized_circuit = self.global_fold_circuit(lam)
            values.append(self.expected_value_hamiltonian())

        scale_factors_2 = [2 * lam + 1 for lam in scale_factors]
        self.parametrized_circuit = circuit_copy
        coeffs = np.polyfit(scale_factors_2, values, 1)
        return coeffs[-1]
    
    def expected_value_hamiltonian_statevector(self):
        paulis = []
        coeffs = []

        for term, coeff in self.hamiltonian.terms.items():
            paulis.append(term[::-1])
            coeffs.append(coeff)

        operator = qiskit.quantum_info.SparsePauliOp(paulis, coeffs)
        circuit_copy = self.parametrized_circuit.copy()
        psi = qiskit.quantum_info.Statevector.from_instruction(circuit_copy)
        return np.real(psi.expectation_value(operator))