# Standard libraries
from dataclasses import dataclass, field

# Local modules
from global_helpers import TOL, smart_round

# Third-party libraries
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

PAULI_PHASES = {
    'I': {
        'I': ('I', 1),
        'X': ('X', 1),
        'Y': ('Y', 1),
        'Z': ('Z', 1),
    },
    'X': {
        'I': ('X', 1),
        'X': ('I', 1),
        'Y': ('Z', 1j),
        'Z': ('Y', -1j),
    },
    'Y': {
        'I': ('Y', 1),
        'X': ('Z', -1j),
        'Y': ('I', 1),
        'Z': ('X', 1j),
    },
    'Z': {
        'I': ('Z', 1),
        'X': ('Y', 1j),
        'Y': ('X', -1j),
        'Z': ('I', 1),
    },
}

VALID_PAULIS = {"I", "X", "Y", "Z"}

@dataclass
class Hamiltonian:
    n_qubits : int = 1
    terms : dict[str, complex] = field(default_factory = dict)

    def __post_init__(self):
        if not isinstance(self.n_qubits, int): raise TypeError(f"n_qubits must be an integer, you have entered a {type(self.n_qubits)}")
        if not isinstance(self.terms, dict): raise TypeError(f"The terms must be a dict, you have entered a {type(self.terms)}")

        if not self.terms:
            self.terms = {"I" * self.n_qubits: 0.0}

        if self.n_qubits <= 0: raise ValueError(f"Hamiltonian must have positive number of qubits, you entered n_qubits = {self.n_qubits}")

        for term in self.terms:
            self.valid_term_check(term)

    def valid_term_check(self, _term : str):
        if not isinstance(_term, str): raise TypeError(f"Term must be a string, you have entered a {type(_term)}")

        if len(_term) != self.n_qubits: raise ValueError(f"Terms must be of length {self.n_qubits} for this Hamiltonian, you have entered a term of length {len(_term)}")

        invalid_check = set(_term) - VALID_PAULIS
        if invalid_check: raise ValueError(f"Terms must be from Pauli matrices ('I', 'X', 'Y', 'Z'), you have entered a term = {_term}")

    def valid_coeff_check(self, coeff: complex) -> None:
        if not isinstance(coeff, int | float | complex | np.number): raise TypeError(f"Coefficients must be numeric, you entered {coeff} with type {type(coeff)}")

    def cleanup(self):
        new_terms = {}
        for term in self.terms:
            if np.abs(self.terms[term]) > TOL:
                new_terms[term] = self.terms[term]
                if abs(self.terms[term].imag) > TOL:
                    print(f"Warning! This Hamiltonian will be non-Hermitian, due to the term: {term} : {self.terms[term]}")
                else:
                    new_terms[term] = self.terms[term].real
        self.terms = new_terms

    def conjugate(self):
        new_terms = {}
        for term in self.terms:
            new_terms[term] = self.terms[term].real - 1j * self.terms[term].imag
        self.terms = new_terms

    def reverse_qubit_ordering(self):
        new_terms = {}
        for term in self.terms:
            new_terms[term[::-1]] = self.terms[term]
        self.terms = new_terms

    def add_term(self, _term : str, _coeff : complex):
        self.valid_term_check(_term)
        self.valid_coeff_check(_coeff)

        if _term in self.terms:
            self.terms[_term] += _coeff
        else:
            self.terms[_term] = _coeff

    def add_terms(self, _terms : list[str], _coeffs : list[complex]):
        if len(_terms) != len(_coeffs): raise ValueError(f"List of terms and list of coefficients should be the same length, you entered with len(terms) = {len(_terms)} and len(coefficients) = {len(_coeffs)}")

        for k in range(len(_terms)):
            self.add_term(_terms[k], _coeffs[k])

    def add_hamiltonian(self, _other : "Hamiltonian | dict[str, complex]"):
        if isinstance(_other, Hamiltonian): other = _other.terms
        elif isinstance(_other, dict): other = _other
        else: raise TypeError(f"Can only add together Hamiltonians that are of type = dict or hamiltonian_class.Hamiltonian, you have entered a {type(_other)}")

        terms = list(other.keys())
        coeffs = list(other.values())
        self.add_terms(terms, coeffs)

    def multiply_by_constant(self, _constant):
        if not isinstance(_constant, int | float | complex | np.number): raise TypeError(f"Constant must be numeric, you entered {_constant} with type {type(_constant)}")
        new_terms = {}
        for term in self.terms:
            new_terms[term] = _constant * self.terms[term]
        self.terms = new_terms

    def multiply_terms(self, _term0, _term1):
        self.valid_term_check(_term0)
        self.valid_term_check(_term1)

        total_term = []
        total_phase = 1 + 0j
        for a, b in zip(_term0, _term1):
            temp_term, temp_phase = PAULI_PHASES[a][b]
            total_term.append(temp_term)
            total_phase *= temp_phase
        return "".join(total_term), total_phase  

    def multiply_by_hamiltonian(self, _other):
        if isinstance(_other, Hamiltonian): other = _other.terms
        elif isinstance(_other, dict): other = _other
        else: raise TypeError(f"Can only multiply together Hamiltonians that are of type = dict or hamiltonian_class.Hamiltonian, you have entered a {type(_other)}")
            
        new_terms = {}
        for self_term, self_coeff in self.terms.items():
            for other_term, other_coeff in other.items():
                new_term, phase = self.multiply_terms(self_term, other_term)
                new_coeff = self_coeff * other_coeff * phase
                if abs(new_coeff.imag) < 1e-12:
                    new_coeff = new_coeff.real
                if new_term in new_terms:
                    new_terms[new_term] += new_coeff
                else:
                    new_terms[new_term] = new_coeff
        self.terms = new_terms

    def commutator(self, _other):
        if isinstance(_other, Hamiltonian): other = _other.terms
        elif isinstance(_other, dict): other = _other
        else: raise TypeError(f"Can only compute a commutator with a Hamiltonian that are of type = dict or hamiltonian_class.Hamiltonian, you have entered a {type(_other)}")

        new_terms = {}
        for self_term, self_coeff in self.terms.items():
            for other_term, other_coeff in other.items():
                new_term, forward_phase = self.multiply_terms(self_term, other_term)
                _, backward_phase = self.multiply_terms(other_term, self_term)
                new_coeff = self_coeff * other_coeff * (forward_phase - backward_phase)
                if abs(new_coeff.imag) < 1e-12:
                    new_coeff = new_coeff.real
                if new_coeff != 0:
                    if new_term in new_terms:
                        new_terms[new_term] += new_coeff
                    else:
                        new_terms[new_term] = new_coeff

        return Hamiltonian(n_qubits = self.n_qubits, terms = new_terms if new_terms else {"I" * self.n_qubits: 0.0})

    def to_matrix(self):
        matrix_dict = {
            'I': np.array([[1, 0], [0, 1]], dtype = complex),
            'X': np.array([[0, 1], [1, 0]], dtype = complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype = complex),
            'Z': np.array([[1, 0],[0, -1]], dtype = complex)
            }
        matrix = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits), dtype = complex)
        for term in self.terms:
            temp_matrix = np.array([1])
            for pauli_term in list(term):
                temp_matrix = np.kron(matrix_dict[pauli_term], temp_matrix)
            matrix += self.terms[term] * temp_matrix
        return matrix
    
    def to_sparse_matrix(self):
        matrix_dict = {'I': sp.csc_matrix([[1, 0], [0, 1]], dtype = complex), 
                       'X': sp.csc_matrix([[0, 1], [1, 0]], dtype = complex), 
                       'Y': sp.csc_matrix([[0, -1j], [1j, 0]], dtype = complex), 
                       'Z': sp.csc_matrix([[1, 0], [0, -1]], dtype = complex)}
        matrix = sp.csc_matrix((2 ** self.n_qubits, 2 ** self.n_qubits), dtype = complex)
        for term, coeff in self.terms.items():
            temp_matrix = sp.csc_matrix([1], dtype = complex)
            for pauli_term in term:
                temp_matrix = sp.kron(matrix_dict[pauli_term], temp_matrix, format = 'csc')
            matrix += coeff * temp_matrix
        return matrix
        
    def _popcount(self, integer_array):
        return np.bitwise_count(integer_array).astype(np.int64)
        
    def to_linear_operator(self):
        dimension = 2 ** self.n_qubits
        masks = []
        for term, coeff in self.terms.items():
            x_mask = z_mask = y_count = 0
            for qubit_k in range(self.n_qubits):
                p = term[qubit_k]
                if p == 'X': x_mask |= (1 << qubit_k)
                elif p == 'Z': z_mask |= (1 << qubit_k)
                elif p == 'Y':
                    x_mask |= (1 << qubit_k); z_mask |= (1 << qubit_k); y_count += 1
            phase = complex(coeff) * ((-1j) ** y_count)
            masks.append((phase, x_mask, z_mask))   # tiny: 3 scalars per term

        indices = np.arange(dimension, dtype=np.int64)   # one 8.6 GB array, shared

        def matvec(vector):
            vector = np.asarray(vector, dtype=np.complex128).reshape(-1)
            result = np.zeros(dimension, dtype=np.complex128)
            for phase, x_mask, z_mask in masks:
                if z_mask:
                    signs = 1 - 2 * (self._popcount(indices & z_mask) & 1)
                    gathered = vector[indices ^ x_mask] if x_mask else vector
                    result += (phase * signs) * gathered
                else:
                    result += phase * (vector[indices ^ x_mask] if x_mask else vector)
            return result
        return spl.LinearOperator((dimension, dimension), matvec=matvec, dtype=np.complex128)
    
    def latex_print(self, _to_print : bool = True):
        if not isinstance(_to_print, bool): raise TypeError(f"to_print is a boolean check, you have entered a {type(_to_print)}")

        string_to_print = ""
        rounded_terms = smart_round(self.terms.copy(), 5)
        counter  = 0
        for term in rounded_terms:
            term_list = list(term)
            if term_list == ['I'] * len(term_list):
                string_to_print += str(rounded_terms[term])
            else:
                temp_string = ""
                for i in range(len(term_list)):
                    if term_list[i] != 'I':
                        temp_string += term_list[i] + r"_{" + str(i) + r"} "    
                              
                if rounded_terms[term] < 0:
                    string_to_print += r" - " + str(abs(rounded_terms[term])) + " " + temp_string
                else:
                    string_to_print += r" + " + str(abs(rounded_terms[term])) + " " + temp_string
                counter += 1
                if (counter % 4) == 0:
                    string_to_print += r"\\ &"
        if _to_print: print(string_to_print)
        return string_to_print