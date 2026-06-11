# Standard libraries
from dataclasses import dataclass, field
import itertools
import copy

# Local modules

# Third-party libraries
import numpy as np
import sympy

@dataclass
class Lattice:
    L_x : int = 2
    L_y : int = 2
    n_g : int = 2
    dynamical_links_list : list[tuple] = field(default_factory = list)
    charge_site : tuple = ()
    anticharge_site : tuple = ()
    background_field : list = field(default_factory = lambda: [0.0, 0.0])

    def __post_init__(self):
        if not isinstance(self.L_x, int): raise TypeError(f"L_x must be an integer, you have entered a {type(self.L_x)}")
        if not isinstance(self.L_y, int): raise TypeError(f"L_y must be an integer, you have entered a {type(self.L_y)}")
        if not isinstance(self.n_g, int): raise TypeError(f"n_g must be an integer, you have entered a {type(self.n_g)}")
        if not isinstance(self.dynamical_links_list, list): raise TypeError(f"The terms must be a dict, you have entered a {type(self.dynamical_links_list)}")
        if not isinstance(self.charge_site, tuple): raise TypeError(f"The charge site must be a tuple of site coordinates, you have entered a {type(self.charge_site)}")
        if not isinstance(self.anticharge_site, tuple): raise TypeError(f"The anticharge site must be a tuple of site coordinates, you have entered a {type(self.anticharge_site)}")
        if not isinstance(self.background_field, list) and len(self.background_field) != 2: raise TypeError(f"The background electric field must be a list of the field strength in the two directions, you have entered a {type(self.background_field)}")

        if self.L_x <= 0: raise ValueError(f"Lattice must have positive number of sites in the x direction, you entered L_x = {self.L_x}")
        if self.L_y <= 0: raise ValueError(f"Lattice must have positive number of sites in the y direction, you entered L_y = {self.L_y}")
        if (self.L_x * self.L_y % 2) != 0: raise ValueError(f"Lattice must have an even number of sites, you entered L_x * L_y = {self.L_x} * {self.L_y} = {self.L_x * self.L_y}")

        counter = 0
        self.labels = {}
        self.reverse_labels = {}
        for y in range(self.L_y):
            for x in range(self.L_x):
                self.labels[counter] = (x, y)
                self.reverse_labels[(x, y)] = counter
                counter += 1
        self.n_fermion_qubits = counter
        self.links_list = self.possible_links()

        if self.charge_site not in self.reverse_labels and self.charge_site != (): raise ValueError(f"The charge site must be on the lattice, you entered charge_site = {self.charge_site}")
        if self.anticharge_site not in self.reverse_labels and self.anticharge_site != (): raise ValueError(f"The anticharge site must be on the lattice, you entered charge_site = {self.anticharge_site}")
        for link in self.dynamical_links_list:
            if link not in self.links_list: raise ValueError(f"The dynamical charges must be on the lattice, you entered dynamical link = {link}")

        if self.dynamical_links_list == []:
            self.dynamical_links_list = self.links_list
        elif self.dynamical_links_list == ['none']:
            self.dynamical_links_list = []
        else:
            self.dynamical_links_list = self.dynamical_links_list

        self.directions, self.link_indexing, self.dynamical_link_indexing = self.link_information()
        self.plaquettes = self.possible_plaquettes()
        self.U_terms, self.E_terms = self.gauge_terms()

        self.gauss_equations = self.gauss_solver()

        self.n_links = (self.L_x - 1) * self.L_y + self.L_x * (self.L_y - 1)
        self.n_gauge_qubits = self.n_links * self.n_g 
        self.n_dynamical_links = len(self.dynamical_links_list)
        self.n_dynamical_gauge_qubits = self.n_dynamical_links * self.n_g

        self.n_qubits = self.n_fermion_qubits + self.n_dynamical_gauge_qubits

        if self.n_dynamical_links < self.n_links - (self.n_fermion_qubits - 1): print(f"This lattice requires at least {self.n_links - (self.n_fermion_qubits - 1)} dynamical links, you entered {self.n_dynamical_links}, which has {len(self.n_dynamical_links)} dynamical links")
        
    def link_information(self):
        labels = self.labels
        possible_labels = list(self.reverse_labels)
        directions = {}
        link_indexing = {}
        dynamical_link_indexing = {}
        counter = 0
        dynamical_counter = 0
        for j in range(len(labels)):
            directions[j] = []
            if (labels[j][0] + 1, labels[j][1]) in possible_labels: # x_link
                directions[j].append(1)
                link_indexing[((labels[j][0], labels[j][1]), 1)] = counter
                counter += 1
            if (labels[j][0], labels[j][1]+1) in possible_labels: # y_link
                directions[j].append(2)
                link_indexing[((labels[j][0], labels[j][1]), 2)] = counter
                counter += 1
            
            if ((labels[j][0], labels[j][1]), 1) in self.dynamical_links_list:
                dynamical_link_indexing[((labels[j][0], labels[j][1]), 1)] = dynamical_counter
                dynamical_counter += 1
            if ((labels[j][0], labels[j][1]), 2) in self.dynamical_links_list:
                dynamical_link_indexing[((labels[j][0], labels[j][1]), 2)] = dynamical_counter
                dynamical_counter += 1
        return directions, link_indexing, dynamical_link_indexing

    def possible_links(self):
        labels = self.labels
        possible_labels = list(self.reverse_labels)
        links = []
        for j in range(len(labels)):
            if (labels[j][0] + 1, labels[j][1]) in possible_labels: # x_link
                links.append(((labels[j][0], labels[j][1]), 1))
            if (labels[j][0], labels[j][1]+1) in possible_labels: # y_link
                links.append(((labels[j][0], labels[j][1]), 2))
        return links

    def possible_plaquettes(self):
        plaquettes = []
        indices_list = list(self.reverse_labels)
        for key in self.labels:
            indices = self.labels[key]
            if (indices[0], indices[1] + 1) in indices_list:
                if (indices[0] + 1, indices[1]) in indices_list:
                    if (indices[0] + 1, indices[1] + 1) in indices_list:
                        plaquettes.append(key)
        return plaquettes

    def gauge_terms(self):
        matrix_dict = {
            'I': np.array([[1, 0], [0, 1]], dtype = complex),
            'X': np.array([[0, 1], [1, 0]], dtype = complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype = complex),
            'Z': np.array([[1, 0],[0, -1]], dtype = complex)
            }

        mappings = {2: {
                        "00": "01",
                        "01": "11",
                        "11": "00",
                        "10": "10"
                        },
                    3: {
                        "000": "001",
                        "001": "010",
                        "010": "011",
                        "011": "101",
                        "101": "111",
                        "111": "110",
                        "110": "000",
                        "100": "100",
                    }
                    }
        U_terms = {}
        E_terms = {}
        U = np.zeros((2 ** self.n_g, 2 ** self.n_g), dtype=complex)
        for from_, to_ in mappings[self.n_g].items():
            i = int(from_, 2)
            j = int(to_, 2)
            U[j, i] = 1.0
        for labels in itertools.product("IXYZ", repeat = self.n_g):
            term = matrix_dict[labels[0]]
            for l in labels[1:]:
                term = np.kron(term, matrix_dict[l])

            coeff = np.trace(term.conj().T @ U) / (2 ** self.n_g)
            if abs(coeff) > 1e-12:
                U_terms["".join(labels)] = coeff
        I_string_list = list('I' * self.n_g)
        coeff = -0.5
        for k in range(self.n_g - 1):
            temp_string = copy.copy(I_string_list)
            temp_string[k] = 'Z'
            E_terms["".join(temp_string)] = coeff * 2**(k)
            
        first_string = copy.copy(I_string_list)    
        first_string[self.n_g - 1] = 'Z'
        E_terms["".join(first_string)] = coeff * (2**(self.n_g - 1) - 1)      
        return U_terms, E_terms

    def gauss_solver(self):     
        link_variables = []
        charge_variables = []
        link_variable_dict = {}
        reverse_link_variable_dict = {}
        for k in range(len(self.links_list)):
            n = self.reverse_labels[self.links_list[k][0]]
            direction = self.links_list[k][1]
            
            link_variables.append(sympy.symbols(f"E{n}{direction}"))
            
            link_variable_dict[self.links_list[k]] = link_variables[-1]
            reverse_link_variable_dict[link_variables[-1]] = self.links_list[k]
            
        eqs = []
        dependant_variables = []
        independant_variables = []
        for n in range(self.n_fermion_qubits):           
            q = sympy.symbols(f"q{n}")
            charge_variables.append(q)
            G_n = 0
            site = self.labels[n]
            if site == self.charge_site:
                G_n += 1
            elif site == self.anticharge_site:
                G_n -= 1
            for direction in [1,2]:
                prev_site = list(copy.copy(site))
                prev_site[direction - 1] -= 1
                prev_site = tuple(prev_site)
                if (prev_site, direction) in self.links_list:
                    G_n += link_variable_dict[(prev_site, direction)]
                    
                if (site, direction) in self.links_list:
                    G_n -= link_variable_dict[(site, direction)]
                
                    if (site, direction) not in self.dynamical_links_list:
                        dependant_variables.append(link_variable_dict[(site, direction)]) 
                    else:
                        independant_variables.append(link_variable_dict[(site, direction)]) 
            G_n -= q
            eqs.append(G_n)
            
        independant_variables += charge_variables
        
        sol = sympy.solve(eqs[:-1], dependant_variables, dict=True)

        return {'solution': sol[0], 
                'dependant_variables': dependant_variables,
                'independant_variables': independant_variables,
                'link_variable_dict': link_variable_dict,
                'reverse_link_variable_dict': reverse_link_variable_dict,
                'equations': eqs,
                'link_variables' : link_variables,
                'charge_variables': charge_variables}

    def get_label(self, x, y):
        if (x, y) in self.reverse_labels:
            return self.reverse_labels[(x, y)]
        raise ValueError(f"The site (coordinate) must be on the lattice, you entered x = {x}, y = {y}")

    def get_indices(self, label):
        if label in self.labels:
            return self.labels[label]
        raise ValueError(f"The site (label) must be on the lattice, you entered label = {label}")

    def get_neighbors(self, x, y):
        if (x, y) not in self.reverse_labels: raise ValueError(f"The site (coordinate) must be on the lattice, you entered x = {x}, y = {y}")
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.L_x and 0 <= ny < self.L_y:
                neighbors.append((nx, ny))
        return neighbors
    
    def SV_strings(self):    
        accepted_single_gauge_parts = {
            2: ["00", "01", "11"],
            3: ["000", "001", "010", "011", "101", "111", "110"]
        }
    
        if self.n_g not in accepted_single_gauge_parts: raise ValueError(f"No SV gauge strings have been implemented for n_g = {self.n_g}")
    
        accepted_gauge_parts = []
        for terms in itertools.product(accepted_single_gauge_parts[self.n_g], repeat = self.n_dynamical_links):
            accepted_gauge_parts.append("".join(terms))
    
        accepted_fermion_parts = [] 
        strings = [format(k, f'0{self.n_fermion_qubits}b') for k in range(2**self.n_fermion_qubits)]
    
        for string in strings:
            string_charge = 0
            for site_n in range(self.n_fermion_qubits):
                parity = (-1)**(self.labels[site_n][0] + self.labels[site_n][1])
                if string[site_n] == "0":
                    Z_value = 1
                else:
                    Z_value = -1
                string_charge += -0.5 * Z_value + 0.5 * parity
    
            if string_charge == 0:
                accepted_fermion_parts.append(string)
    
        accepted_strings = []
        for gauss_string in accepted_gauge_parts:
            for fermion_string in accepted_fermion_parts:
                accepted_strings.append(fermion_string + gauss_string)
    
        for state in accepted_strings:
            if not len(state) == self.n_qubits: raise ValueError(f"SV state has length {len(state)}, but n_qubits = {self.n_qubits}")
    
        return accepted_strings