# Standard libraries
import copy

# Local modules
from hamiltonian_class import Hamiltonian
from lattice_class import Lattice

# Third-party libraries
   
def electric_n_direction(lattice : Lattice, site_n : int, direction : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")
    if not isinstance(direction, int): raise TypeError(f"direction must be an integer, you have entered a {type(direction)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")
    if direction not in [1, 2]: raise ValueError(f"direction must be either 1 or 2, you entered direction = {direction}")
    if direction not in lattice.directions[site_n]: raise ValueError(f"The direction must be a possible direction from site_n = {site_n}, you entered direction = {direction}")
    
    hamiltonian = Hamiltonian(lattice.n_qubits)
    if (lattice.labels[site_n], direction) in lattice.dynamical_links_list:
        link_index = lattice.dynamical_link_indexing[(lattice.labels[site_n], direction)]
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * ((link_index) * lattice.n_g)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.n_g)

        for key in list(lattice.E_terms):
            term = gauge_before + key + gauge_after + fermion_string
            hamiltonian.add_term(term, lattice.E_terms[key])
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 0)

    return hamiltonian

def electric_field_n(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")

    hamiltonian = Hamiltonian(lattice.n_qubits)
    for direction in lattice.directions[site_n]:
        E_temp = electric_n_direction(lattice, site_n, direction)
        hamiltonian.add_hamiltonian(E_temp)

    return hamiltonian

def U_n_direction(lattice : Lattice, site_n : int, direction : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")
    if not isinstance(direction, int): raise TypeError(f"direction must be an integer, you have entered a {type(direction)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")
    if direction not in [1, 2]: raise ValueError(f"direction must be either 1 or 2, you entered direction = {direction}")
    if direction not in lattice.directions[site_n]: raise ValueError(f"The direction must be a possible direction from site_n = {site_n}, you entered direction = {direction}")

    hamiltonian = Hamiltonian(lattice.n_qubits)
    if (lattice.labels[site_n], direction) in lattice.dynamical_links_list:
        link_index = lattice.dynamical_link_indexing[(lattice.labels[site_n], direction)]
        fermion_string = 'I' * lattice.n_fermion_qubits

        gauge_before = 'I' * (link_index * lattice.n_g)
        gauge_after = 'I' * (lattice.n_dynamical_gauge_qubits - (link_index + 1) * lattice.n_g)
        
        for key in list(lattice.U_terms):
            term = gauge_before + key + gauge_after + fermion_string
            hamiltonian.add_term(term, lattice.U_terms[key])       
    else:
        hamiltonian.add_term('I'*lattice.n_qubits, 1.0)

    return hamiltonian

def magnetic_term_n(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")
    if site_n not in lattice.plaquettes: raise ValueError(f"site_n must be the bottom-left site of a plaquette, you entered site_n = {site_n}")

    index = lattice.labels[site_n]
    ns_directions = [
        (lattice.reverse_labels[(index[0], index[1])], 1),
        (lattice.reverse_labels[(index[0] + 1, index[1])], 2),
        (lattice.reverse_labels[(index[0], index[1] + 1)], 1),
        (lattice.reverse_labels[(index[0], index[1])], 2)
    ]

    Us = [Hamiltonian(lattice.n_qubits) for _ in range(4)]
    for i in range(4): Us[i] = U_n_direction(lattice, ns_directions[i][0], ns_directions[i][1])

    Us[2].conjugate()
    Us[3].conjugate()

    P_n = Hamiltonian(lattice.n_qubits)
    P_n.add_term('I'*lattice.n_qubits, 1.0)

    for i in range(4): P_n.multiply_by_hamiltonian(Us[i])

    P_n_dagger = copy.copy(P_n)
    P_n_dagger.conjugate()

    hamiltonian = Hamiltonian(lattice.n_qubits)
    hamiltonian.add_hamiltonian(P_n)
    hamiltonian.add_hamiltonian(P_n_dagger)

    return hamiltonian

def creation_operator_n(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")

    hamiltonian = Hamiltonian(lattice.n_qubits)

    gauge_string = 'I' * lattice.n_dynamical_gauge_qubits
    fermions_before = 'Z' * site_n
    fermions_after = 'I' * (lattice.n_fermion_qubits - (site_n + 1))
    coeff = (1j)**site_n

    hamiltonian.add_term(gauge_string + fermions_before + 'X' + fermions_after, coeff / 2)
    hamiltonian.add_term(gauge_string + fermions_before + 'Y' + fermions_after, -1j * coeff / 2)

    return hamiltonian

def annihilation_operator_n(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")

    hamiltonian = Hamiltonian(lattice.n_qubits)

    gauge_string = 'I' * lattice.n_dynamical_gauge_qubits
    fermions_before = 'Z' * site_n
    fermions_after = 'I' * (lattice.n_fermion_qubits - (site_n + 1))
    coeff = (-1j)**site_n

    hamiltonian.add_term(gauge_string + fermions_before + 'X' + fermions_after, coeff / 2)
    hamiltonian.add_term(gauge_string + fermions_before + 'Y' + fermions_after, 1j * coeff / 2)

    return hamiltonian

def mass_term_n(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")

    hamiltonian = creation_operator_n(lattice, site_n)
    annihilation_operator = annihilation_operator_n(lattice, site_n)
    
    hamiltonian.multiply_by_hamiltonian(annihilation_operator)
    
    indices = lattice.labels[site_n]
    coeff = ((-1)**(indices[0] + indices[1]))
    hamiltonian.multiply_by_constant(coeff)
    
    return hamiltonian

def kinetic_subterm_n(lattice : Lattice, site_n : int, direction : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")
    if not isinstance(direction, int): raise TypeError(f"direction must be an integer, you have entered a {type(direction)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")
    if direction not in [1, 2]: raise ValueError(f"direction must be either 1 or 2, you entered direction = {direction}")
    if direction not in lattice.directions[site_n]: raise ValueError(f"The direction must be a possible direction from site_n = {site_n}, you entered direction = {direction}")

    indices = lattice.labels[site_n]

    if direction == 1:
        indices_mu = (indices[0] + 1, indices[1])
    elif direction == 2:
        indices_mu = (indices[0], indices[1] + 1)

    site_n_mu = lattice.reverse_labels[indices_mu]

    hamiltonian = Hamiltonian(lattice.n_qubits)
    hamiltonian.add_term('I'*lattice.n_qubits, 1.0)

    creation_hamiltonian = creation_operator_n(lattice, site_n)
    U_hamiltonian = U_n_direction(lattice, site_n, direction)
    U_hamiltonian.conjugate()
    annihilation_hamiltonian = annihilation_operator_n(lattice, site_n_mu)

    hamiltonian.multiply_by_hamiltonian(creation_hamiltonian) # M -> c
    hamiltonian.multiply_by_hamiltonian(U_hamiltonian) # M -> cU
    hamiltonian.multiply_by_hamiltonian(annihilation_hamiltonian) # M -> cUa
    
    return hamiltonian

def kinetic_term_n(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")

    hamiltonian = Hamiltonian(lattice.n_qubits)
    indices = lattice.labels[site_n]
    coeffs = {
        1: 1j,
        2: -1*((-1)**(indices[0] + indices[1]))
    }

    for direction in lattice.directions[site_n]:
        temp_hamiltonian = kinetic_subterm_n(lattice, site_n, direction)

        temp_hamiltonian_dagger = copy.copy(temp_hamiltonian)
        temp_hamiltonian_dagger.conjugate()

        if direction == 1: temp_hamiltonian_dagger.multiply_by_constant(-1)

        temp_hamiltonian.add_hamiltonian(temp_hamiltonian_dagger)
        temp_hamiltonian.multiply_by_constant(coeffs[direction])

        hamiltonian.add_hamiltonian(temp_hamiltonian)

    return hamiltonian

def particle_n_hamiltonian(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")

    x, y = lattice.labels[site_n]
    hamiltonian = Hamiltonian(lattice.n_qubits)
    I_term = 'I' * lattice.n_qubits
    sign = (-1)**(x + y)
    hamiltonian.add_term(I_term, 0.5)

    Z_term = list(I_term)
    Z_term[lattice.n_dynamical_gauge_qubits + site_n] = 'Z'
    hamiltonian.add_term(''.join(Z_term), -0.5*sign)
    
    return hamiltonian

def particle_number_hamiltonian(lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")

    total_pn = Hamiltonian(lattice.n_qubits)
    for site_n in range(lattice.n_fermion_qubits):
        pn_n = particle_n_hamiltonian(lattice, site_n)
        total_pn.add_hamiltonian(pn_n)

    return total_pn

def charge_n_hamiltonian(lattice : Lattice, site_n : int):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")
    if not isinstance(site_n, int): raise TypeError(f"site_n must be an integer, you have entered a {type(site_n)}")

    if not 0 <= site_n < lattice.n_fermion_qubits: raise ValueError(f"site_n must be positive and on the lattice, you entered with site_n = {site_n}, lattice.n_fermion_qubits = {lattice.n_fermion_qubits}")

    x, y = lattice.labels[site_n]
    charge_hamiltonian = Hamiltonian(lattice.n_qubits)

    I_term = 'I' * lattice.n_qubits
    Z_term = list(I_term)
    Z_term[lattice.n_dynamical_gauge_qubits + site_n] = 'Z'

    parity = (-1)**(x + y)

    charge_hamiltonian.add_term(''.join(Z_term), -0.5)
    charge_hamiltonian.add_term(I_term, parity * 0.5)

    return charge_hamiltonian

def charge_total_hamiltonian(lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")

    total_charge = Hamiltonian(lattice.n_qubits)
    for site_n in range(lattice.n_fermion_qubits):
        ch_n = charge_n_hamiltonian(lattice, site_n)
        total_charge.add_hamiltonian(ch_n)

    return total_charge

def electric_solve_gauss(lattice : Lattice):
    if not isinstance(lattice, Lattice): raise TypeError(f"The lattice must be a lattice.Lattice class, you have entered a {type(lattice)}")

    gauss_equations = lattice.gauss_equations

    link_variables = gauss_equations['link_variables']
    charge_variables = gauss_equations['charge_variables']
    reverse_link_variable_dict = gauss_equations['reverse_link_variable_dict']
        
    dependant_variables = gauss_equations['dependant_variables']
    independant_variables = gauss_equations['independant_variables']
    
    solution = gauss_equations['solution']
    
    hamiltonian_variables = {}
    for variable in link_variables:
        link = reverse_link_variable_dict[variable]
        site_n = lattice.reverse_labels[link[0]]
        hamiltonian_variables[variable] = electric_n_direction(lattice, site_n, link[1])
    
    for site_n in range(lattice.n_fermion_qubits):
        variable = charge_variables[site_n]
        hamiltonian_variables[variable] = charge_n_hamiltonian(lattice, site_n)
        
    total_hamiltonian = Hamiltonian(lattice.n_qubits)
    for dependant_variable in dependant_variables:
        direction = reverse_link_variable_dict[dependant_variable][1]
        if dependant_variable in solution and dependant_variable not in charge_variables:
            equation = solution[dependant_variable]
            temp_hamiltonian = Hamiltonian(lattice.n_qubits)

            for variable in independant_variables:
                coefficient = complex(equation.coeff(variable))
                temp_hamiltonian_inside = copy.copy(hamiltonian_variables[variable])
                temp_hamiltonian_inside.multiply_by_constant(coefficient)
                
                temp_hamiltonian.add_hamiltonian(temp_hamiltonian_inside)
            
            constant = complex(equation.subs({v: 0 for v in independant_variables}))
            constant_hamiltonian = Hamiltonian(lattice.n_qubits)
            constant_hamiltonian.add_term("I"*lattice.n_qubits, constant + lattice.background_field[direction - 1])

            temp_hamiltonian.add_hamiltonian(constant_hamiltonian)
            temp_hamiltonian.multiply_by_hamiltonian(temp_hamiltonian)
            temp_hamiltonian.cleanup()

            total_hamiltonian.add_hamiltonian(temp_hamiltonian)
    
    for variable in independant_variables:
        if variable not in charge_variables:
            direction = reverse_link_variable_dict[variable][1]
            temp_hamiltonian = copy.copy(hamiltonian_variables[variable])
            temp_hamiltonian.add_term("I"*lattice.n_qubits, lattice.background_field[direction - 1])
            temp_hamiltonian.multiply_by_hamiltonian(temp_hamiltonian)

            total_hamiltonian.add_hamiltonian(temp_hamiltonian)

    total_hamiltonian.cleanup()

    return total_hamiltonian