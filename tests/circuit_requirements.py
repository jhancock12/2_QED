# Standard libraries

# Local modules
from lattice_class import Lattice

# Third-party libraries

lattice_parameters = {
    'L_x' : 3,
    'L_y' : 2,
    'n_g' : 2,
    'dynamical_links_list' : [((0, 0), 1), ((1, 0), 2)],# ((2, 0), 2)],#  ((0, 1), 1), ((1, 1), 2), ((2, 1), 2)],
    'charge_site' : (),
    'anticharge_site' : (),
    'background_field' : [0.0, 0.0]
}

lattice = Lattice(
    L_x = lattice_parameters['L_x'],
    L_y = lattice_parameters['L_y'],
    n_g = lattice_parameters['n_g'],
    dynamical_links_list = lattice_parameters['dynamical_links_list'],
    charge_site = lattice_parameters['charge_site'],
    anticharge_site = lattice_parameters['anticharge_site'],
    background_field = lattice_parameters['background_field']
)

sizes = [(3, 2), (4, 2), (4, 3)]
links = [[((0, 0), 1), ((1, 0), 2)], [((0, 0), 1), ((1, 0), 2) , ((2, 0), 2)], [((0, 0), 1), ((1, 0), 2), ((2, 0), 2), ((0, 1), 1), ((1, 1), 2), ((2, 1), 2)]]

n_gs = [2, 3, 4, 5]

def iSwap_block_calculate_qed_func(lattice):
    possible_pairs = []

    for i in range(lattice.n_fermion_qubits):
        for j in range(i):
            if i != j:
                indices_i = lattice.labels[i]
                indices_j = lattice.labels[j]

                dx = abs(indices_i[0] - indices_j[0])
                dy = abs(indices_i[1] - indices_j[1])

                if dx + dy == 1:
                    possible_pairs.append([j,i])

    return len(possible_pairs)

def gauge_calculate(lattice):
    total = 0
    for i in range(lattice.n_g - 1):
        total += 1

    # last qubit is only excited via a control -> can't be 1 while all others are 0
    p = lattice.n_g - 1
    for i in range(lattice.n_g - 2, -1, -1):     # descending controls, matches your n=3 case
        total += 1

    return total

for size_k in range(len(sizes)):
    L_x = sizes[size_k][0]
    L_y = sizes[size_k][1]
    dynamical_links_list = links[size_k]
    for n_g in n_gs:
        # print("Size:", (L_x, L_y))
        # print("n_g:", n_g)
        lattice = Lattice(
            L_x = L_x,
            L_y = L_y,
            n_g = n_g,
            dynamical_links_list = dynamical_links_list,
            charge_site = (),
            anticharge_site = (),
            background_field = [0.0, 0.0]
        )
        # print("n_qubits:", lattice.n_qubits)
        iSwap_number = iSwap_block_calculate_qed_func(lattice)
        gauge_number = gauge_calculate(lattice) * lattice.n_dynamical_links
        # print("Fermion parameters required:", iSwap_number)

        print(f"${L_x}", r"\times", f"{L_y}$ & {n_g} & {lattice.n_qubits} & {gauge_number} & {iSwap_number} & {gauge_number + iSwap_number}", r"\\")