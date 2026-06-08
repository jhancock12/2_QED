# Standard libraries

# Local modules
from observables import qed_hamiltonian
from circuits.circuit_class import CircuitForLattice
from circuits.measurer_class import CircuitMeasurer
from lattice_class import Lattice
from solvers.mixed import solve_noiseless_sample_many
from solvers.sparse import solve_and_observe_sparse
from global_helpers import smart_round, dict_print

# Third-party libraries
import qiskit_aer
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

def full_runner(lattice_parameters, qed_parameters, vqe_parameters, SPSA_parameters):
    # -------------------------------
    lattice = Lattice(lattice_parameters['L_x'], lattice_parameters['L_y'], 
                    lattice_parameters['n_g'], lattice_parameters['dynamical_links_list'], 
                    lattice_parameters['charge_site'], lattice_parameters['anticharge_site'], 
                    lattice_parameters['background_field'])
    hamiltonian = qed_hamiltonian(qed_parameters, lattice)
    circuit_class = CircuitForLattice(lattice, vqe_parameters['n_fermion_layers'])
    # -------------------------------
    fake_backend = FakeCairoV2()
    noisy_simulator = qiskit_aer.AerSimulator.from_backend(fake_backend)
    # -------------------------------
    accepted_strings = lattice.SV_strings()

    measurer_noiseless = CircuitMeasurer(circuit_class, hamiltonian, qiskit_aer.AerSimulator(), vqe_parameters['shots'])

    measurer_noisy = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])

    measurer_noisy_SV_only = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])
    measurer_noisy_SV_only.add_SV(accepted_strings)

    measurer_noisy_mem_only = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])
    measurer_noisy_mem_only.build_MEM()
    measurer_noisy_all = CircuitMeasurer(circuit_class, hamiltonian, noisy_simulator, vqe_parameters['shots'])
    measurer_noisy_all.add_SV(accepted_strings)
    measurer_noisy_all.build_MEM()
    # -------------------------------
    measurer_classes = [measurer_noiseless, measurer_noisy, measurer_noisy_mem_only, measurer_noisy_SV_only, measurer_noisy_all]
    names = ['No quantum noise', 'No QEM', 'MEM only', 'SV only', 'Both']
    # -------------------------------
    results_sparse = solve_and_observe_sparse(hamiltonian, lattice)
    results = solve_noiseless_sample_many(hamiltonian, lattice, vqe_parameters, SPSA_parameters, measurer_classes, names)
    results['sparse'] = results_sparse
    for key in results:
        print(f"----- {key} -----")
        dict_print(results[key])