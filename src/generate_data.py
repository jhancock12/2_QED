#!/usr/bin/env python3
"""
Efficient data generation for (2+1)-QED with BEF.
All solves are performed once per unique parameter set; results are reused.
Chiral data is only computed for n_g=3.
Run on HPC with many cores. Set RUN_TEST = True for a quick check.
"""

import os
import sys
import time
import pickle
import multiprocessing as mp
from functools import partial
from itertools import product
import numpy as np

# ---------- Import your existing modules ----------
sys.path.append('.')
from lattice_class import Lattice
from observables import qed_hamiltonian
from solvers.sparse import solve_and_observe_sparse
from plotter import nice_scatter_plotter

# ---------- Configuration ----------
RUN_TEST = False          # Set True for a quick test (fewer points)
SAVE_DATA = False        # Save computed results to disk (pickle)
DATA_FILE = "qed_data.pkl"

# Lattice definitions (adjust to your exact parameters)
LATTICE_3x2 = {
    'L_x': 3, 'L_y': 2,
    'dynamical_links_list': [((0,0),1), ((1,0),2)],
    'charge_site': (0,0),
    'anticharge_site': (2,1),
}
LATTICE_4x2 = {
    'L_x': 4, 'L_y': 2,
    'dynamical_links_list': [((0,0),1), ((1,0),2), ((1,1),1)],
    'charge_site': (0,0),
    'anticharge_site': (3,1),
}

# Physics parameters
A = 1.0
MASS_FIXED = 3.0
CHARGE_WEIGHT = 20.0

# ---------- Sweep ranges ----------
if RUN_TEST:
    # Test ranges that include all needed values (or close)
    g_range = np.array([0.3, 1.0, 2.0, 3.0])
    bef_small_fine = np.linspace(0.0, 1.0, 5)        # quick test: 5 points
    bef_small_plot = bef_small_fine                  # same in test mode
    bef_large = np.linspace(0.0, 4.0, 5)
    bef_upto15 = np.linspace(0.0, 15.0, 4)
    lambda_range = np.linspace(-1.0, 1.0, 5)
    x_range = np.array([0.5, 1.0, 2.5, 4.0])
    e_range = np.linspace(0.0, 2.0, 5)
    n_g_list = [2, 3]
    chiral_n_g_list = [3]
    PN_THRESHOLD = 2.0
else:
    g_range = np.linspace(0.3, 3.0, 25)              # 25 points for better resolution
    bef_small_fine = np.linspace(0.0, 1.0, 15)         # 9 points for sigma/g*
    bef_small_plot = np.array([0.0, 0.25, 0.5, 0.75, 1.0])  # 5 points for other plots
    bef_large = np.linspace(0.0, 4.0, 5)
    bef_upto15 = np.linspace(0.0, 15.0, 16)
    lambda_range = np.linspace(-2.0, 2.0, 21)
    x_range = np.linspace(0.5, 4.0, 8)
    e_range = np.linspace(0.0, 2.0, 11)
    n_g_list = [2, 3, 4]
    chiral_n_g_list = [3]
    PN_THRESHOLD = 2.0

# ---------- Helper function to select g values ----------
def select_g_values(targets, g_range):
    """
    For each target, return the smallest g in g_range that is >= target.
    If none, return the largest g.
    """
    selected = []
    for t in targets:
        candidates = g_range[g_range >= t]
        if len(candidates) > 0:
            selected.append(np.min(candidates))
        else:
            selected.append(np.max(g_range))
    return selected

# ---------- Helper functions ----------
def create_lattice(lattice_type, n_g, background_field, has_charges=True):
    """Return a Lattice instance."""
    params = LATTICE_3x2 if lattice_type == '3x2' else LATTICE_4x2
    if not has_charges:
        params = params.copy()
        params['charge_site'] = ()
        params['anticharge_site'] = ()
    return Lattice(
        L_x=params['L_x'], L_y=params['L_y'],
        n_g=n_g,
        dynamical_links_list=params['dynamical_links_list'],
        charge_site=params['charge_site'],
        anticharge_site=params['anticharge_site'],
        background_field=background_field,
    )

def solve_one(params, electric_multi=1.0):
    """
    Solve for a single set of parameters.
    Returns a dict with energy, particle_number, chiral_condensate,
    electric_field_dict, max_E.
    """
    n_g, lattice_type, g, bef, mass, has_charges = params
    bef_vec = [bef, 0.0]   # BEF in x-direction
    lattice = create_lattice(lattice_type, n_g, bef_vec, has_charges)
    qed_params = {'m': mass, 'g': g, 'a': A, 'charge_weight': CHARGE_WEIGHT}
    H = qed_hamiltonian(qed_params, lattice, electric_multi=electric_multi)
    res = solve_and_observe_sparse(H, lattice)
    # Add max electric field
    ef = res.get('electric_field_dict', {})
    max_abs = max(abs(v) for v in ef.values()) if ef else 0.0
    res['max_E'] = max_abs
    # Keep only needed fields to save memory
    return {
        'energy': res['energy'],
        'particle_number': res['particle_number_total'],
        'chiral_condensate': res['chiral_condensate'],
        'max_E': max_abs,
    }

# ---------- Master data collection ----------
def build_fixed_mass_data():
    """Compute all states for fixed mass m=3.0: n_g, g, BEF, with/without charges."""
    print("Building fixed-mass data set...")
    # Combine all BEF values needed for fixed-mass plots
    all_bef = sorted(set(bef_upto15) | set(bef_large) | set(bef_small_fine) | set(bef_small_plot))
    # Generate all parameter combinations (n_g, g, bef, has_charges)
    combos = []
    for n_g in n_g_list:
        for g in g_range:
            for bef in all_bef:
                # With charges
                combos.append((n_g, '3x2', g, bef, MASS_FIXED, True))
                # Without charges (vacuum)
                combos.append((n_g, '3x2', g, bef, MASS_FIXED, False))
    print(f"Total fixed-mass solves: {len(combos)}")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(solve_one, combos)
    # Store in nested dict: data[(n_g, g, bef, has_charges)] = results
    data = {}
    for idx, combo in enumerate(combos):
        n_g, _, g, bef, _, has_charges = combo
        key = (n_g, g, bef, has_charges)
        data[key] = results[idx]
    print(f"Fixed-mass data size: {len(data)}")
    if data:
        sample_key = next(iter(data))
        print(f"Sample key: {sample_key}, energy: {data[sample_key]['energy']}")
    else:
        print("WARNING: Fixed-mass data is empty!")
    return data

def build_chiral_data():
    """
    Compute chiral condensate for n_g=3 only.
    - C(lambda) for various x (Fig.24)
    - Shift crossover (lambda* vs e) at x=2.5 (Fig.25)
    - Chiral with BEF for 3x2 and 4x2 (Figs.26-29)
    """
    print("Building chiral data set (n_g=3 only)...")
    data = {}
    g_fixed = np.sqrt(2.5 / A)   # x=2.5

    # 1. Chiral condensate C(lambda) for various x (Fig.24)
    for x in x_range:
        g = np.sqrt(x / A)
        for lam in lambda_range:
            mass = lam * g * g
            key = (3, g, 0.0, mass, 'chiral')
            if key not in data:
                res = solve_one((3, '3x2', g, 0.0, mass, False))
                data[key] = res

    # 2. Shift crossover (Fig.25): lambda* vs e at x=2.5
    for e_val in e_range:
        lam_scan = np.linspace(-0.5, 0.5, 11)
        for lam in lam_scan:
            mass = lam * g_fixed * g_fixed
            key = (3, g_fixed, 0.0, mass, 'chiral_shift', e_val)
            if key not in data:
                res = solve_one((3, '3x2', g_fixed, 0.0, mass, False),
                                electric_multi=e_val)
                data[key] = res

    # 3. Chiral with BEF for 3x2 lattice (Figs.26-27)
    bef_for_chiral = [0.0, 0.125, 0.25, 0.375, 0.5]   # fixed 5 values
    for bef in bef_for_chiral:
        for lam in lambda_range:
            mass = lam * g_fixed * g_fixed
            key = (3, g_fixed, bef, mass, 'chiral_bef')
            if key not in data:
                res = solve_one((3, '3x2', g_fixed, bef, mass, False))
                data[key] = res

    # 4. Chiral with BEF for 4x2 lattice (Figs.28-29) – uncomment if needed
    # for bef in bef_for_chiral:
    #     for lam in lambda_range:
    #         mass = lam * g_fixed * g_fixed
    #         key = (3, g_fixed, bef, mass, 'chiral_bef_4x2')
    #         if key not in data:
    #             res = solve_one((3, '4x2', g_fixed, bef, mass, False))
    #             data[key] = res

    print(f"Chiral data size: {len(data)}")
    return data

def plot_from_data(data_fixed, data_chiral):
    """
    Generate all required plots from the precomputed data.
    Each plot is saved in two versions: square and non‑square.
    The square version gets '_square' appended to the filename.
    """
    # Helper functions
    def get_energy(n_g, g, bef, has_charges):
        key = (n_g, g, bef, has_charges)
        return data_fixed[key]['energy'] if key in data_fixed else np.nan

    def get_particle_number(n_g, g, bef):
        key = (n_g, g, bef, True)   # charged system
        return data_fixed[key]['particle_number'] if key in data_fixed else np.nan

    def get_chiral(n_g, g, bef, mass):
        key = (n_g, g, bef, mass, 'chiral')
        return data_chiral[key]['chiral_condensate'] if key in data_chiral else np.nan

    def get_chiral_shift(n_g, g, bef, mass, e):
        key = (n_g, g, bef, mass, 'chiral_shift', e)
        return data_chiral[key]['chiral_condensate'] if key in data_chiral else np.nan

    # Helper to plot both square and non‑square versions
    def plot_both(**kwargs):
        # kwargs should contain all arguments to nice_scatter_plotter except square and label_save_title
        # We will extract those and call twice.
        square = kwargs.pop('square', False)
        title_base = kwargs.pop('label_save_title')
        # First: non‑square
        nice_scatter_plotter(
            square=False,
            label_save_title=title_base,
            **kwargs
        )
        # Second: square
        nice_scatter_plotter(
            square=True,
            label_save_title=title_base + '_square',
            **kwargs
        )

    # ---------- 1. Potential V(g) for various n_g (Fig.15) ----------
    ng_list = [n for n in n_g_list if n in [2,3,4,5]]
    V_data = []
    for n in ng_list:
        V = []
        for g in g_range:
            E_charge = get_energy(n, g, 0.0, True)
            V.append(E_charge if not np.isnan(E_charge) else np.nan)
        V_data.append(V)
    if any(not np.isnan(v).all() for v in V_data):
        plot_both(
            data_x_line=[g_range.tolist()]*len(ng_list),
            data_y_line=V_data,
            label_x=r"$g$", label_y=r"$V(g)$",
            label_save_title="V_mixed_ng",
            labels_line=[f"$n_g={n}$" for n in ng_list],
            show=False
        )
    else:
        print("Warning: No data for V_mixed_ng plot")

    # ---------- 2. Potential with n_g=3, large BEF (Fig.16) ----------
    bef_vals = [b for b in bef_large]
    V_large = []
    for bef in bef_vals:
        V = []
        for g in g_range:
            E_charge = get_energy(3, g, bef, True)
            V.append(E_charge if not np.isnan(E_charge) else np.nan)
        V_large.append(V)
    if any(not np.isnan(v).all() for v in V_large):
        plot_both(
            data_x_line=[g_range.tolist()]*len(bef_vals),
            data_y_line=V_large,
            label_x=r"$g$", label_y=r"$V(g)$",
            label_save_title="V_BEF_bigger",
            labels_line=[f"$\\mathcal{{E}}={b:.1f}$" for b in bef_vals],
            show=False
        )
    else:
        print("Warning: No data for V_BEF_bigger plot")

    # ---------- 3. Potential n_g=2, small BEF (Fig.17) – use coarse BEF set ----------
    bef_small_2 = [b for b in bef_small_plot if b < 0.5]
    V_small_2 = []
    for bef in bef_small_2:
        V = []
        for g in g_range:
            E_charge = get_energy(2, g, bef, True)
            V.append(E_charge if not np.isnan(E_charge) else np.nan)
        V_small_2.append(V)
    if any(not np.isnan(v).all() for v in V_small_2):
        plot_both(
            data_x_line=[g_range.tolist()]*len(bef_small_2),
            data_y_line=V_small_2,
            label_x=r"$g$", label_y=r"$V(g)$",
            label_save_title="V_BEF_2",
            labels_line=[f"$\\mathcal{{E}}={b:.2f}$" for b in bef_small_2],
            show=False
        )
    else:
        print("Warning: No data for V_BEF_2 plot")

    # ---------- 4. Potential n_g=3, small BEF (Fig.18) – use coarse BEF set ----------
    bef_small_3 = [b for b in bef_small_plot if b < 1.0]
    V_small_3 = []
    for bef in bef_small_3:
        V = []
        for g in g_range:
            E_charge = get_energy(3, g, bef, True)
            V.append(E_charge if not np.isnan(E_charge) else np.nan)
        V_small_3.append(V)
    if any(not np.isnan(v).all() for v in V_small_3):
        plot_both(
            data_x_line=[g_range.tolist()]*len(bef_small_3),
            data_y_line=V_small_3,
            label_x=r"$g$", label_y=r"$V(g)$",
            label_save_title="V_BEF_2_bigger",
            labels_line=[f"$\\mathcal{{E}}={b:.2f}$" for b in bef_small_3],
            show=False
        )
    else:
        print("Warning: No data for V_BEF_2_bigger plot")

    # ---------- 5. rho vs BEF (Fig.19) ----------
    g_targets = [1.0, 2.0, 3.0]
    g_selected = select_g_values(g_targets, g_range)
    for idx, g_sel in enumerate(g_selected):
        rho_data = []
        ng_plot = [n for n in n_g_list if n > 1]
        for n in ng_plot:
            rho = []
            for bef in bef_upto15:
                key = (n, g_sel, bef, False)
                if key in data_fixed:
                    maxE = data_fixed[key]['max_E']
                    rho.append(maxE / (n - 1) if n > 1 else 0.0)
                else:
                    rho.append(np.nan)
            rho_data.append(rho)
        if any(not np.isnan(r).all() for r in rho_data):
            plot_both(
                data_x_line=[bef_upto15.tolist()]*len(ng_plot),
                data_y_line=rho_data,
                label_x=r"$\mathcal{E}$", label_y=r"$\rho$",
                label_save_title=f"s_3_{idx}",
                labels_line=[f"$n_g={n}$" for n in ng_plot],
                show=False
            )
        else:
            print(f"Warning: No data for rho plot (g={g_sel})")

    # ---------- 6. ΔH vs BEF (Fig.20) ----------
    for idx, g_sel in enumerate(g_selected):
        dH_data = []
        ng_plot = [n for n in n_g_list if n > n_g_list[0]]
        for n in ng_plot:
            dH = []
            for bef in bef_upto15:
                E_n = get_energy(n, g_sel, bef, False) if (n, g_sel, bef, False) in data_fixed else np.nan
                E_nm1 = get_energy(n-1, g_sel, bef, False) if (n-1, g_sel, bef, False) in data_fixed else np.nan
                dH.append(E_n - E_nm1 if not np.isnan(E_n) and not np.isnan(E_nm1) else np.nan)
            dH_data.append(dH)
        if any(not np.isnan(d).all() for d in dH_data):
            plot_both(
                data_x_line=[bef_upto15.tolist()]*len(ng_plot),
                data_y_line=dH_data,
                label_x=r"$\mathcal{E}$", label_y=r"$\Delta H_{n_g}$",
                label_save_title=f"e_3_{idx}",
                labels_line=[f"$n_g={n}$" for n in ng_plot],
                show=False
            )
        else:
            print(f"Warning: No data for ΔH plot (g={g_sel})")

    # ---------- 7. sigma and g* (Fig.21,22) – use FINE BEF set ----------
    THRESHOLD = PN_THRESHOLD

    def find_g_star_pn(n_g, bef):
        for g in g_range:
            pn = get_particle_number(n_g, g, bef)
            if not np.isnan(pn) and pn > THRESHOLD:
                return g
        return None

    def sigma_from_V(n_g, bef):
        g_star = find_g_star_pn(n_g, bef)
        if g_star is None:
            return None
        start_g = 1.0
        g_vals = []
        V_vals = []
        for g in g_range:
            if g < start_g:
                continue
            E_charge = get_energy(n_g, g, bef, True)
            if np.isnan(E_charge):
                continue
            g_vals.append(g)
            V_vals.append(E_charge)
            if g == g_star:
                break
        if len(g_vals) < 2:
            return None
        slope, _ = np.polyfit(g_vals, V_vals, 1)
        return slope

    print("\n===== DEBUG: Charged particle numbers for sigma/g* =====")
    for n in n_g_list:
        for bef in bef_small_fine:
            for g in g_range:
                pn = get_particle_number(n, g, bef)
                print(f"n_g={n}, bef={bef:.3f}, g={g:.3f}, pn={pn:.6f}")

    sigmas_dict = {}
    gstars_dict = {}
    for n in n_g_list:
        sigmas = []
        gstars = []
        for bef in bef_small_fine:
            gstar = find_g_star_pn(n, bef)
            gstars.append(gstar if gstar is not None else np.nan)
            sigma_val = sigma_from_V(n, bef)
            sigmas.append(sigma_val if sigma_val is not None else np.nan)
        sigmas_dict[n] = sigmas
        gstars_dict[n] = gstars

    print("\n===== Computed g* and sigma =====")
    for n in n_g_list:
        print(f"n_g={n}: gstars={gstars_dict[n]}, sigmas={sigmas_dict[n]}")

    if any(not np.isnan(s).all() for s in sigmas_dict.values()):
        plot_both(
            data_x_line=[bef_small_fine.tolist()]*len(n_g_list),
            data_y_line=[sigmas_dict[n] for n in n_g_list],
            label_x=r"$\mathcal{E}$", label_y=r"$\sigma(\mathcal{E})$",
            label_save_title="sigmas_ng",
            labels_line=[f"$n_g={n}$" for n in n_g_list],
            show=False
        )
    else:
        print("Warning: No data for sigma plot")

    if any(not np.isnan(g).all() for g in gstars_dict.values()):
        plot_both(
            data_x_line=[bef_small_fine.tolist()]*len(n_g_list),
            data_y_line=[gstars_dict[n] for n in n_g_list],
            label_x=r"$\mathcal{E}$", label_y=r"$g^*(\mathcal{E})$",
            label_save_title="gs_ng",
            labels_line=[f"$n_g={n}$" for n in n_g_list],
            show=False
        )
    else:
        print("Warning: No data for g* plot")

    # ---------- 8. Delta energies (Fig.23) – use COARSE BEF set ----------
    for idx, g_sel in enumerate(g_selected):
        dH_charges = []
        dH_vac = []
        dH_pairs = []
        for bef in bef_small_plot:
            E_ch = get_energy(3, g_sel, bef, True)
            E_ch0 = get_energy(3, g_sel, 0.0, True)
            E_v = get_energy(3, g_sel, bef, False)
            E_v0 = get_energy(3, g_sel, 0.0, False)
            dH_charges.append(E_ch - E_ch0 if not np.isnan(E_ch) and not np.isnan(E_ch0) else np.nan)
            dH_vac.append(E_v - E_v0 if not np.isnan(E_v) and not np.isnan(E_v0) else np.nan)
            if not np.isnan(dH_charges[-1]) and not np.isnan(dH_vac[-1]):
                dH_pairs.append(dH_charges[-1] - dH_vac[-1])
            else:
                dH_pairs.append(np.nan)
        if any(not np.isnan(x) for x in dH_charges):
            plot_both(
                data_x_line=[bef_small_plot.tolist()]*3,
                data_y_line=[dH_charges, dH_vac, dH_pairs],
                label_x=r"$\mathcal{E}$", label_y=r"$\Delta H$",
                label_save_title=f"charges_vacuum_pairs_{idx+1}",
                labels_line=[r"$\Delta H_{\mathrm{charges}}$",
                             r"$\Delta H_{\mathrm{vacuum}}$",
                             r"$\Delta H_{\mathrm{pairs}}$"],
                show=False
            )
        else:
            print(f"Warning: No data for ΔH energies (g={g_sel})")

    # ---------- 9. Chiral condensate (Fig.24) ----------
    n_chiral = 3
    C_all = []
    chi_all = []
    x_vals_used = []
    for x in x_range:
        g = np.sqrt(x / A)
        C_vals = []
        for lam in lambda_range:
            mass = lam * g * g
            val = get_chiral(n_chiral, g, 0.0, mass)
            C_vals.append(val if not np.isnan(val) else np.nan)
        C_all.append(C_vals)
        if len(C_vals) > 1 and not all(np.isnan(c) for c in C_vals):
            dC = np.gradient(C_vals, lambda_range)
            chi_all.append(dC.tolist())
        else:
            chi_all.append([np.nan]*len(lambda_range))
        x_vals_used.append(x)
    if any(not np.isnan(c).all() for c in C_all):
        plot_both(
            data_x_line=[lambda_range.tolist()]*len(x_vals_used),
            data_y_line=C_all,
            label_x=r"$\lambda$", label_y=r"$\mathcal{C}$",
            label_save_title="chiral_initial",
            labels_line=[f"$x={x:.1f}$" for x in x_vals_used],
            show=False
        )
    else:
        print("Warning: No data for chiral_initial plot")

    if any(not np.isnan(c).all() for c in chi_all):
        plot_both(
            data_x_line=[lambda_range.tolist()]*len(x_vals_used),
            data_y_line=chi_all,
            label_x=r"$\lambda$", label_y=r"$\chi_{\mathcal{C}}$",
            label_save_title="chiral_initial_2",
            labels_line=[f"$x={x:.1f}$" for x in x_vals_used],
            show=False
        )
    else:
        print("Warning: No data for chiral_initial_2 plot")

    # ---------- 10. Shift crossover (Fig.25) ----------
    g_fixed = np.sqrt(2.5 / A)
    shift_data = []
    for n in n_g_list:
        lam_stars = []
        for e_val in e_range:
            lam_scan = np.linspace(-0.5, 0.5, 11)
            C_scan = []
            for lam in lam_scan:
                mass = lam * g_fixed * g_fixed
                val = get_chiral_shift(n, g_fixed, 0.0, mass, e_val) if n == 3 else np.nan
                C_scan.append(val if not np.isnan(val) else np.nan)
            if len(C_scan) > 1:
                for i in range(len(C_scan)-1):
                    if C_scan[i]*C_scan[i+1] < 0:
                        lam_star = lam_scan[i] - C_scan[i]*(lam_scan[i+1]-lam_scan[i])/(C_scan[i+1]-C_scan[i])
                        lam_stars.append(lam_star)
                        break
                else:
                    lam_stars.append(np.nan)
            else:
                lam_stars.append(np.nan)
        shift_data.append(lam_stars)
    if any(not np.isnan(s).all() for s in shift_data):
        plot_both(
            data_x_line=[e_range.tolist()]*len(n_g_list),
            data_y_line=shift_data,
            label_x=r"$e$", label_y=r"$\lambda^*$",
            label_save_title="shifting_shift",
            labels_line=[f"$n_g={n}$" for n in n_g_list],
            show=False
        )
    else:
        print("Warning: No data for shifting_shift plot")

    # ---------- 11. Chiral with BEF (Figs.26-29) ----------
    bef_for_chiral = [0.0, 0.125, 0.25, 0.375, 0.5]
    # 3x2 lattice
    C_all = []
    chi_all = []
    bef_vals = []
    for bef in bef_for_chiral:
        C_vals = []
        for lam in lambda_range:
            mass = lam * g_fixed * g_fixed
            key = (3, g_fixed, bef, mass, 'chiral_bef')
            val = data_chiral[key]['chiral_condensate'] if key in data_chiral else np.nan
            C_vals.append(val)
        if any(not np.isnan(c) for c in C_vals):
            C_all.append(C_vals)
            if len(C_vals) > 1:
                dC = np.gradient(C_vals, lambda_range)
                chi_all.append(dC.tolist())
            else:
                chi_all.append([np.nan]*len(lambda_range))
            bef_vals.append(bef)
    if C_all:
        plot_both(
            data_x_line=[lambda_range.tolist()]*len(bef_vals),
            data_y_line=C_all,
            label_x=r"$\lambda$", label_y=r"$\mathcal{C}$",
            label_save_title="chiral_small_bef",
            labels_line=[f"$\\mathcal{{E}}={b:.3f}$" for b in bef_vals],
            show=False
        )
        if chi_all:
            plot_both(
                data_x_line=[lambda_range.tolist()]*len(bef_vals),
                data_y_line=chi_all,
                label_x=r"$\lambda$", label_y=r"$\chi_{\mathcal{C}}$",
                label_save_title="chiral_chi_small_bef",
                labels_line=[f"$\\mathcal{{E}}={b:.3f}$" for b in bef_vals],
                show=False
            )

    # 4x2 lattice (if data exists)
    has_4x2 = any('chiral_bef_4x2' in str(key) for key in data_chiral.keys())
    if has_4x2:
        C_all_4x2 = []
        chi_all_4x2 = []
        bef_vals_4x2 = []
        for bef in bef_for_chiral:
            C_vals = []
            for lam in lambda_range:
                mass = lam * g_fixed * g_fixed
                key = (3, g_fixed, bef, mass, 'chiral_bef_4x2')
                val = data_chiral[key]['chiral_condensate'] if key in data_chiral else np.nan
                C_vals.append(val)
            if any(not np.isnan(c) for c in C_vals):
                C_all_4x2.append(C_vals)
                if len(C_vals) > 1:
                    dC = np.gradient(C_vals, lambda_range)
                    chi_all_4x2.append(dC.tolist())
                else:
                    chi_all_4x2.append([np.nan]*len(lambda_range))
                bef_vals_4x2.append(bef)
        if C_all_4x2:
            plot_both(
                data_x_line=[lambda_range.tolist()]*len(bef_vals_4x2),
                data_y_line=C_all_4x2,
                label_x=r"$\lambda$", label_y=r"$\mathcal{C}$",
                label_save_title="chiral_medium_bef",
                labels_line=[f"$\\mathcal{{E}}={b:.3f}$" for b in bef_vals_4x2],
                show=False
            )
            if chi_all_4x2:
                plot_both(
                    data_x_line=[lambda_range.tolist()]*len(bef_vals_4x2),
                    data_y_line=chi_all_4x2,
                    label_x=r"$\lambda$", label_y=r"$\chi_{\mathcal{C}}$",
                    label_save_title="chiral_chi_medium_bef",
                    labels_line=[f"$\\mathcal{{E}}={b:.3f}$" for b in bef_vals_4x2],
                    show=False
                )

    print("All plots saved.")

# ---------- Main ----------
def main():
    # Build or load fixed-mass data
    if SAVE_DATA and os.path.exists(DATA_FILE):
        print(f"Loading cached data from {DATA_FILE}...")
        with open(DATA_FILE, 'rb') as f:
            data_fixed, data_chiral = pickle.load(f)
    else:
        data_fixed = build_fixed_mass_data()
        data_chiral = build_chiral_data()
        if SAVE_DATA:
            print(f"Saving data to {DATA_FILE}...")
            with open(DATA_FILE, 'wb') as f:
                pickle.dump((data_fixed, data_chiral), f)

    # Generate all plots
    plot_from_data(data_fixed, data_chiral)

if __name__ == "__main__":
    main()