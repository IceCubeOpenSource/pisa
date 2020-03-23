# pylint: disable = not-callable, invalid-name, too-many-nested-blocks


"""
Neutrino flavour oscillation in matter calculation
Based on the original prob3++ implementation of Roger Wendell
http://www.phy.duke.edu/~raw22/public/Prob3++/ (2012)

See `numba_osc_tests.py` for unit tests of functions in this module.
"""


from __future__ import absolute_import, print_function, division

__all__ = [
    "osc_probs_vacuum_kernel",
    "osc_probs_layers_kernel",
    "get_transition_matrix",
]

__version__ = "0.2"

import cmath
import math

from pisa.utils.numba_tools import (
    myjit,
    conjugate_transpose,
    conjugate,
    matrix_dot_matrix,
    matrix_dot_vector,
    clear_matrix,
    copy_matrix,
    cuda,
    ctype,
    ftype,
)


@myjit
def osc_probs_vacuum_kernel(dm, mix, nubar, energy, distance_in_layer, osc_probs):
    """ Calculate vacumm mixing probabilities

    Parameters
    ----------
    dm : real 2d array
        Mass splitting matrix, eV^2

    mix : complex 2d array
        PMNS mixing matrix

    nubar : int
        +1 for neutrinos, -1 for antineutrinos

    energy : float
        Neutrino energy, GeV

    distance_in_layer : real 1d-array
        Baselines (will be summed up), km

    osc_probs : real 2d array (empty)
        Returned oscillation probabilities in the form:
        osc_prob[i,j] = probability of flavor i to oscillate into flavor j
        with 0 = electron, 1 = muon, 3 = tau


    Notes
    -----
    This is largely unvalidated so far

    """

    # no need to conjugate mix matrix, as we anyway only need real part
    # can this be right?

    clear_matrix(osc_probs)
    osc_probs_local = cuda.local.array(shape=(3, 3), dtype=ftype)

    # sum up length from all layers
    baseline = 0.0
    for i in range(distance_in_layer.shape[0]):
        baseline += distance_in_layer[i]

    # make more precise 20081003 rvw
    l_over_e = 1.26693281 * baseline / energy
    s21 = math.sin(dm[1, 0] * l_over_e)
    s32 = math.sin(dm[2, 0] * l_over_e)
    s31 = math.sin((dm[2, 1] + dm[3, 2]) * l_over_e)

    # does anybody understand this loop?
    # ista = abs(*nutype) - 1
    for ista in range(3):
        for iend in range(2):
            osc_probs_local[ista, iend] = (
                (mix[ista, 0].real * mix[ista, 1].real * s21) ** 2
                + (mix[ista, 1].real * mix[ista, 2].real * s32) ** 2
                + (mix[ista, 2].real * mix[ista, 0].real * s31) ** 2
            )
            if iend == ista:
                osc_probs_local[ista, iend] = 1.0 - 4.0 * osc_probs_local[ista, iend]
            else:
                osc_probs_local[ista, iend] = -4.0 * osc_probs_local[ista, iend]

        osc_probs_local[ista, 2] = (
            1.0 - osc_probs_local[ista, 0] - osc_probs_local[ista, 1]
        )

    # is this necessary?
    if nubar > 0:
        copy_matrix(osc_probs_local, osc_probs)
    else:
        for i in range(3):
            for j in range(3):
                osc_probs[i, j] = osc_probs_local[j, i]


@myjit
def osc_probs_layers_kernel(
    dm, mix, nsi_eps, nubar, energy, density_in_layer, distance_in_layer, osc_probs
):
    """ Calculate oscillation probabilities

    given layers of length and density

    Parameters
    ----------
    dm : real 2d array
        Mass splitting matrix, eV^2

    mix : complex 2d array
        PMNS mixing matrix

    nsi_eps : 2d array
        Non-standard interactions (set to 3x3 zeros for only standard oscillations)

    nubar : int
        1 for neutrinos, -1 for antineutrinos

    energy : float
        Neutrino energy, GeV

    density_in_layer : 1d-array
        Density per layer, g/cm^2

    distance_in_layer : 1d-array
        Distance per layer traversed, km

    osc_probs : 2d array (empty)
        Returned oscillation probabilities in the form:
        osc_prob[i,j] = probability of flavor i to oscillate into flavor j
        with 0 = electron, 1 = muon, 3 = tau


    Notes
    -----
    !!! Right now, because of CUDA, the maximum number of layers
    is hard coded and set to 120 (59Layer PREM + Atmosphere).
    This is used for cached layer computation, where earth layer, which
    are typically traversed twice (it's symmetric) are not recalculated
    but rather cached..

    """

    # 3x3 complex
    H_vac = cuda.local.array(shape=(3, 3), dtype=ctype)
    mix_nubar = cuda.local.array(shape=(3, 3), dtype=ctype)
    mix_nubar_conj_transp = cuda.local.array(shape=(3, 3), dtype=ctype)
    transition_product = cuda.local.array(shape=(3, 3), dtype=ctype)
    transition_matrix = cuda.local.array(shape=(3, 3), dtype=ctype)
    tmp = cuda.local.array(shape=(3, 3), dtype=ctype)

    clear_matrix(H_vac)
    clear_matrix(osc_probs)

    # 3-vector complex
    raw_input_psi = cuda.local.array(shape=(3), dtype=ctype)
    output_psi = cuda.local.array(shape=(3), dtype=ctype)

    use_mass_eigenstates = False

    cache = True
    # cache = False

    # TODO:
    # * ensure convention below is respected in MC reweighting
    #   (nubar > 0 for nu, < 0 for anti-nu)
    # * nubar is passed in, so could already pass in the correct form
    #   of mixing matrix, i.e., possibly conjugated
    if nubar > 0:
        # in this case the mixing matrix is left untouched
        copy_matrix(mix, mix_nubar)

    else:
        # here we need to complex conjugate all entries
        # (note that this only changes calculations with non-clear_matrix deltacp)
        conjugate(mix, mix_nubar)

    conjugate_transpose(mix_nubar, mix_nubar_conj_transp)

    get_H_vac(mix_nubar, mix_nubar_conj_transp, dm, H_vac)

    if cache:
        # allocate array to store all the transition matrices
        # doesn't work in cuda...needs fixed shape
        transition_matrices = cuda.local.array(shape=(120, 3, 3), dtype=ctype)

        # loop over layers
        for i in range(distance_in_layer.shape[0]):
            density = density_in_layer[i]
            distance = distance_in_layer[i]
            if distance > 0.0:
                layer_matrix_index = -1
                # chaeck if exists
                for j in range(i):
                    # if density_in_layer[j] == density and distance_in_layer[j] == distance:
                    if (abs(density_in_layer[j] - density) < 1e-5) and (
                        abs(distance_in_layer[j] - distance) < 1e-5
                    ):
                        layer_matrix_index = j

                # use from cached
                if layer_matrix_index >= 0:
                    for j in range(3):
                        for k in range(3):
                            transition_matrices[i, j, k] = transition_matrices[
                                layer_matrix_index, j, k
                            ]

                # only calculate if necessary
                else:
                    get_transition_matrix(
                        nubar,
                        energy,
                        density,
                        distance,
                        mix_nubar,
                        mix_nubar_conj_transp,
                        nsi_eps,
                        H_vac,
                        dm,
                        transition_matrix,
                    )
                    # copy
                    for j in range(3):
                        for k in range(3):
                            transition_matrices[i, j, k] = transition_matrix[j, k]
            else:
                # identity matrix
                for j in range(3):
                    for k in range(3):
                        if j == k:
                            transition_matrix[j, k] = 0.0
                        else:
                            transition_matrix[j, k] = 1.0

        # now multiply them all
        first_layer = True
        for i in range(distance_in_layer.shape[0]):
            distance = distance_in_layer[i]
            if distance > 0.0:
                for j in range(3):
                    for k in range(3):
                        transition_matrix[j, k] = transition_matrices[i, j, k]
                if first_layer:
                    copy_matrix(transition_matrix, transition_product)
                    first_layer = False
                else:
                    matrix_dot_matrix(transition_matrix, transition_product, tmp)
                    copy_matrix(tmp, transition_product)

    else:
        # non-cache loop
        first_layer = True
        for i in range(distance_in_layer.shape[0]):
            density = density_in_layer[i]
            distance = distance_in_layer[i]
            # only do something if distance > 0.
            if distance > 0.0:
                get_transition_matrix(
                    nubar,
                    energy,
                    density,
                    distance,
                    mix_nubar,
                    mix_nubar_conj_transp,
                    nsi_eps,
                    H_vac,
                    dm,
                    transition_matrix,
                )
                if first_layer:
                    copy_matrix(transition_matrix, transition_product)
                    first_layer = False
                else:
                    matrix_dot_matrix(transition_matrix, transition_product, tmp)
                    copy_matrix(tmp, transition_product)

    # convrt to flavour eigenstate basis
    matrix_dot_matrix(transition_product, mix_nubar_conj_transp, tmp)
    matrix_dot_matrix(mix_nubar, tmp, transition_product)

    # loop on neutrino types, and compute probability for neutrino i:
    for i in range(3):
        for j in range(3):
            raw_input_psi[j] = 0.0

        if use_mass_eigenstates:
            convert_from_mass_eigenstate(i + 1, mix_nubar, raw_input_psi)
        else:
            raw_input_psi[i] = 1.0

        matrix_dot_vector(transition_product, raw_input_psi, output_psi)
        osc_probs[i][0] += output_psi[0].real ** 2 + output_psi[0].imag ** 2
        osc_probs[i][1] += output_psi[1].real ** 2 + output_psi[1].imag ** 2
        osc_probs[i][2] += output_psi[2].real ** 2 + output_psi[2].imag ** 2


@myjit
def get_transition_matrix(
    nubar,
    energy,
    rho,
    baseline,
    mix_nubar,
    mix_nubar_conj_transp,
    nsi_eps,
    H_vac,
    dm,
    transition_matrix,
):
    """ Calculate neutrino flavour transition amplitude matrix

    Parameters
    ----------
    nubar : int
        +1 for neutrinos, -1 for antineutrinos

    energy : real float
        Neutrino energy, GeV

    rho : real float
        Density, g/cm^3

    baseline : real float
        Baseline, km

    mix_nubar : complex 2d array
        Mixing matrix, already conjugated if antineutrino

    mix_nubar_conj_transp : complex conjugate 2d array
        Conjugate transpose of mix_nubar

    nsi_eps : complex 2d array
        Non-standard interactions (set to 3x3 zeros for only standard oscillations)

    H_vac : complex 2d array
        Hamiltonian in vacuum, without the 1/2E term

    dm : real 2d array
        Mass splitting matrix, eV^2

    transition_matrix : complex 2d array (empty)
        Transition matrix in mass eigenstate basis

    Notes
    -----
    For neutrino (nubar > 0) or antineutrino (nubar < 0) with energy energy
    traversing layer of matter of uniform density rho with thickness baseline

    """

    H_mat = cuda.local.array(shape=(3, 3), dtype=ctype)
    dm_mat_vac = cuda.local.array(shape=(3, 3), dtype=ctype)
    dm_mat_mat = cuda.local.array(shape=(3, 3), dtype=ctype)
    H_full = cuda.local.array(shape=(3, 3), dtype=ctype)
    tmp = cuda.local.array(shape=(3, 3), dtype=ctype)
    H_mat_mass_eigenstate_basis = cuda.local.array(shape=(3, 3), dtype=ctype)

    # Compute the matter potential including possible non-standard interactions
    # in the flavor basis
    get_H_mat(rho, nsi_eps, nubar, H_mat)

    # Get the full Hamiltonian by adding together matter and vacuum parts
    one_over_two_e = 0.5 / energy
    for i in range(3):
        for j in range(3):
            H_full[i, j] = H_vac[i, j] * one_over_two_e + H_mat[i, j]

    # Calculate modified mass eigenvalues in matter from the full Hamiltonian and
    # the vacuum mass splittings
    get_dms(energy, H_full, dm, dm_mat_mat, dm_mat_vac)

    # Now we transform the matter (TODO: matter? full?) Hamiltonian back into the
    # mass eigenstate basis so we don't need to compute products of the effective
    # mixing matrix elements explicitly
    matrix_dot_matrix(H_mat, mix_nubar, tmp)
    matrix_dot_matrix(mix_nubar_conj_transp, tmp, H_mat_mass_eigenstate_basis)

    # We can now proceed to calculating the transition amplitude from the Hamiltonian
    # in the mass basis and the effective mass splittings
    get_transition_matrix_massbasis(
        baseline,
        energy,
        dm_mat_vac,
        dm_mat_mat,
        H_mat_mass_eigenstate_basis,
        transition_matrix,
    )


@myjit
def get_transition_matrix_massbasis(
    baseline,
    energy,
    dm_mat_vac,
    dm_mat_mat,
    H_mat_mass_eigenstate_basis,
    transition_matrix,
):
    """
    Calculate the transition amplitude matrix

    Parameters
    ----------
    baseline : float
        Baseline traversed, km

    energy : float
        Neutrino energy, GeV

    dm_mat_vac : complex 2d array

    dm_mat_mat : complex 2d array

    H_mat_mass_eigenstate_basis : complex 2d array

    transition_matrix : complex 2d array (empty)
        Transition matrix in mass eigenstate basis

    Notes
    -----
    - corrsponds to matrix A (equation 10) in original Barger paper
    - take into account generic potential matrix (=Hamiltonian)

    """

    product = cuda.local.array(shape=(3, 3, 3), dtype=ctype)

    clear_matrix(transition_matrix)

    get_product(energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product)

    # (1/2)*(1/(h_bar*c)) in units of GeV/(eV^2 km)
    hbar_c_factor = 2.534

    for k in range(3):
        arg = -dm_mat_vac[k, 0] * (baseline / energy) * hbar_c_factor
        c = cmath.exp(arg * 1.0j)
        for i in range(3):
            for j in range(3):
                transition_matrix[i, j] += c * product[i, j, k]


@myjit
def get_H_vac(mix_nubar, mix_nubar_conj_transp, dm_vac_vac, H_vac):
    """ Calculate vacuum Hamiltonian in flavor basis for neutrino or antineutrino

    Parameters:
    -----------
    mix_nubar : complex 2d array
        Mixing matrix, already conjugated if antineutrino

    mix_nubar_conj_transp : conjugate 2d array
        Conjugate transpose of mix_nubar

    dm_vac_vac : 2d array
        Matrix of mass splittings

    H_vac : complex 2d array
        Hamiltonian in vacuum, without the 1/2E term


    Notes
    ------
    The Hailtonian does not contain the energy dependent factor of
    1/(2 * E), as it will be added later

    """

    dm_vac_diag = cuda.local.array(shape=(3, 3), dtype=ctype)
    tmp = cuda.local.array(shape=(3, 3), dtype=ctype)

    clear_matrix(dm_vac_diag)

    dm_vac_diag[1, 1] = dm_vac_vac[1, 0] + 0j
    dm_vac_diag[2, 2] = dm_vac_vac[2, 0] + 0j

    matrix_dot_matrix(dm_vac_diag, mix_nubar_conj_transp, tmp)
    matrix_dot_matrix(mix_nubar, tmp, H_vac)


@myjit
def get_H_mat(rho, nsi_eps, nubar, H_mat):
    """ Calculate matter Hamiltonian in flavor basis

    Parameters:
    -----------
    rho : real float
        density, g/cm^3

    nsi_eps : complex 2d array
        Non-standard interactions (set to 3x3 zeros for only standard oscillations)

    nubar : int
        +1 for neutrinos, -1 for antineutrinos

    H_mat : complex 2d array (empty)
        matter hamiltonian

    Notes
    -----
    in the following, `a` is just the standard effective matter potential
    induced by charged-current weak interactions with electrons

    """

    # 2*sqrt(2)*Gfermi in (eV^2-cm^3)/(mole-GeV)
    tworttwoGf = 1.52588e-4
    a = 0.5 * rho * tworttwoGf
    if nubar == -1:
        a = -a

    # standard matter interaction Hamiltonian
    clear_matrix(H_mat)
    H_mat[0, 0] = a

    # Obtain effective non-standard matter interaction Hamiltonian
    nsi_rho_scale = (
        3.0  # // assume 3x electron density for "NSI"-quark (e.g., d) density
    )
    fact = nsi_rho_scale * a
    for i in range(3):
        for j in range(3):
            H_mat[i, j] += fact * nsi_eps[i, j]


@myjit
def get_dms(energy, H_mat, dm_vac_vac, dm_mat_mat, dm_mat_vac):
    """Compute the matter-mass vector M, dM = M_i-M_j and dMimj

    Parameters
    ----------
    energy : float
        Neutrino energy, GeV

    H_mat : complex 2d array
        matter hamiltonian

    dm_vac_vac : 2d array

    dm_mat_mat : complex 2d array (empty)

    dm_mat_vac : complex 2d array (empty)


    Notes
    -----
    Calculate mass eigenstates in matter
    neutrino or anti-neutrino (type already taken into account in Hamiltonian)
    of energy energy.

    - only god knows what happens in this function, somehow it seems to work

    """

    real_product_a = (H_mat[0, 1] * H_mat[1, 2] * H_mat[2, 0]).real
    real_product_b = (H_mat[0, 0] * H_mat[1, 1] * H_mat[2, 2]).real

    norm_H_e_mu_sq = H_mat[0, 1].real ** 2 + H_mat[0, 1].imag ** 2
    norm_H_e_tau_sq = H_mat[0, 2].real ** 2 + H_mat[0, 2].imag ** 2
    norm_H_mu_tau_sq = H_mat[1, 2].real ** 2 + H_mat[1, 2].imag ** 2

    c1 = (
        (H_mat[0, 0].real * (H_mat[1, 1] + H_mat[2, 2])).real
        - (H_mat[0, 0].imag * (H_mat[1, 1] + H_mat[2, 2])).imag
        + (H_mat[1, 1].real * H_mat[2, 2]).real
        - (H_mat[1, 1].imag * H_mat[2, 2]).imag
        - norm_H_e_mu_sq
        - norm_H_mu_tau_sq
        - norm_H_e_tau_sq
    )

    c0 = (
        H_mat[0, 0].real * norm_H_mu_tau_sq
        + H_mat[1, 1].real * norm_H_e_tau_sq
        + H_mat[2, 2].real * norm_H_e_mu_sq
        - 2.0 * real_product_a
        - real_product_b
    )

    c2 = -H_mat[0, 0].real - H_mat[1, 1].real - H_mat[2, 2].real

    one_over_two_e = 0.5 / energy
    one_third = 1.0 / 3.0
    two_third = 2.0 / 3.0

    x = dm_vac_vac[1, 0]
    y = dm_vac_vac[2, 0]

    c2_v = -one_over_two_e * (x + y)

    p = c2 ** 2 - 3.0 * c1
    p_v = one_over_two_e ** 2 * (x ** 2 + y ** 2 - x * y)
    p = max(0.0, p)

    q = -13.5 * c0 - c2 ** 3 + 4.5 * c1 * c2
    q_v = one_over_two_e ** 3 * (x + y) * ((x + y) ** 2 - 4.5 * x * y)

    tmp = p ** 3 - q ** 2
    tmp_v = p_v ** 3 - q_v ** 2

    tmp = max(0.0, tmp)

    theta = cuda.local.array(shape=(3), dtype=ftype)
    theta_v = cuda.local.array(shape=(3), dtype=ftype)
    m_mat = cuda.local.array(shape=(3), dtype=ftype)
    m_mat_u = cuda.local.array(shape=(3), dtype=ftype)
    m_mat_v = cuda.local.array(shape=(3), dtype=ftype)

    a = two_third * math.pi
    res = math.atan2(math.sqrt(tmp), q) * one_third
    theta[0] = res + a
    theta[1] = res - a
    theta[2] = res
    res_v = math.atan2(math.sqrt(tmp_v), q_v) * one_third
    theta_v[0] = res_v + a
    theta_v[1] = res_v - a
    theta_v[2] = res_v

    b = two_third * math.sqrt(p)
    b_v = two_third * math.sqrt(p_v)

    for i in range(3):
        m_mat_u[i] = (
            2.0 * energy * (b * math.cos(theta[i]) - c2 * one_third + dm_vac_vac[0, 0])
        )
        m_mat_v[i] = (
            2.0
            * energy
            * (b_v * math.cos(theta_v[i]) - c2_v * one_third + dm_vac_vac[0, 0])
        )

    # Sort according to which reproduce the vaccum eigenstates
    for i in range(3):
        tmp_v = abs(dm_vac_vac[i, 0] - m_mat_v[0])
        k = 0
        for j in range(3):
            tmp = abs(dm_vac_vac[i, 0] - m_mat_v[j])
            if tmp < tmp_v:
                k = j
                tmp_v = tmp
        m_mat[i] = m_mat_u[k]

    for i in range(3):
        for j in range(3):
            dm_mat_mat[i, j] = m_mat[i] - m_mat[j]
            dm_mat_vac[i, j] = m_mat[i] - dm_vac_vac[j, 0]


@myjit
def get_product(energy, dm_mat_vac, dm_mat_mat, H_mat_mass_eigenstate_basis, product):
    """
    Parameters
    ----------
    energy : float
        Neutrino energy, GeV

    dm_mat_vac : complex 2d array

    dm_mat_mat : complex 2d array

    H_mat_mass_eigenstate_basis : complex 2d array

    product : complex 3d-array (empty)

    """

    H_minus_M = cuda.local.array(shape=(3, 3, 3), dtype=ctype)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                H_minus_M[i, j, k] = 2.0 * energy * H_mat_mass_eigenstate_basis[i, j]
                if i == j:
                    H_minus_M[i, j, k] -= dm_mat_vac[k, j]
                # also, cler product
                product[i, j, k] = 0.0

    # Calculate the product in eq.(10) of H_minus_M for j!=k
    for i in range(3):
        for j in range(3):
            for k in range(3):
                product[i, j, 0] += H_minus_M[i, k, 1] * H_minus_M[k, j, 2]
                product[i, j, 1] += H_minus_M[i, k, 2] * H_minus_M[k, j, 0]
                product[i, j, 2] += H_minus_M[i, k, 0] * H_minus_M[k, j, 1]
            product[i, j, 0] /= dm_mat_mat[0, 1] * dm_mat_mat[0, 2]
            product[i, j, 1] /= dm_mat_mat[1, 2] * dm_mat_mat[1, 0]
            product[i, j, 2] /= dm_mat_mat[2, 0] * dm_mat_mat[2, 1]


@myjit
def convert_from_mass_eigenstate(state, mix_nubar, psi):
    """
    Parameters
    ----------
    state : (un?)signed int

    mix_nubar : complex 2d array
        Mixing matrix, already conjugated if antineutrino

    psi : complex 1d-array (empty)


    Notes
    -----
    this is untested!

    """

    mass = cuda.local.array(shape=(3), dtype=ctype)

    lstate = state - 1
    for i in range(3):
        mass[i] = 1.0 if lstate == i else 0.0

    # note: mix_nubar is already taking into account whether we're considering
    # nu or anti-nu
    matrix_dot_vector(mix_nubar, mass, psi)
