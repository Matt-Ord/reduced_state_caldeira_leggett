from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.kernel.conversion import (
    convert_diagonal_noise_operator_list_to_basis,
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_parameters,
    get_gaussian_isotropic_noise_kernel,
)
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    as_diagonal_kernel,
    as_diagonal_noise_operators,
    as_isotropic_kernel,
    as_noise_kernel,
    get_diagonal_noise_kernel,
)
from surface_potential_analysis.kernel.kernel import (
    get_noise_kernel as get_noise_kernel_generic,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_isotropic_kernel_error,
    plot_isotropic_noise_kernel_1d_x,
    plot_kernel_truncation_error,
)
from surface_potential_analysis.kernel.plot import plot_kernel as plot_kernel_generic
from surface_potential_analysis.kernel.solve import (
    get_noise_operators_stacked_taylor_expansion,
)
from surface_potential_analysis.operator.operator import as_operator
from surface_potential_analysis.operator.operator_list import (
    select_operator,
    select_operator_diagonal,
)
from surface_potential_analysis.operator.plot import (
    plot_eigenstate_occupations,
    plot_operator_2d,
    plot_operator_along_diagonal,
)
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
    plot_potential_2d_x,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_x,
    plot_average_band_occupation,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from reduced_state_caldeira_leggett.dynamics import (
    get_initial_state,
    get_stochastic_evolution,
)
from reduced_state_caldeira_leggett.system import (
    PeriodicSystem,
    SimulationConfig,
    get_hamiltonian,
    get_noise_kernel,
    get_noise_operators,
    get_potential_1d,
    get_potential_2d,
    get_temperature_corrected_noise_operators,
)


def plot_system_eigenstates(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_potential_1d(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = calculate_eigenvectors_hermitian(hamiltonian)

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(eigenvectors)):
        plot_state_1d_x(state, ax=ax1)
        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_basis_states(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_potential_1d(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("black")
    line.set_linewidth(3)

    hamiltonian = get_hamiltonian(system, config)
    states = hamiltonian["basis"][0].vectors

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for i, state in enumerate(state_vector_list_into_iter(states)):
        _, _, line = plot_state_1d_x(state, ax=ax1)
        line.set_label(f"state {i}")

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_thermal_occupation(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    hamiltonian = get_hamiltonian(system, config)
    fig, _, _ = plot_eigenstate_occupations(hamiltonian, 150)

    fig.show()
    input()


def plot_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    kernel = get_noise_kernel(system, config)
    diagonal = as_diagonal_kernel(kernel)

    fig, _, _ = plot_diagonal_kernel(diagonal)
    fig.show()
    input()

    fig, _, _ = plot_kernel_generic(as_noise_kernel(diagonal))
    fig.show()

    fig, _ = plot_kernel_truncation_error(kernel)
    fig.show()

    corrected_operators = get_temperature_corrected_noise_operators(system, config)
    kernel_full = get_noise_kernel_generic(corrected_operators)

    fig, _, _ = plot_kernel_generic(kernel_full)
    fig.show()

    diagonal = as_diagonal_kernel(kernel_full)
    fig, _, _ = plot_diagonal_kernel(diagonal)
    fig.show()

    input()


def plot_lindblad_operator(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    operators = get_noise_operators(system, config)

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    for idx in args[:10]:
        operator = select_operator(operators, idx=idx)

        fig, ax, _ = plot_operator_2d(operator)
        ax.set_title("Operator")
        fig.show()

    input()


def plot_state_against_t(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    n: int,
    step: int,
    dt_ratio: float = 500,
) -> None:
    potential = get_potential_1d(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("black")
    line.set_linewidth(3)

    states = get_stochastic_evolution(system, config, n=n, step=step, dt_ratio=dt_ratio)

    _fig, _, _animnation_ = animate_state_over_list_1d_x(states, ax=ax.twinx())

    fig.show()
    input()


def plot_stochastic_occupation(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    dt_ratio: float = 500,
    n: int = 800,
    step: int = 4000,
) -> None:
    states = get_stochastic_evolution(
        system,
        config,
        n=n,
        step=step,
        dt_ratio=dt_ratio,
    )
    hamiltonian = get_hamiltonian(system, config)

    fig2, ax2, line = plot_average_band_occupation(hamiltonian, states)

    for ax in [ax2]:
        _, _, line = plot_eigenstate_occupations(hamiltonian, 150, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Boltzmann occupation"])

    fig2.show()
    input()


def plot_initial_state(system: PeriodicSystem, config: SimulationConfig) -> None:
    initial = get_initial_state(system, config)
    fig, _ax, _ = plot_state_1d_x(initial)

    fig.show()
    input()


def plot_noise_operator(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    operator = select_operator(get_noise_operators(system, config), 0)
    fig, _ax, _ = plot_operator_along_diagonal(operator)

    fig.show()
    input()


def plot_2d_111_potential(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    potential = get_potential_2d(system, config.shape, config.resolution)
    fig, _, _ = plot_potential_2d_x(potential)
    fig.show()
    input()


_B0 = TypeVar("_B0", bound=TupleBasisWithLengthLike[Any, Any])


def plot_new_noise_operators(
    kernel: IsotropicNoiseKernel[_B0],
    *,
    n: int = 1,
) -> None:
    """Plot the noise operators generated."""
    operators = get_noise_operators_stacked_taylor_expansion(kernel, n=n)
    op = select_operator_diagonal(operators, idx=1)
    fig1, ax1, _ = plot_operator_along_diagonal(as_operator(op), measure="real")
    ax1.set_title("fitted noise operator")
    fig1.show()
    input()


def plot_noise_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot 1d isotropic noise kernel.

    True kernel and the fitted kernel compared.
    """
    hamiltonian = get_hamiltonian(system, config)
    basis = hamiltonian["basis"][0]
    basis_x = stacked_basis_as_fundamental_position_basis(basis)
    a, lambda_ = get_effective_gaussian_parameters(
        hamiltonian["basis"][0],
        system.eta,
        config.temperature,
        lambda_factor=2 * np.sqrt(2),
    )
    kernel_real = get_gaussian_isotropic_noise_kernel(
        basis,
        a,
        lambda_,
    )
    fig, ax, line1 = plot_isotropic_noise_kernel_1d_x(kernel_real)
    line1.set_label("true noise, no temperature correction")
    fig.show()

    fit_op_ = get_noise_operators(system, config)
    converted_ = convert_diagonal_noise_operator_list_to_basis(
        fit_op_,
        TupleBasis(basis_x, basis_x),
    )
    kernel_isotropic_ = as_isotropic_kernel(
        get_diagonal_noise_kernel(as_diagonal_noise_operators(converted_)),
    )
    fig, _, line2 = plot_isotropic_noise_kernel_1d_x(kernel_isotropic_, ax=ax)
    line2.set_label("fitted noise, no temperature correction")

    fit_op = get_temperature_corrected_noise_operators(system, config)
    converted = convert_noise_operator_list_to_basis(
        fit_op,
        TupleBasis(basis_x, basis_x),
    )
    kernel_isotropic = as_isotropic_kernel(
        get_diagonal_noise_kernel(as_diagonal_noise_operators(converted)),
    )
    fig, _, line3 = plot_isotropic_noise_kernel_1d_x(kernel_isotropic, ax=ax)
    line3.set_label("fitted noise, with temperature correction")

    ax.set_title(
        f"noise kernel, fit method = {config.FitMethod}, n = {config.n_polynomial}, temperature = {config.temperature}",
    )
    ax.legend()
    fig.show()
    input()


def plot_isotropic_kernel_percentage_error(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    to_compare: bool = False,
    config1: SimulationConfig,
) -> None:
    true_kernel = get_noise_kernel(system, config)
    operators = get_noise_operators(system, config)
    fitted_kernel = get_diagonal_noise_kernel(operators)
    fitted_kernel = as_isotropic_kernel(fitted_kernel)
    fig, ax, line = plot_isotropic_kernel_error(true_kernel, fitted_kernel)

    # try test temperature correction------------
    # fit_op_no_T = get_noise_operators(system, config)
    # kernel_isotropic_no_T = as_isotropic_kernel(
    #     get_diagonal_noise_kernel(fit_op_no_T),
    # )
    # fit_op_T = get_temperature_corrected_noise_operators(system, config)
    # kernel_isotropic_T = as_isotropic_kernel(
    #     get_diagonal_noise_kernel(as_diagonal_noise_operators(fit_op_T)),
    # )
    # _see_T = as_diagonal_noise_operators(fit_op_T)
    # fig, ax, line = plot_isotropic_kernel_error(
    #     kernel_isotropic_no_T,
    #     kernel_isotropic_T,
    # )
    # ax.set_title("percentage error between kernel with and without temperature")
    # line.set_label(
    #     f"fit method = {config.FitMethod}, power of polynomial terms included = {config.n_polynomial}",
    # )
    # ------------------------------

    # to compare the errors between different methods directly
    if to_compare:
        operators1 = get_noise_operators(system, config1)
        fitted_kernel1 = get_diagonal_noise_kernel(operators1)
        fitted_kernel1 = as_isotropic_kernel(fitted_kernel1)
        fig, _, line1 = plot_isotropic_kernel_error(true_kernel, fitted_kernel1, ax=ax)
        line1.set_label(
            f"fit method = {config1.FitMethod}, power of polynomial terms included = {config1.n_polynomial}",
        )

    ax.set_title("comparison of noise kernel percentage error")
    ax.set_ylabel("Percentage Error, %")
    line.set_label(
        f"fit method = {config.FitMethod}, power of polynomial terms included = {config.n_polynomial}",
    )
    ax.legend()
    fig.show()
    input()
