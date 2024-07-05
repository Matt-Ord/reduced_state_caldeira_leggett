from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.explicit_basis import (
    explicit_stacked_basis_as_fundamental,
)
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.kernel.kernel import as_diagonal_kernel, as_noise_kernel
from surface_potential_analysis.kernel.kernel import (
    get_noise_kernel as get_noise_kernel_generic,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_kernel_truncation_error,
)
from surface_potential_analysis.kernel.plot import plot_kernel as plot_kernel_generic
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator_list import select_operator
from surface_potential_analysis.operator.plot import (
    plot_eigenstate_occupations,
    plot_operator_2d,
    plot_operator_along_diagonal,
)
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_basis,
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
    get_extended_interpolated_potential,
    get_hamiltonian,
    get_noise_kernel,
    get_noise_operators,
)


def plot_system_eigenstates(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = calculate_eigenvectors_hermitian(hamiltonian)
    basis = explicit_stacked_basis_as_fundamental(hamiltonian["basis"][0])
    converted = convert_state_vector_list_to_basis(eigenvectors, basis)

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(converted)):
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
    potential = get_extended_interpolated_potential(
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
    *,
    temperature: float = 155,
) -> None:
    kernel = get_noise_kernel(system, config, temperature)
    diagonal = as_diagonal_kernel(kernel)

    fig, _, _ = plot_diagonal_kernel(diagonal)
    fig.show()

    fig, _, _ = plot_kernel_generic(as_noise_kernel(diagonal))
    fig.show()

    fig, _ = plot_kernel_truncation_error(kernel)
    fig.show()

    corrected_operators = get_noise_operators(system, config, temperature)
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
    *,
    temperature: float = 155,
) -> None:
    operators = get_noise_operators(system, config, temperature)

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
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("black")
    line.set_linewidth(3)

    states = get_stochastic_evolution(system, config, n=n, step=step, dt_ratio=dt_ratio)
    converted = convert_state_vector_list_to_basis(
        states,
        explicit_stacked_basis_as_fundamental(states["basis"][1]),
    )
    _fig, _, _animnation_ = animate_state_over_list_1d_x(converted, ax=ax.twinx())

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

    # fig0, ax0 = plot_all_band_occupations(hamiltonian, states)

    # fig1, ax1 = fig0, ax0
    # fig1, ax1, _ani = animate_all_band_occupations(hamiltonian, states)

    fig2, ax2, line = plot_average_band_occupation(hamiltonian, states)

    for ax in [ax2]:
        _, _, line = plot_eigenstate_occupations(hamiltonian, 150, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Boltzmann occupation"])

    # fig0.show()
    # fig1.show()
    fig2.show()
    input()


def plot_initial_state(system: PeriodicSystem, config: SimulationConfig) -> None:
    initial = get_initial_state(system, config)
    converted = convert_state_vector_to_basis(
        initial,
        explicit_stacked_basis_as_fundamental(initial["basis"]),
    )
    fig, _ax, _ = plot_state_1d_x(converted)

    fig.show()
    input()


def plot_noise_operator(
    system: PeriodicSystem,
    config: SimulationConfig,
    temperature: float,
) -> None:
    operator = select_operator(get_noise_operators(system, config, temperature), 0)
    basis = explicit_stacked_basis_as_fundamental(operator["basis"][0])
    converted = convert_operator_to_basis(operator, StackedBasis(basis, basis))
    fig, _ax, _ = plot_operator_along_diagonal(converted)

    fig.show()
    input()
