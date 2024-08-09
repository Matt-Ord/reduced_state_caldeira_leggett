from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.kernel.kernel import (
    as_diagonal_kernel_from_full,
    as_diagonal_kernel_from_isotropic,
    as_full_kernel_from_diagonal,
    get_full_kernel_from_operators,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel_2d,
    plot_diagonal_kernel_truncation_error,
    plot_isotropic_noise_kernel_1d_x,
    plot_kernel_2d,
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
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_x,
    plot_average_eigenstate_occupation,
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
    get_operators_fit_time,
    get_potential_1d,
    get_potential_2d,
    get_temperature_corrected_noise_operators,
    get_true_noise_kernel,
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
    diagonal = as_diagonal_kernel_from_isotropic(kernel)

    fig, _, _ = plot_diagonal_kernel_2d(diagonal)
    fig.show()
    input()

    fig, _, _ = plot_kernel_2d(as_full_kernel_from_diagonal(diagonal))
    fig.show()

    fig, _, _ = plot_diagonal_kernel_truncation_error(diagonal)
    fig.show()

    corrected_operators = get_temperature_corrected_noise_operators(system, config)
    kernel_full = get_full_kernel_from_operators(corrected_operators)

    fig, _, _ = plot_kernel_2d(kernel_full)
    fig.show()

    diagonal = as_diagonal_kernel_from_full(kernel_full)
    fig, _, _ = plot_diagonal_kernel_2d(diagonal)
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

    fig2, ax2, line = plot_average_eigenstate_occupation(hamiltonian, states)

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


def plot_2d_111_potential(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    potential = get_potential_2d(system, config.shape, config.resolution)
    fig, _, _ = plot_potential_2d_x(potential)
    fig.show()
    input()


def plot_noise_operators(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the noise operators generated."""
    operators = get_noise_operators(system, config)
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
    kernel_real = get_true_noise_kernel(system, config)
    fig, ax, line1 = plot_isotropic_noise_kernel_1d_x(kernel_real)
    line1.set_label("true noise, no temperature correction")
    fig.show()

    kernel_isotropic_fitted = get_noise_kernel(system, config)
    fig, _, line2 = plot_isotropic_noise_kernel_1d_x(kernel_isotropic_fitted, ax=ax)
    line2.set_label("fitted noise, no temperature correction")

    ax.set_title(
        f"noise kernel, fit method = {config.fit_method}, n = {config.n_polynomial}, temperature = {config.temperature}",
    )
    ax.legend()
    fig.show()
    input()


def plot_kernel_fit_runtime(
    system: PeriodicSystem,
    config: SimulationConfig,
    size: np.ndarray[tuple[int, int], np.dtype[Any]],
    n_run: int,
) -> None:
    runtime = []
    error = []
    shape = []
    for s in size:
        config.shape = s
        times = []
        for _ in range(n_run):
            times.append(get_operators_fit_time(system, config))
        avg_time = np.mean(np.array(times)).item()
        time_err = np.std(np.array(times)).item() / np.sqrt(n_run)
        runtime.append(avg_time)
        error.append(time_err)
        shape.append(s[0])

    plt.errorbar(x=np.array(shape), y=np.array(runtime), yerr=np.array(error))
    plt.xlabel("number of unit cell")
    plt.ylabel("runtime/seconds")
    plt.title(
        f"Runtime for fit method = {config.fit_method}, n = {config.n_polynomial}, temperature = {config.temperature}, number of run = {n_run}",
    )
    plt.show()

    input()
