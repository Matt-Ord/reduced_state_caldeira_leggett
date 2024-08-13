from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy
import scipy.optimize
from matplotlib import pyplot as plt
from surface_potential_analysis.kernel.kernel import (
    as_diagonal_kernel_from_full,
    as_diagonal_kernel_from_isotropic,
    as_full_kernel_from_diagonal,
    as_isotropic_kernel_from_diagonal,
    get_diagonal_kernel_from_diagonal_operators,
    get_full_kernel_from_operators,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel_2d,
    plot_diagonal_kernel_truncation_error,
    plot_isotropic_kernel_error,
    plot_isotropic_noise_kernel_1d_x,
    plot_kernel_2d,
)
from surface_potential_analysis.operator.operator import as_operator
from surface_potential_analysis.operator.operator_list import (
    select_diagonal_operator,
    select_operator,
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
    op = select_diagonal_operator(operators, idx=1)
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


# fitted with complexity
def plot_chebyshev_fit_time(
    system: PeriodicSystem,
    config: SimulationConfig,
    size: np.ndarray[tuple[int, int], np.dtype[Any]],
    n_run: int,
) -> None:
    np.random.shuffle(size)
    n_data_pts = []
    config.fit_method = "poly fit"
    runtime_fit = []
    error_fit = []
    for s in size:
        config.shape = (s,)
        times_fit = []
        for n in range(n_run):
            times = get_operators_fit_time(system, config)
            times_fit.append(times[0][0])
        avg_time_fit = np.mean(np.array(times_fit)).item()
        runtime_fit.append(avg_time_fit)
        time_err_fit = np.std(np.array(times_fit)).item() / np.sqrt(n_run)
        error_fit.append(time_err_fit)
        n_data_pts.append(s * config.resolution[0])
        if (n + 1) % 100 == 0 and n != 0:
            time.sleep(2.0)

    def _runtime_scale(x, a, b):
        return a * x**2 + b * x**3

    popt, _ = scipy.optimize.curve_fit(
        _runtime_scale,
        size * config.resolution,
        np.array(runtime_fit),
        sigma=np.array(error_fit),
    )
    plt.plot(
        size * config.resolution,
        popt[0] * np.array(size * config.resolution) ** 2
        + popt[1] * np.array(size * config.resolution) ** 3,
        marker="o",
        label=f"complexity\n, {popt[0]/config.n_polynomial}"
        r"$N\cdot n^2$"
        f"+{popt[1]}"
        r"$n^3$",
        linestyle="none",
    )
    plt.errorbar(
        x=n_data_pts,
        y=np.array(runtime_fit),
        yerr=np.array(error_fit),
        fmt="x",
        capsize=5.0,
        linestyle="none",
        label="numpy fit",
    )

    plt.xlabel("number of states")
    plt.ylabel("runtime/seconds")
    plt.title(
        f"Chebyshev, N = {config.n_polynomial},\n"
        f"temperature = {config.temperature}, number of run = {n_run}",
    )
    plt.legend()
    plt.show()

    input()


def plot_get_trig_operators_time(
    system: PeriodicSystem,
    config: SimulationConfig,
    size: np.ndarray[tuple[int, int], np.dtype[Any]],
    n_run: int,
) -> None:
    np.random.shuffle(size)
    nk_pts_length = []
    config.fit_method = "poly fit"
    runtime_get_op = []
    error_get_op = []
    for s in size:
        config.shape = (s,)
        times_get_op = []
        for n in range(n_run):
            times = get_operators_fit_time(system, config)
            times_get_op.append(times[0][1])
        avg_time_get_op = np.mean(np.array(times_get_op)).item()
        runtime_get_op.append(avg_time_get_op)
        time_err_get_op = np.std(np.array(times_get_op)).item() / np.sqrt(n_run)
        error_get_op.append(time_err_get_op)
        nk_pts_length.append(s * config.resolution[0])
        if (n + 1) % 100 == 0 and n != 0:
            time.sleep(2.0)

    def _runtime_scale(x, a):
        return a * x

    popt, _ = scipy.optimize.curve_fit(
        _runtime_scale,
        size * config.resolution,
        np.array(runtime_get_op),
        sigma=np.array(error_get_op),
    )
    plt.plot(
        size * config.resolution,
        popt[0] * np.array(size * config.resolution),
        marker="o",
        label=f"complexity\n"
        f"{popt[0]/(config.n_polynomial+1)}n_terms"
        r"\cdot"
        f"nk_points length",
        linestyle="none",
    )

    plt.errorbar(
        x=(size * config.resolution[0]),
        y=np.array(runtime_get_op),
        yerr=np.array(error_get_op),
        fmt="x",
        capsize=5.0,
        linestyle="none",
        label="get operators",
    )
    plt.xlabel("number of states")
    plt.ylabel("runtime/seconds")
    plt.title(
        f"Get trig operators, n = {config.n_polynomial},\n"
        f"temperature = {config.temperature}, number of run = {n_run}",
    )
    plt.legend()
    plt.show()

    input()


def plot_fft_fit_time(
    system: PeriodicSystem,
    config: SimulationConfig,
    size: np.ndarray[tuple[int, int], np.dtype[Any]],
    n_run: int,
) -> None:
    np.random.shuffle(size)
    n_data_pts = []
    config.fit_method = "fft"
    runtime_fft = []
    error_fft = []
    for s in size:
        config.shape = (s,)
        times_fft = []
        for n in range(n_run):
            times = get_operators_fit_time(system, config)
            times_fft.append(times[0])
        avg_time_fft = np.mean(np.array(times_fft)).item()
        runtime_fft.append(avg_time_fft)
        time_err_fft = np.std(np.array(times_fft)).item() / np.sqrt(n_run)
        error_fft.append(time_err_fft)
        n_data_pts.append(s * config.resolution[0])
        if (n + 1) % 100 == 0 and n != 0:
            time.sleep(2.0)

    def _runtime_scale(x, a):
        return a * x * np.log(x)

    popt, _ = scipy.optimize.curve_fit(
        _runtime_scale,
        size * config.resolution,
        np.array(runtime_fft),
        sigma=np.array(error_fft),
    )
    plt.plot(
        size * config.resolution,
        popt[0]
        * np.array(size * config.resolution)
        * np.log(np.array(size * config.resolution)),
        marker="o",
        label=f"complexity\n, {popt[0]}" r"$N\cdot$" "ln" r"$N$",
        linestyle="none",
    )
    plt.errorbar(
        x=n_data_pts,
        y=np.array(runtime_fft),
        yerr=np.array(error_fft),
        fmt="x",
        capsize=5.0,
        linestyle="none",
        label="fft",
    )

    plt.xlabel("number of states")
    plt.ylabel("runtime/seconds")
    plt.title(
        f"FFT, N = {config.n_polynomial},\n"
        f"temperature = {config.temperature}, number of run = {n_run}",
    )
    plt.legend()
    plt.show()

    input()


def plot_isotropic_kernel_percentage_error(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    base_config: SimulationConfig | None = None,
) -> None:
    true_kernel = get_true_noise_kernel(system, config)
    operators = get_noise_operators(system, config)
    fitted_kernel = get_diagonal_kernel_from_diagonal_operators(operators)
    fitted_kernel = as_isotropic_kernel_from_diagonal(
        fitted_kernel,
        assert_isotropic=False,
    )
    fig, ax, line = plot_isotropic_kernel_error(true_kernel, fitted_kernel)

    if base_config is None:
        base_config = SimulationConfig(
            shape=config.shape,
            resolution=config.resolution,
            n_bands=config.n_bands,
            type=config.type,
            temperature=config.temperature,
            fit_method="fft",
            n_polynomial=None,
        )

    # to compare the errors between different methods directly
    base_operators = get_noise_operators(system, base_config)
    base_kernel = get_diagonal_kernel_from_diagonal_operators(base_operators)
    base_kernel = as_isotropic_kernel_from_diagonal(
        base_kernel,
        assert_isotropic=False,
    )
    fig, _, line1 = plot_isotropic_kernel_error(true_kernel, base_kernel, ax=ax)
    line1.set_label(
        f"fit method = {base_config.fit_method}, power of polynomial terms included = {base_config.n_polynomial}",
    )
    ax.set_title(
        "comparison of noise kernel percentage error,\n"
        f"number of states = {config.shape[0]*config.resolution[0]}",
    )
    ax.set_ylabel("Percentage Error, %")
    line.set_label(
        f"fit method = {config.fit_method}, power of polynomial terms included = {config.n_polynomial}",
    )
    ax.legend()
    fig.show()
    input()
