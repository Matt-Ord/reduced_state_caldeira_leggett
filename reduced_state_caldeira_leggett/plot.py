from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

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
    plot_isotropic_noise_kernel_2d_x,
    plot_kernel_2d,
)
from surface_potential_analysis.operator.operator_list import (
    select_diagonal_operator,
    select_operator,
)
from surface_potential_analysis.operator.plot import (
    plot_diagonal_operator_along_diagonal_1d_x,
    plot_diagonal_operator_along_diagonal_2d_x,
    plot_eigenstate_occupations,
    plot_operator_2d,
)
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
    plot_potential_2d_x,
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
    get_potential,
    get_temperature_corrected_noise_operators,
    get_true_noise_kernel,
)

if TYPE_CHECKING:
    from matplotlib.lines import Line2D
    from surface_potential_analysis.types import SingleFlatIndexLike


def plot_basis_states(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_potential(
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
    potential = get_potential(
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
    potential = get_potential(system, config.shape, config.resolution)
    fig, _, _ = plot_potential_2d_x(potential)
    fig.show()
    input()


def plot_noise_operators(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    idx: SingleFlatIndexLike = 0,
) -> None:
    """Plot the noise operators generated."""
    operators = get_noise_operators(system, config)
    operator = select_diagonal_operator(operators, idx=idx)

    for i in range(len(config.shape)):
        fig, ax, _ = plot_diagonal_operator_along_diagonal_1d_x(
            operator,
            axes=(i,),
            measure="abs",
        )
        ax.set_title("fitted noise operator")
        fig.show()

    for i in range(len(config.shape)):
        for j in range(i + 1, len(config.shape)):
            fig, ax, _ = plot_diagonal_operator_along_diagonal_2d_x(
                operator,
                axes=(i, j),
                measure="abs",
            )
            ax.set_title("fitted noise operator")
            fig.show()

    input()


def plot_noise_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot 1d isotropic noise kernel.

    True kernel and the fitted kernel compared.
    """
    kernel_real = get_true_noise_kernel(system, config)
    kernel_fitted = get_noise_kernel(system, config)

    for i in range(len(config.shape)):
        fig, ax, line1 = plot_isotropic_noise_kernel_1d_x(kernel_real, axes=(i,))
        line1.set_label("actual noise")
        fig.show()

        fig, _, line2 = plot_isotropic_noise_kernel_1d_x(
            kernel_fitted,
            axes=(i,),
            ax=ax,
        )
        line2.set_linestyle("--")
        line2.set_label("fitted noise")

        ax.set_title(
            f"noise kernel, fit method = {config.fit_method}, "
            f"n = {config.n_polynomial}, "
            f"temperature = {config.temperature}",
        )
        ax.legend()
        fig.show()

    for i in range(len(config.shape)):
        for j in range(i + 1, len(config.shape)):
            fig, ax, line1 = plot_isotropic_noise_kernel_2d_x(kernel_real, axes=(i, j))
            line1.set_label("actual noise")
            fig.show()

            ax.set_title("True kernel in 2d")

            fig, ax, line2 = plot_isotropic_noise_kernel_2d_x(
                kernel_fitted,
                axes=(i, j),
            )
            line2.set_linestyle("--")
            line2.set_label("fitted noise")

            ax.set_title("Fitted kernel in 2d")
            ax.legend()
            fig.show()

    operators = get_noise_operators(system, config)
    diagonal = get_diagonal_kernel_from_diagonal_operators(operators)
    fig, ax, _ = plot_diagonal_kernel_2d(diagonal)
    ax.set_title("Full noise kernel")
    fig.show()

    input()


def get_time_and_run(
    system: PeriodicSystem,
    config: SimulationConfig,
    parameter: np.ndarray[tuple[int], np.dtype[Any]],
    *,
    n_run: int,
) -> list[list[list[float]] | list[float]]:
    rng = np.random.default_rng()
    runtime: list[list[float]] = [[] for _ in range(len(parameter))]
    error: list[float] = []
    for n in range(n_run):
        shuffled_param_list = rng.permutation(parameter)
        for s in shuffled_param_list:
            config.shape = (s,)
            times = get_operators_fit_time(system, config)
            idx_in_original = np.argwhere(parameter == s)[0][0]
            runtime[idx_in_original].append(
                times,
            )
        if (n + 1) % 100 == 0 and n != 0:
            time.sleep(2.0)
    error = np.std(runtime, axis=1) / np.sqrt(n_run)
    runtime = np.mean(runtime, axis=1)
    return [runtime, error]


def plot_operators_fit_time_against_number_of_states(
    system: PeriodicSystem,
    config: SimulationConfig,
    size: np.ndarray[tuple[int], np.dtype[Any]],
    *,
    n_run: int,
) -> None:
    n_data_pts = size * config.resolution[0]
    runtime = get_time_and_run(system, config, size, n_run=n_run)[0]
    error = get_time_and_run(system, config, size, n_run=n_run)[1]

    def _get_complexity(x, a):
        match config.fit_method:
            case "fitted polynomial" | "explicit polynomial":
                return a * x / x
            case "fft":
                return a * x * np.log(x)
            case "eigenvalue":
                return a * x**3

    popt, _ = cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        scipy.optimize.curve_fit(
            _get_complexity,
            n_data_pts,
            runtime,
            sigma=error,
        ),
    )
    match config.fit_method:
        case "fitted polynomial" | "explicit polynomial":
            plt.plot(
                n_data_pts,
                _get_complexity(n_data_pts, popt[0]),
                marker="o",
                label=f"complexity {popt[0]}",
                linestyle="none",
            )
        case "fft":
            plt.plot(
                n_data_pts,
                _get_complexity(n_data_pts, popt[0]),
                marker="o",
                label=f"complexity {popt[0]}" r"$N \cdot$" "ln" r"$N$",
                linestyle="none",
            )
        case "eigenvalue":
            plt.plot(
                n_data_pts,
                _get_complexity(n_data_pts, popt[0]),
                marker="o",
                label=f"complexity {popt[0]}" r"$n^3$",
                linestyle="none",
            )
    plt.errorbar(
        x=n_data_pts,
        y=runtime,
        yerr=error,
        fmt="x",
        capsize=5.0,
        linestyle="none",
        label="Measured runtime",
    )
    plt.xlabel("number of states")
    plt.ylabel("runtime/seconds")
    plt.title(
        f"{config.fit_method}, N = {config.n_polynomial},\n"
        f"temperature = {config.temperature}, number of runs = {n_run}",
    )
    plt.legend()
    plt.show()

    input()


def plot_operators_fit_time_against_n_polynomial(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    n_terms_range: int,
    n_run: int,
) -> None:
    n_range = np.arange(10, n_terms_range, 10)
    runtime = get_time_and_run(system, config, n_range, n_run=n_run)[0]
    error = get_time_and_run(system, config, n_range, n_run=n_run)[1]

    def _get_complexity(x, a):
        match config.fit_method:
            case "eigenvalue" | "fft":
                return a * x / x
            case "fitted polynomial" | "explicit polynomial":
                return a * x**3

    popt, _ = cast(
        np.ndarray[tuple[int], np.dtype[np.float64]],
        scipy.optimize.curve_fit(
            _get_complexity,
            n_range,
            runtime,
            sigma=error,
        ),
    )

    match config.fit_method:
        case "eigenvalue" | "fft":
            plt.plot(
                n_range,
                _get_complexity(n_range, popt[0]),
                marker="o",
                label=f"complexity {popt[0]}",
                linestyle="none",
            )
        case "fitted polynomial" | "explicit polynomial":
            plt.plot(
                n_range,
                _get_complexity(n_range, popt[0]),
                marker="o",
                label=f"complexity {popt[0]}" r"$n^3$",
                linestyle="none",
            )
    plt.errorbar(
        x=n_range,
        y=runtime,
        yerr=error,
        fmt="x",
        capsize=5.0,
        linestyle="none",
        label="Measured runtime",
    )
    plt.xlabel("n terms")
    plt.ylabel("runtime/seconds")
    plt.title(
        f"{config.fit_method}, number of states = {config.shape[0]*config.resolution[0]},\n"
        f"temperature = {config.temperature}, number of runs = {n_run}",
    )
    plt.legend()
    plt.show()

    input()


def plot_isotropic_kernel_percentage_error(
    system: PeriodicSystem,
    config_list: list[SimulationConfig],
) -> None:
    true_kernel = get_true_noise_kernel(system, config_list[0])
    _, ax, hold_place = plot_isotropic_kernel_error(true_kernel, true_kernel)
    line: list[Line2D] = [hold_place for _ in range(len(config_list))]

    for i in range(len(config_list)):
        operators = get_noise_operators(system, config_list[i])
        fitted_kernel = get_diagonal_kernel_from_diagonal_operators(operators)
        fitted_kernel = as_isotropic_kernel_from_diagonal(
            fitted_kernel,
            assert_isotropic=True,
        )
        fig, _, line[i] = plot_isotropic_kernel_error(true_kernel, fitted_kernel, ax=ax)
        line[i].set_label(
            f"fit method = {config_list[i].fit_method}, power of polynomial terms included = {config_list[i].n_polynomial}",
        )

    ax.set_title(
        "comparison of noise kernel percentage error,\n"
        f"number of states = {config_list[0].shape[0]*config_list[0].resolution[0]}",
    )
    ax.set_ylabel("Percentage Error, %")
    ax.legend()
    fig.show()

    input()
