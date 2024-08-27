from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import scipy
import scipy.optimize
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.basis import FundamentalBasis
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
from surface_potential_analysis.util.plot import get_figure

from reduced_state_caldeira_leggett.dynamics import (
    get_initial_state,
    get_stochastic_evolution,
)
from reduced_state_caldeira_leggett.system import (
    FitMethod,
    PeriodicSystem,
    SimulationConfig,
    get_full_noise_operators,
    get_hamiltonian,
    get_noise_kernel,
    get_noise_operators,
    get_potential,
    get_temperature_corrected_noise_operators,
    get_true_noise_kernel,
)

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_list import (
        StatisticalValueList,
    )
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
    operators = get_full_noise_operators(system, config)
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


def _get_operators_fit_time(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> float:
    ts = datetime.datetime.now(tz=datetime.UTC)
    _operators = get_noise_operators(system, config)
    te = datetime.datetime.now(tz=datetime.UTC)
    return (te - ts).total_seconds()


def _get_runtime_of_get_operator(
    system: PeriodicSystem,
    configs: list[SimulationConfig],
    *,
    n_run: int,
) -> StatisticalValueList[FundamentalBasis[int]]:
    rng = np.random.default_rng()
    runtime = np.zeros((len(configs), n_run), dtype=float)
    for n in range(n_run):
        for i in rng.permutation(np.arange(len(configs))):
            config = configs[cast(int, i)]
            times = _get_operators_fit_time(system, config)
            runtime[i, n] = times

    error = np.std(runtime, axis=1) / np.sqrt(n_run)
    runtime = np.mean(runtime, axis=1)
    return {
        "basis": FundamentalBasis(runtime.size),
        "data": runtime,
        "standard_deviation": error,
    }


def plot_operators_fit_time_against_number_of_states(
    system: PeriodicSystem,
    fit_method: FitMethod,
    n_states: tuple[np.ndarray[tuple[int], np.dtype[np.int64]]],
    *,
    n_run: int = 100,
) -> None:
    """Plot operator fit time against n states.

    Parameters
    ----------
    system : PeriodicSystem
    config : SimulationConfig
    sizes : tuple[np.ndarray[tuple[int], np.dtype[np.int64]]]
        The size parameter in SimulationConfig.
    n_run : int, optional
        n_run, by default 100

    Returns
    -------
    None

    """
    min_n_states = np.min(n_states).item()
    n_polynomial = (
        (min_n_states - 1) // 2 if fit_method in ("fft", "eigenvalue") else min_n_states
    )
    configs: list[SimulationConfig] = [
        SimulationConfig(
            shape=(1,),
            resolution=(n_state,),
            fit_method=fit_method,
            n_bands=n_state,
            type="bloch",
            temperature=0,
            n_polynomial=n_polynomial,
        )
        for n_state in n_states[0]
    ]

    runtime = _get_runtime_of_get_operator(
        system,
        configs,
        n_run=n_run,
    )

    def _get_complexity(
        x: np.ndarray[tuple[int], np.dtype[np.float64]],
        a: float,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        match fit_method:
            case "fitted polynomial" | "explicit polynomial":
                return a * np.ones_like(x)
            case "fft":
                return a * x * np.log(x)
            case "eigenvalue":
                return a * x**3

    def _get_complexity_label(a: float) -> str | None:
        match fit_method:
            case "fitted polynomial" | "explicit polynomial":
                return f"complexity {a}"
            case "fft":
                return rf"complexity {a} $N \cdot \ln N$"
            case "eigenvalue":
                rf"complexity {a} $n^3$"
                return None

    popt, _ = cast(
        tuple[np.ndarray[tuple[int], np.dtype[np.float64]], Any],
        scipy.optimize.curve_fit(
            _get_complexity,
            n_states[0],
            runtime["data"],
            sigma=runtime["standard_deviation"],
        ),
    )
    fig, ax = plt.subplots()
    ax.plot(
        n_states[0],
        _get_complexity(n_states[0].astype(np.float64), popt[0]),
        label=_get_complexity_label(popt[0]),
    )
    ax.errorbar(
        x=n_states[0],
        y=runtime["data"],
        yerr=runtime["standard_deviation"],
        fmt="x",
        capsize=5.0,
        linestyle="none",
        label="Measured runtime",
    )
    ax.set_xlabel("number of states")
    ax.set_ylabel("runtime /s")
    ax.set_title(
        f"{fit_method} method with {n_polynomial} polynomial terms"
        f"\naveraged over {n_run} repeats",
    )
    ax.legend()
    fig.show()

    input()


def plot_operators_fit_time_against_n_polynomial(
    system: PeriodicSystem,
    fit_method: FitMethod,
    n_polynomials: np.ndarray[tuple[int], np.dtype[np.int64]],
    *,
    n_run: int = 100,
) -> None:
    max_n_polynomials = np.max(n_polynomials).item()
    n_states = (
        2 * max_n_polynomials + 1
        if fit_method in ("fft", "eigenvalue")
        else max_n_polynomials
    )
    configs: list[SimulationConfig] = [
        SimulationConfig(
            shape=(1,),
            resolution=(n_states,),
            fit_method=fit_method,
            n_bands=n_states,
            type="bloch",
            temperature=0,
            n_polynomial=n_polynomial,
        )
        for n_polynomial in n_polynomials
    ]

    runtime = _get_runtime_of_get_operator(
        system,
        configs,
        n_run=n_run,
    )

    def _get_complexity(
        x: np.ndarray[tuple[int], np.dtype[np.float64]],
        a: float,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        match fit_method:
            case "eigenvalue" | "fft":
                return a * np.ones_like(x)
            case "fitted polynomial" | "explicit polynomial":
                return a * x**3

    def _get_complexity_label(a: float) -> str | None:
        match fit_method:
            case "eigenvalue" | "fft":
                return f"complexity {a}"
            case "fitted polynomial" | "explicit polynomial":
                return rf"complexity {a} $n^3$"

    parameters, _ = cast(
        tuple[np.ndarray[tuple[int], np.dtype[np.float64]], Any],
        scipy.optimize.curve_fit(
            _get_complexity,
            n_polynomials,
            runtime["data"],
            sigma=runtime["standard_deviation"],
        ),
    )
    fig, ax = get_figure(None)
    ax.plot(
        n_polynomials,
        _get_complexity(n_polynomials.astype(float), *parameters),
        label=_get_complexity_label(*parameters),
    )
    ax.errorbar(
        x=n_polynomials,
        y=runtime["data"],
        yerr=runtime["standard_deviation"],
        fmt="x",
        capsize=5.0,
        linestyle="none",
        label="Runtime",
    )
    ax.set_xlabel("n terms")
    ax.set_ylabel("runtime/seconds")
    ax.set_title(
        f"{fit_method} method with {n_states} states,\naveraged over {n_run} runs",
    )
    ax.legend()
    fig.show()

    input()


def plot_kernel_error_comparison(
    system: PeriodicSystem,
    configs: Sequence[SimulationConfig],
) -> None:
    fig, ax = get_figure(None)
    lines = []

    for config in configs:
        true_kernel = get_true_noise_kernel(system, config)
        operators = get_noise_operators(system, config)
        fitted_kernel = get_diagonal_kernel_from_diagonal_operators(operators)
        fitted_kernel = as_isotropic_kernel_from_diagonal(
            fitted_kernel,
            assert_isotropic=True,
        )
        fig, _, line = plot_isotropic_kernel_error(true_kernel, fitted_kernel, ax=ax)
        line.set_label(
            f"{config.fit_method} method, {config.n_polynomial} terms",
        )
        lines.append(line)

    ax.set_title(
        "Comparison of noise kernel percentage error,\n"
        f"with {configs[0].shape[0]*configs[0].resolution[0]} states",
    )
    ax.set_ylabel("Percentage Error, %")
    ax.legend()
    fig.show()

    input()
