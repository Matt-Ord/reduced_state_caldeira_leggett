from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.constants import hbar
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    solve_stochastic_schrodinger_equation_rust_banded,
)
from surface_potential_analysis.operator.operator import SingleBasisOperator
from surface_potential_analysis.operator.operator_list import (
    select_operator,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
)
from surface_potential_analysis.util.decorators import npy_cached_dict, timed

from .system import (
    PeriodicSystem,
    SimulationConfig,
    get_hamiltonian,
    get_initial_state,
    get_noise_operators,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )


def _get_stochastic_evolution_cache(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    n: int,
    step: int,
    dt_ratio: float = 500,
) -> Path:
    return Path(
        f"examples/data/{system.id}/stochastic.{config.shape[0]}.{config.n_bands}.{n}.{step}.{dt_ratio}.milsten.0.npz",
    )


@npy_cached_dict(
    _get_stochastic_evolution_cache,
    load_pickle=True,
)
@timed
def get_stochastic_evolution(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    n: int,
    step: int,
    dt_ratio: float = 500,
) -> StateVectorList[
    StackedBasisLike[
        FundamentalBasis[Literal[1]],
        EvenlySpacedTimeBasis[int, int, int],
    ],
    StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
]:
    hamiltonian = get_hamiltonian(system, config)

    initial_state = get_initial_state(system, config)
    dt = hbar / (np.max(np.abs(hamiltonian["data"])) * dt_ratio)
    times = EvenlySpacedTimeBasis(n, step, 0, n * step * dt)
    temperature = 155

    operators = get_noise_operators(system, config, temperature)
    operator_list = list[SingleBasisOperator[Any]]()
    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    print(operators["basis"])  # noqa: T201
    print("Collapse Operators")  # noqa: T201
    print("------------------")  # noqa: T201
    for idx in args[1:7]:
        operator = select_operator(
            operators,
            idx=idx,
        )
        print(operators["eigenvalue"][idx])  # noqa: T201
        operator["data"] *= np.lib.scimath.sqrt(operators["eigenvalue"][idx] * hbar)

        print(np.max(np.abs(operator["data"])) ** 2)  # noqa: T201
        operator_list.append(operator)

    print("")  # noqa: T201
    print("Coherent Operator")  # noqa: T201
    print("------------------")  # noqa: T201
    print(np.max(np.abs(hamiltonian["data"])))  # noqa: T201
    ## This is roughly the number of timesteps per full rotation of phase
    ## should be much less than 1...
    print(times.fundamental_dt * np.max(np.abs(hamiltonian["data"])) / hbar)  # noqa: T201

    return solve_stochastic_schrodinger_equation_rust_banded(
        initial_state,
        times,
        hamiltonian,
        operator_list,
        method="Order2ExplicitWeak",
    )
