from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis1d,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedBasis,
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.kernel.conversion import (
    convert_diagonal_kernel_to_basis,
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_noise_kernel,
    get_temperature_corrected_effective_gaussian_noise_operators,
)
from surface_potential_analysis.operator.operator import as_operator
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_full_bloch_hamiltonian,
    get_full_wannier_hamiltonian,
    get_wannier_basis,
)
from surface_potential_analysis.wavepacket.localization import (
    Wannier90Options,
    get_localization_operator_wannier90,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionListWithEigenvaluesList,
    generate_wavepacket,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisNoiseKernel,
        SingleBasisNoiseOperatorList,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.localization_operator import (
        LocalizationOperator,
    )

_L0Inv = TypeVar("_L0Inv", bound=int)


@dataclass
class PeriodicSystem:
    """Represents the properties of a 1D Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    barrier_energy: float
    lattice_constant: float
    mass: float
    gamma: float

    @property
    def eta(self) -> float:  # noqa: D102, ANN101
        return 2 * self.mass * self.gamma


@dataclass
class SimulationConfig:
    """Configure the detail of the simulation."""

    shape: tuple[int, ...]
    resolution: tuple[int, ...]
    n_bands: int
    type: Literal["bloch", "wannier"]
    temperature: float


HYDROGEN_NICKEL_SYSTEM = PeriodicSystem(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
    gamma=0.2e12,
)

FREE_LITHIUM_SYSTEM = PeriodicSystem(
    id="LiFree",
    barrier_energy=0,
    lattice_constant=3.615e-10,
    mass=1.152414898e-26,
    gamma=1.2e12,
)


SODIUM_COPPER_SYSTEM = PeriodicSystem(
    id="NaCu",
    barrier_energy=8.8e-21,
    lattice_constant=3.615e-10 / np.sqrt(2),
    mass=3.8175458e-26,
    gamma=0.2e12,
)


def _get_potential_1d(
    system: PeriodicSystem,
) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    """Generate potential for a periodic 1D system."""
    delta_x = np.sqrt(3) * system.lattice_constant / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": TupleBasis(axis), "data": vector}


def _get_interpolated_potential(
    system: PeriodicSystem,
    resolution: tuple[_L0Inv],
) -> Potential[
    TupleBasisWithLengthLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = _get_potential_1d(system)
    old = potential["basis"][0]
    basis = TupleBasis(
        TransformedPositionBasis1d[_L0Inv, Literal[3]](
            old.delta_x,
            old.n,
            resolution[0],
        ),
    )
    scaled_potential = potential["data"] * np.sqrt(resolution[0] / old.n)
    return convert_potential_to_basis(
        {"basis": basis, "data": scaled_potential},
        stacked_basis_as_fundamental_momentum_basis(basis),
    )


def get_extended_interpolated_potential(
    system: PeriodicSystem,
    shape: tuple[int],
    resolution: tuple[int],
) -> Potential[
    TupleBasisWithLengthLike[
        EvenlySpacedTransformedPositionBasis[int, int, Literal[0], Literal[1]]
    ]
]:
    """Generate potential for periodic 1D system."""
    interpolated = _get_interpolated_potential(system, resolution)
    old = interpolated["basis"][0]
    basis = TupleBasis(
        EvenlySpacedTransformedPositionBasis[int, int, Literal[0], Literal[1]](
            old.delta_x * shape[0],
            n=old.n,
            step=shape[0],
            offset=0,
        ),
    )
    scaled_potential = interpolated["data"] * np.sqrt(basis.fundamental_n / old.n)

    return {"basis": basis, "data": scaled_potential}


def get_2d_111_potential(
    system: PeriodicSystem,
    shape: tuple[int, int],
    resolution: tuple[int, int],
) -> Potential[
    TupleBasisWithLengthLike[
        FundamentalPositionBasis[int, Any],
        FundamentalPositionBasis[int, Any],
    ]
]:
    """Generate potential for 2D periodic system, for 111 plane of FCC lattice.

    Expression for potential from:

    [1] D. J. Ward
        A study of spin-echo lineshapes in helium atom scattering from adsorbates.

    [2]S. P. Rittmeyer et al
        Energy Dissipation during Diffusion at Metal Surfaces:
        Disentangling the Role of Phonons vs Electron-Hole Pairs.

    """
    vector_x = np.array(
        [system.lattice_constant * shape[0], 0],
    )
    vector_y = np.array(
        [
            system.lattice_constant * shape[1] * np.cos(np.pi / 3),
            system.lattice_constant * shape[1] * np.sin(np.pi / 3),
        ],
    )
    basis_x = FundamentalPositionBasis(vector_x, resolution[0] * shape[0])
    basis_y = FundamentalPositionBasis(vector_y, resolution[1] * shape[1])
    full_basis = TupleBasis(basis_x, basis_y)
    util = BasisUtil(full_basis)
    x_points = util.x_points_stacked

    zeta = 4 * np.pi / (np.sqrt(3) * system.lattice_constant)
    g = np.array(
        [
            np.array([zeta, 0]),
            np.array([zeta * np.cos(np.pi / 3), zeta * np.sin(np.pi / 3)]),
            np.array([-zeta * np.cos(np.pi / 3), zeta * np.sin(np.pi / 3)]),
        ],
    )
    potential = []
    for r in x_points.T:
        V_i = system.barrier_energy * np.cos(np.sum(np.multiply(g, r)))
        potential.append(V_i)
    potential = np.array(potential)
    return {"basis": full_basis, "data": potential}


def _get_full_hamiltonian(
    system: PeriodicSystem,
    shape: tuple[_L0Inv, ...],
    resolution: tuple[_L0Inv, ...],
    *,
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]],
]:
    bloch_fraction = np.array([0]) if bloch_fraction is None else bloch_fraction

    match len(shape):
        case 1:
            potential = get_extended_interpolated_potential(
                system,
                cast(tuple[int], shape),
                cast(tuple[int], resolution),
            )
        case 2:
            potential = get_2d_111_potential(
                system,
                cast(tuple[int, int], shape),
                cast(tuple[int, int], resolution),
            )
        case _:
            msg = "Currently only support 1 and 2D potentials"
            raise ValueError(msg)

    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_position_basis(potential["basis"]),
    )
    return total_surface_hamiltonian(converted, system.mass, bloch_fraction)


def get_wavepacket(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> BlochWavefunctionListWithEigenvaluesList[
    EvenlySpacedBasis[int, int, int],
    TupleBasisLike[*tuple[FundamentalBasis[int], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]],
]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]]
    ]:
        return _get_full_hamiltonian(
            system,
            tuple(1 for _ in config.shape),
            config.resolution,
            bloch_fraction=bloch_fraction,
        )

    return generate_wavepacket(
        hamiltonian_generator,
        save_bands=EvenlySpacedBasis(config.n_bands, 1, 0),
        list_basis=fundamental_stacked_basis_from_shape(config.shape),
    )


def get_localisation_operator(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[
        EvenlySpacedBasis[int, int, int],
        TupleBasisLike[*tuple[FundamentalBasis[int], ...]],
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]],
    ],
) -> LocalizationOperator[
    TupleBasisLike[*tuple[FundamentalBasis[int], ...]],
    FundamentalBasis[int],
    EvenlySpacedBasis[int, int, int],
]:
    return get_localization_operator_wannier90(
        wavefunctions,
        options=Wannier90Options[FundamentalBasis[int]](
            projection={
                "basis": TupleBasis(
                    FundamentalBasis[int](wavefunctions["basis"][0][0].n),
                ),
            },
        ),
    )


def get_hamiltonian(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> SingleBasisOperator[ExplicitStackedBasisWithLength[Any, Any]]:
    wavefunctions = get_wavepacket(system, config)

    if config.type == "bloch":
        return as_operator(get_full_bloch_hamiltonian(wavefunctions))

    operator = get_localisation_operator(wavefunctions)
    return get_full_wannier_hamiltonian(wavefunctions, operator)


def get_noise_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> SingleBasisNoiseKernel[ExplicitStackedBasisWithLength[Any, Any]]:
    hamiltonian = get_hamiltonian(system, config)

    return convert_diagonal_kernel_to_basis(
        get_effective_gaussian_noise_kernel(
            hamiltonian["basis"][0],
            system.eta,
            config.temperature,
        ),
        hamiltonian["basis"],
    )


def get_noise_operators(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    ExplicitStackedBasisWithLength[Any, Any],
]:
    hamiltonian = _get_full_hamiltonian(system, config.shape, config.resolution)
    operators = get_temperature_corrected_effective_gaussian_noise_operators(
        hamiltonian,
        system.eta,
        config.temperature,
    )

    actual_hamiltonian = get_hamiltonian(system, config)
    return convert_noise_operator_list_to_basis(operators, actual_hamiltonian["basis"])


def get_initial_state(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    wavefunctions = get_wavepacket(system, config)
    operator = get_localisation_operator(wavefunctions)
    basis = get_wannier_basis(wavefunctions, operator)
    data = np.zeros(basis.n, dtype=np.complex128)
    data[0] = 1
    return {"basis": basis, "data": data}
