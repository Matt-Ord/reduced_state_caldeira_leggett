from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import Boltzmann, hbar
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
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.kernel.conversion import (
    convert_diagonal_kernel_to_basis,
    convert_noise_operator_list_to_basis,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_noise_operators,
    get_gaussian_noise_kernel,
    get_temperature_corrected_noise_operators,
)
from surface_potential_analysis.kernel.kernel import (
    get_noise_kernel as get_noise_kernel_generic,
)
from surface_potential_analysis.kernel.kernel import (
    get_single_factorized_noise_operators,
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

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


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

    shape: tuple[int]
    resolution: tuple[int]
    n_bands: int
    type: Literal["bloch", "wannier"]


HYDROGEN_NICKEL_SYSTEM = PeriodicSystem(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
    gamma=0.2e12,
)


def get_potential(
    system: PeriodicSystem,
) -> Potential[StackedBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    delta_x = np.sqrt(3) * system.lattice_constant / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": StackedBasis(axis), "data": vector}


def get_interpolated_potential(
    system: PeriodicSystem,
    resolution: tuple[_L0Inv],
) -> Potential[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential(system)
    old = potential["basis"][0]
    basis = StackedBasis(
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
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
) -> Potential[
    StackedBasisLike[
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]]
    ]
]:
    interpolated = get_interpolated_potential(system, resolution)
    old = interpolated["basis"][0]
    basis = StackedBasis(
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]](
            old.delta_x * shape[0],
            n=old.n,
            step=shape[0],
            offset=0,
        ),
    )
    scaled_potential = interpolated["data"] * np.sqrt(basis.fundamental_n / old.n)

    return {"basis": basis, "data": scaled_potential}


def _get_full_hamiltonian(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    *,
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],]:
    bloch_fraction = np.array([0]) if bloch_fraction is None else bloch_fraction

    potential = get_extended_interpolated_potential(system, shape, resolution)
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_position_basis(potential["basis"]),
    )
    return total_surface_hamiltonian(converted, system.mass, bloch_fraction)


def _get_wavepacket(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> BlochWavefunctionListWithEigenvaluesList[
    EvenlySpacedBasis[int, int, int],
    StackedBasisLike[FundamentalBasis[int]],
    StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]],
]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[
        StackedBasisLike[FundamentalPositionBasis[int, Literal[1]]]
    ]:
        return _get_full_hamiltonian(
            system,
            (1,),
            config.resolution,
            bloch_fraction=bloch_fraction,
        )

    StackedBasis(FundamentalBasis)
    return generate_wavepacket(
        hamiltonian_generator,
        save_bands=EvenlySpacedBasis(config.n_bands, 1, 0),
        list_basis=fundamental_stacked_basis_from_shape(config.shape),
    )


def get_hamiltonian(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> SingleBasisOperator[ExplicitStackedBasisWithLength[Any, Any]]:
    wavefunctions = _get_wavepacket(system, config)

    if config.type == "bloch":
        return as_operator(get_full_bloch_hamiltonian(wavefunctions))

    operator = get_localization_operator_wannier90(
        wavefunctions,
        options=Wannier90Options[FundamentalBasis[int]](
            projection={"basis": StackedBasis(FundamentalBasis[int](config.n_bands))},
        ),
    )
    return get_full_wannier_hamiltonian(wavefunctions, operator)


def get_noise_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
    temperature: float,
) -> SingleBasisNoiseKernel[ExplicitStackedBasisWithLength[Any, Any]]:
    hamiltonian = _get_full_hamiltonian(system, config.shape, config.resolution)
    lambda_ = system.lattice_constant / 2
    # mu = A / lambda
    mu = np.sqrt(2 * system.eta * Boltzmann * temperature / hbar**2)
    a = mu * lambda_

    kernel = get_gaussian_noise_kernel(
        hamiltonian["basis"][0],
        a,
        lambda_,
    )

    actual_hamiltonian = get_hamiltonian(system, config)
    converted = convert_diagonal_kernel_to_basis(kernel, actual_hamiltonian["basis"])

    data = (
        converted["data"]
        .reshape(*converted["basis"][0].shape, *converted["basis"][1].shape)
        .swapaxes(0, 1)
        .reshape(converted["basis"][0].n, converted["basis"][1].n)
    )
    data += np.conj(np.transpose(data))
    data /= 2
    converted["data"] = (
        data.reshape(
            converted["basis"][0].shape[1],
            converted["basis"][0].shape[0],
            *converted["basis"][1].shape,
        )
        .swapaxes(0, 1)
        .ravel()
    )
    operators = get_single_factorized_noise_operators(converted)
    corrected = get_temperature_corrected_noise_operators(
        actual_hamiltonian,
        operators,
        temperature,
    )
    corrected_kernel = get_noise_kernel_generic(corrected)
    data = (
        corrected_kernel["data"]
        .reshape(
            *corrected_kernel["basis"][0].shape,
            *corrected_kernel["basis"][1].shape,
        )
        .swapaxes(0, 1)
        .reshape(corrected_kernel["basis"][0].n, corrected_kernel["basis"][1].n)
    )
    data += np.conj(np.transpose(data))
    data /= 2
    corrected_kernel["data"] = (
        data.reshape(
            corrected_kernel["basis"][0].shape[1],
            corrected_kernel["basis"][0].shape[0],
            *corrected_kernel["basis"][1].shape,
        )
        .swapaxes(0, 1)
        .ravel()
    )
    return corrected_kernel


def get_noise_operators(
    system: PeriodicSystem,
    config: SimulationConfig,
    temperature: float,
) -> SingleBasisNoiseOperatorList[
    FundamentalBasis[int],
    ExplicitStackedBasisWithLength[Any, Any],
]:
    hamiltonian = _get_full_hamiltonian(system, config.shape, config.resolution)
    operators = get_effective_gaussian_noise_operators(
        hamiltonian,
        system.eta,
        temperature,
    )

    actual_hamiltonian = get_hamiltonian(system, config)
    return convert_noise_operator_list_to_basis(operators, actual_hamiltonian["basis"])
