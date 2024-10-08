from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis,
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
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.kernel.build import (
    get_temperature_corrected_diagonal_noise_operators,
    truncate_diagonal_noise_operator_list,
)
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_parameters,
    get_gaussian_isotropic_noise_kernel,
    get_gaussian_operators_explicit_taylor_stacked,
)
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    as_diagonal_kernel_from_isotropic,
    get_isotropic_kernel_from_diagonal_operators_stacked,
)
from surface_potential_analysis.kernel.solve import (
    get_noise_operators_diagonal_eigenvalue,
    get_noise_operators_real_isotropic_stacked_fft,
)
from surface_potential_analysis.kernel.solve._taylor import (
    get_noise_operators_real_isotropic_stacked_taylor_expansion,
)
from surface_potential_analysis.operator.operator import as_operator
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_basis,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_transformed_stacked_basis_from_shape,
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
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.kernel.kernel import (
        SingleBasisDiagonalNoiseOperatorList,
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


FitMethod = Literal[
    "fitted polynomial",
    "eigenvalue",
    "fft",
    "explicit polynomial",
]


@dataclass
class SimulationConfig:
    """Configure the detail of the simulation."""

    shape: tuple[int, ...]
    resolution: tuple[int, ...]
    n_bands: int
    type: Literal["bloch", "wannier"]
    temperature: float
    fit_method: FitMethod = "fft"
    n_polynomial: tuple[int, ...] | None = None


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


def _get_fundamental_potential_1d(
    system: PeriodicSystem,
) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    """Generate potential for a periodic 1D system."""
    delta_x = np.sqrt(3) * system.lattice_constant / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": TupleBasis(axis), "data": vector}


def _get_interpolated_potential(
    potential: Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ],
    resolution: tuple[_L0Inv, ...],
) -> Potential[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
    interpolated_basis = TupleBasis(
        *tuple(
            TransformedPositionBasis[Any, Any, Any](
                potential["basis"][i].delta_x,
                potential["basis"][i].n,
                r,
            )
            for (i, r) in enumerate(resolution)
        ),
    )

    scaled_potential = potential["data"] * np.sqrt(
        interpolated_basis.fundamental_n / potential["basis"].n,
    )

    return convert_potential_to_basis(
        {"basis": interpolated_basis, "data": scaled_potential},
        stacked_basis_as_fundamental_momentum_basis(interpolated_basis),
    )


def _get_extrapolated_potential(
    potential: Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ],
    shape: tuple[_L0Inv, ...],
) -> Potential[
    TupleBasisWithLengthLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
]:
    extrapolated_basis = TupleBasis(
        *tuple(
            EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any](
                potential["basis"][i].delta_x * s,
                n=potential["basis"][i].n,
                step=s,
                offset=0,
            )
            for (i, s) in enumerate(shape)
        ),
    )

    scaled_potential = potential["data"] * np.sqrt(
        extrapolated_basis.fundamental_n / potential["basis"].n,
    )

    return {"basis": extrapolated_basis, "data": scaled_potential}


def _get_potential_1d(
    system: PeriodicSystem,
    shape: tuple[int, ...],
    resolution: tuple[int, ...],
) -> Potential[
    TupleBasisWithLengthLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
]:
    potential = _get_fundamental_potential_1d(system)
    interpolated = _get_interpolated_potential(potential, resolution)

    return _get_extrapolated_potential(interpolated, shape)


def _get_fundamental_potential_2d(
    system: PeriodicSystem,
) -> Potential[
    TupleBasis[
        FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
        FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
    ]
]:
    # We want the simplest possible potential in 2d with symmetry
    # (x0,x1) -> (x1,x0)
    # (x0,x1) -> (-x0,x1)
    # (x0,x1) -> (x0,-x1)
    # We therefore occupy G = +-K0, +-K1, +-(K0+K1) equally
    data = [[0, 1, 1], [1, 1, 0], [1, 0, 1]]
    vector = 0.5 * system.barrier_energy * np.array(data) / np.sqrt(9)
    return {
        "basis": TupleBasis(
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]](
                system.lattice_constant * np.array([0, 1]),
                3,
            ),
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]](
                system.lattice_constant
                * np.array(
                    [np.sin(np.pi / 3), np.cos(np.pi / 3)],
                ),
                3,
            ),
        ),
        "data": vector.ravel(),
    }


def _get_potential_2d(
    system: PeriodicSystem,
    shape: tuple[_L0Inv, ...],
    resolution: tuple[int, ...],
) -> Potential[
    TupleBasisWithLengthLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
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
    potential = _get_fundamental_potential_2d(system)
    interpolated = _get_interpolated_potential(potential, resolution)
    return _get_extrapolated_potential(interpolated, shape)


def get_potential(
    system: PeriodicSystem,
    shape: tuple[int, ...],
    resolution: tuple[int, ...],
) -> Potential[
    TupleBasisWithLengthLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
]:
    match len(shape):
        case 1:
            return _get_potential_1d(
                system,
                cast(tuple[int], shape),
                cast(tuple[int], resolution),
            )
        case 2:
            return _get_potential_2d(
                system,
                cast(tuple[int, int], shape),
                cast(tuple[int, int], resolution),
            )
        case _:
            msg = "Currently only support 1 and 2D potentials"
            raise ValueError(msg)


def _get_basis(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> TupleBasisWithLengthLike[
    *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
]:
    return get_potential(system, config.shape, config.resolution)["basis"]


def _get_bloch_hamiltonian(
    system: PeriodicSystem,
    shape: tuple[int, ...],
    resolution: tuple[int, ...],
    *,
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]],
]:
    bloch_fraction = np.array([0]) if bloch_fraction is None else bloch_fraction

    potential = get_potential(system, shape, resolution)

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
    TupleBasisLike[*tuple[FundamentalTransformedBasis[int], ...]],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]],
]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]]
    ]:
        return _get_bloch_hamiltonian(
            system,
            tuple(1 for _ in config.shape),
            config.resolution,
            bloch_fraction=bloch_fraction,
        )

    return generate_wavepacket(
        hamiltonian_generator,
        band_basis=EvenlySpacedBasis(config.n_bands, 1, 0),
        list_basis=fundamental_transformed_stacked_basis_from_shape(config.shape),
    )


def get_localisation_operator(
    wavefunctions: BlochWavefunctionListWithEigenvaluesList[
        EvenlySpacedBasis[int, int, int],
        TupleBasisLike[*tuple[FundamentalTransformedBasis[int], ...]],
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]],
    ],
) -> LocalizationOperator[
    TupleBasisLike[*tuple[FundamentalTransformedBasis[int], ...]],
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


def get_true_noise_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> IsotropicNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    basis = _get_basis(system, config)

    a, lambda_ = get_effective_gaussian_parameters(
        basis,
        system.eta,
        config.temperature,
    )
    return get_gaussian_isotropic_noise_kernel(basis, a, lambda_)


def get_noise_operators(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> SingleBasisDiagonalNoiseOperatorList[
    BasisLike[Any, Any],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
]:
    kernel = get_true_noise_kernel(system, config)
    if config.fit_method == "explicit polynomial":
        basis = _get_basis(system, config)

        a, lambda_ = get_effective_gaussian_parameters(
            basis,
            system.eta,
            config.temperature,
        )
        return get_gaussian_operators_explicit_taylor_stacked(
            basis,
            a,
            lambda_,
            shape=config.n_polynomial,
        )
    match config.fit_method:
        case "fitted polynomial":
            return get_noise_operators_real_isotropic_stacked_taylor_expansion(
                kernel,
                shape=config.n_polynomial,
            )
        case "fft":
            operators = get_noise_operators_real_isotropic_stacked_fft(
                kernel,
            )

            if config.n_polynomial is None:
                return operators
            return truncate_diagonal_noise_operator_list(
                operators,
                range(2 * config.n_polynomial[0] + 1),
            )

        case "eigenvalue":
            operators = get_noise_operators_diagonal_eigenvalue(
                as_diagonal_kernel_from_isotropic(kernel),
            )

            if config.n_polynomial is None:
                return operators
            return truncate_diagonal_noise_operator_list(
                operators,
                range(2 * config.n_polynomial[0] + 1),
            )


def get_noise_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> IsotropicNoiseKernel[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    operators = get_noise_operators(system, config)
    return get_isotropic_kernel_from_diagonal_operators_stacked(operators)


def get_temperature_corrected_noise_operators(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> SingleBasisNoiseOperatorList[
    BasisLike[Any, Any],
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]],
]:
    operators = get_noise_operators(system, config)
    hamiltonian = get_hamiltonian(system, config)
    return get_temperature_corrected_diagonal_noise_operators(
        hamiltonian,
        operators,
        config.temperature,
    )


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
