from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
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
    get_effective_gaussian_parameters,
    get_gaussian_isotropic_noise_kernel,
    get_temperature_corrected_effective_gaussian_noise_operators,
)
from surface_potential_analysis.kernel.kernel import get_coefficient_matrix_taylor
from surface_potential_analysis.operator.operator import as_operator
from surface_potential_analysis.potential.conversion import convert_potential_to_basis
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.util.interpolation import pad_ft_points
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
        IsotropicNoiseKernel,
        SingleBasisDiagonalNoiseOperatorList,
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
                old.delta_x,
                old.n,
                r,
            )
            for (old, r) in zip(potential["basis"], resolution)
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
                old.delta_x * s,
                n=old.n,
                step=s,
                offset=0,
            )
            for (old, s) in zip(potential["basis"], shape)
        ),
    )

    scaled_potential = potential["data"] * np.sqrt(
        extrapolated_basis.fundamental_n / potential["basis"].n,
    )

    return {"basis": extrapolated_basis, "data": scaled_potential}


def get_potential_1d(
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


def get_potential_2d(
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
            potential = get_potential_1d(
                system,
                cast(tuple[int], shape),
                cast(tuple[int], resolution),
            )
        case 2:
            potential = get_potential_2d(
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
        TupleBasisLike[FundamentalBasis[int]],
        TupleBasisWithLengthLike[FundamentalPositionBasis[int, Literal[1]]],
    ],
) -> LocalizationOperator[
    TupleBasisLike[FundamentalBasis[int]],
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


def new_noise_operators(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    n: int = 1,
    lambda_factor: float = 2 * np.sqrt(2),
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    """To fit the noise correlation using trig functions generated from hamiltonian,
    expressed using a truncated trig series include only the first n sine/cos terms,
    found using fft method.

    Return in the order of [const term, first n sine terms, first n cos terms]
    and also their corresponding coefficients.

    """
    hamiltonian = _get_full_hamiltonian(system, config.shape, config.resolution)
    a, lambda_ = get_effective_gaussian_parameters(
        hamiltonian["basis"][0],
        system.eta,
        config.temperature,
        lambda_factor=lambda_factor,
    )
    basis_x = stacked_basis_as_fundamental_position_basis(hamiltonian["basis"][0])
    k = 2 * np.pi / basis_x.shape[0]
    nx_points = BasisUtil(basis_x).fundamental_stacked_nx_points[0]

    sines = [
        np.sin(i * k * nx_points).astype(np.complex128) for i in np.arange(1, n + 1)
    ]
    coses = [
        np.cos(i * k * nx_points).astype(np.complex128) for i in np.arange(1, n + 1)
    ]
    data = np.append(np.ones_like(nx_points).astype(np.complex128), [sines, coses])
    # get coeff
    correlation = get_gaussian_isotropic_noise_kernel(basis_x, a, lambda_)
    peak = np.concatenate(
        (correlation["data"][(n + 1) :], correlation["data"][: (n + 1)]),
    )
    ft_peak = np.fft.rfft(peak, norm="forward")
    zero_freq = np.array([ft_peak[0]])
    coeff = np.concatenate(
        [
            zero_freq,
            (ft_peak[1:]) / 2,
            (-ft_peak[1:][::-1]) / 2,
        ],
    )

    return {
        "basis": TupleBasis(FundamentalBasis(2 * n + 1), TupleBasis(basis_x, basis_x)),
        "data": data.astype(np.complex128),
        "eigenvalue": coeff.astype(np.complex128),
    }


_B0 = TypeVar("_B0", bound=TupleBasisWithLengthLike[Any, Any])


def get_noise_operators_taylor_expansion(
    kernel: IsotropicNoiseKernel[_B0],
    *,
    n: int = 1,
) -> SingleBasisDiagonalNoiseOperatorList[
    FundamentalBasis[int],
    TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]],
]:
    """Calculate the noise operators for a general isotropic noise kernel, using
    an explicit Taylor expansion.

    Parameters
    ----------
    kernel: IsotropicNoiseKernel[TupleBasisWithLengthLike[Any, Any]]
    n: int, by default 1

    Returns
    -------
    The noise operators formed using the2n+1 lowest fourier terms, and the corresponding coefficients.

    """
    basis_x = stacked_basis_as_fundamental_position_basis(kernel["basis"])
    delta_x = np.linalg.norm(BasisUtil(basis_x).delta_x_stacked[0])
    k = 2 * np.pi / basis_x.shape[0]
    delta_k = 2 * np.pi / delta_x
    nx_points = BasisUtil(basis_x).fundamental_stacked_nx_points[0]
    displacements = (
        (BasisUtil(basis_x).fundamental_stacked_nk_points[0])
        * (BasisUtil(basis_x).dx_stacked[0])
    )
    kernel_data = kernel["data"]
    # weight is chosen such that the 2n+1 points around the origin are selected for fitting
    weight = pad_ft_points(np.ones(2 * n + 1), (basis_x.n,), (0,))
    noise_polynomial = np.polynomial.Polynomial.fit(
        x=displacements,
        y=kernel_data,
        deg=np.arange(0, 2 * n + 1, 2),
        w=weight,
        domain=[-np.min(displacements), np.min(displacements)],
    )
    noise_coefficients = noise_polynomial.convert().coef[::2]
    operator_coefficients = get_coefficient_matrix_taylor(
        true_noise_coeff=noise_coefficients,
        delta_k=delta_k,
        n=n,
    )

    sines = [
        np.sin(i * k * nx_points).astype(np.complex128) for i in np.arange(1, n + 1)
    ]
    coses = [
        np.cos(i * k * nx_points).astype(np.complex128) for i in np.arange(1, n + 1)
    ]
    data = np.append(np.ones_like(nx_points).astype(np.complex128), [sines, coses])

    return {
        "basis": TupleBasis(FundamentalBasis(2 * n + 1), TupleBasis(basis_x, basis_x)),
        "data": data.astype(np.complex128),
        "eigenvalue": operator_coefficients.astype(np.complex128),
    }
