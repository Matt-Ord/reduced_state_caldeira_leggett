import numpy as np

from reduced_state_caldeira_leggett.plot import (
    plot_noise_kernel,
    plot_noise_kernel_error_comparison,
    plot_noise_operators,
    plot_operators_fit_time_against_n_polynomial,
    plot_operators_fit_time_against_number_of_states,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(2,),
        resolution=(51,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="fitted polynomial",
        n_polynomial=(6,),
    )

    plot_noise_kernel(system, config)
    plot_noise_operators(system, config, idx=1)

    config1 = SimulationConfig(
        shape=config.shape,
        resolution=config.resolution,
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="fft",
        n_polynomial=(10,),
    )
    plot_noise_kernel_error_comparison(
        system,
        [config, config1],
    )

    plot_operators_fit_time_against_n_polynomial(
        system,
        fit_method="fitted polynomial",
        n_polynomials=np.arange(10, 400, 40),
        n_run=10,
    )

    plot_operators_fit_time_against_number_of_states(
        system,
        fit_method="fft",
        n_states=(np.arange(10, 110, 10),),
        n_run=10,
    )
