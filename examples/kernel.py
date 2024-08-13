import numpy as np

from reduced_state_caldeira_leggett.plot import (
    plot_chebyshev_fit_time,
    plot_fft_fit_time,
    plot_get_trig_operators_time,
    plot_isotropic_kernel_percentage_error,
    plot_noise_kernel,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(3,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="poly fit",
        n_polynomial=6,
    )
    config1 = SimulationConfig(
        shape=(2,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="fft",
        n_polynomial=10,
    )
    size = np.arange(1, 11)
    n_run = 50
    # add 2d example here

    plot_fft_fit_time(system, config, size, n_run)

    plot_chebyshev_fit_time(system, config, size, n_run)

    plot_get_trig_operators_time(system, config, size, n_run)

    plot_noise_kernel(system, config)
    plot_isotropic_kernel_percentage_error(
        system,
        config,
        base_config=config1,
    )
