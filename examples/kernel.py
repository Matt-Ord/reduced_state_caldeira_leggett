import numpy as np

from reduced_state_caldeira_leggett.plot import (
    plot_isotropic_kernel_percentage_error,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(5,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="poly fit",
        n_polynomial=75,
    )
    config1 = SimulationConfig(
        shape=(5,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="fft",
        n_polynomial=10,
    )
    size = np.array([(3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)])
    n_run = 1000
    # add 2d example here

    # plot_noise_kernel(system, config)
    # plot_kernel_fit_runtime(system, config, size, n_run)
    plot_isotropic_kernel_percentage_error(
        system,
        config,
        base_config=config1,
    )
