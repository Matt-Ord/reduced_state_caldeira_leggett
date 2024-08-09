import numpy as np

from reduced_state_caldeira_leggett.plot import (
    plot_kernel_fit_runtime,
    plot_noise_kernel,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(2,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="fft",
        n_polynomial=20,
    )
    config1 = SimulationConfig(
        shape=(3,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="fft",
        n_polynomial=5,
    )
    size = np.array([(3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)])
    n_run = 1000
    # add 2d example here

    plot_noise_kernel(system, config)
    plot_kernel_fit_runtime(system, config, size, n_run)
