from reduced_state_caldeira_leggett.plot import (
    plot_noise_kernel,
    plot_noise_operators,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(3, 3),
        resolution=(4, 3),
        n_bands=9,
        type="bloch",
        temperature=150,
        fit_method="fft",
        n_polynomial=(3, 4),
    )

    plot_noise_kernel(system, config)

    print("here")

    for i in range(9):
        plot_noise_operators(system, config, idx=i)

    print("here again")
