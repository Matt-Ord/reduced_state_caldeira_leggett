from reduced_state_caldeira_leggett.plot import (
    plot_noise_kernel,
    plot_noise_operators,
    try_plot_2d_kernel,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(3, 3),
        resolution=(41, 31),
        n_bands=9,
        type="bloch",
        temperature=150,
        fit_method="fitted polynomial",
        n_polynomial=(75, 51),
    )
    try_plot_2d_kernel(system, config)

    for i in range(9):
        plot_noise_operators(system, config, idx=i)

    plot_noise_kernel(system, config)
