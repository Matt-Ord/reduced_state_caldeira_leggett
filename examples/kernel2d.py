from reduced_state_caldeira_leggett.plot import (
    plot_noise_kernel,
    plot_noise_kernel_error_1d,
    plot_noise_operators,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(1, 1),
        resolution=(3, 3),
        n_bands=9,
        type="bloch",
        temperature=150,
        fit_method="fitted polynomial",
        n_polynomial=(3, 4),
    )
    fig, _, _ = plot_noise_kernel_error_1d(system, config)
    fig.show()
    input()

    plot_noise_kernel(system, config)

    for i in range(9):
        plot_noise_operators(system, config, idx=i)
