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
        shape=(2,),
        resolution=(3,),
        n_bands=3,
        type="bloch",
        temperature=150,
        fit_method="explicit polynomial",
        n_polynomial=3,
    )

    plot_noise_kernel(system, config)
    plot_noise_operators(system, config)
