from reduced_state_caldeira_leggett.plot import plot_isotropic_noise_kernel
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(1, 1),
        resolution=(31, 31),
        n_bands=31,
        type="bloch",
        temperature=150,
    )
    plot_isotropic_noise_kernel(system, config)
