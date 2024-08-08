from reduced_state_caldeira_leggett.plot import (
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
        FitMethod="poly fit",
        n_polynomial=11,
    )
    config1 = SimulationConfig(
        shape=(3,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        FitMethod="fft",
        n_polynomial=5,
    )

    plot_noise_kernel(system, config)
    # plot_isotropic_kernel_percentage_error(system, config, config1=config1)
