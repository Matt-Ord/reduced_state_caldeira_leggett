from reduced_state_caldeira_leggett.plot import (
    plot_basis_states,
    plot_thermal_occupation,
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
        n_polynomial=3,
    )

    plot_thermal_occupation(system, config)
    plot_basis_states(system, config)
