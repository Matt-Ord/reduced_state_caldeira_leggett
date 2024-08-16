from reduced_state_caldeira_leggett.plot import (
    plot_basis_states,
    plot_initial_state,
    plot_state_against_t,
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
        fit_method="fft",
        n_polynomial=9,
    )

    plot_basis_states(system, config)
    plot_state_against_t(system, config, n=1000, step=500)
    plot_initial_state(system, config)
