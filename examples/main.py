from reduced_state_caldeira_leggett.plot import (
    plot_basis_states,
    plot_initial_state,
    plot_state_against_t,
)
from reduced_state_caldeira_leggett.system import (
    SODIUM_COPPER_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = SODIUM_COPPER_SYSTEM
    config = SimulationConfig(
        shape=(1, 1),
        resolution=(31, 31),
        n_bands=31,
        type="bloch",
        temperature=150,
    )

    plot_basis_states(system, config)
    plot_state_against_t(system, config, n=1000, step=500)
    plot_initial_state(system, config)
