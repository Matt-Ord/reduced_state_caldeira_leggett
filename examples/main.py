from reduced_state_caldeira_leggett.plot import (
    plot_2d_111_potential,
    plot_2d_state_111_against_t,
    plot_basis_states,
    plot_initial_state,
    plot_state_against_t,
)
from reduced_state_caldeira_leggett.system import (
    FREE_LITHIUM_SYSTEM,
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
    get_2d_111_potential,
)

system = HYDROGEN_NICKEL_SYSTEM
config = SimulationConfig(shape=(1, 1), resolution=(60, 50), n_bands=3, type="bloch")
test = get_2d_111_potential(system, config.resolution)
plot_2d_111_potential(test)
plot_2d_state_111_against_t(system, config, n=20, step=10)


if __name__ == "__main__":
    system = FREE_LITHIUM_SYSTEM
    config = SimulationConfig(shape=(2,), resolution=(31,), n_bands=3, type="bloch")

    plot_basis_states(system, config)
    plot_state_against_t(system, config, n=1000, step=500)
    plot_initial_state(system, config)
