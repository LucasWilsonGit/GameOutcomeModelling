import numpy as np

from GameSimulator.Core.simulator import Parameter
from GameSimulator.LeagueOfLegends.league import LeagueSimulator


if __name__ == "__main__":
    # Sweep parameter definitions
    delta_wr_param = Parameter("delta_wr", values = np.linspace(0.0, 0.05, 10))
    p_troll_param = Parameter("p_troll", values = np.linspace(0.0, .5, 10))

    # Sweep with progress bar
    simulator = LeagueSimulator()
    simulator.simulate(delta_wr_param, p_troll_param)