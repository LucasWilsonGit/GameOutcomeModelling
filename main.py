import numpy as np


from simulator import *
from plot_helper import *

class LeaguePlayer:
    base_wr=0.53
    delta_troll=0.05
    points_per_win=20
    points_per_loss=-20 

    def __init__(self, delta_wr=.01, p_troll=.01):
        self.effective_wr = LeaguePlayer.base_wr
        self.rating = 0
        self.delta_wr = delta_wr
        self.p_troll = p_troll

    def process_match_outcome(self, won: bool = False, had_troll: bool = False) -> None:
        if won:
            self.rating += LeaguePlayer.points_per_win
            self.effective_wr = LeaguePlayer.base_wr
        else:
            self.rating += LeaguePlayer.points_per_loss 
            self.effective_wr = max(0, self.effective_wr - (self.delta_wr + (LeaguePlayer.delta_troll if had_troll else 0)))
    
    def step_trial(self) -> None:
        rng = np.random.rand()
        
        has_troll: bool = rng < self.p_troll
        is_win: bool = rng < self.effective_wr
        
        self.process_match_outcome(is_win, has_troll)

class LeagueSimulator(Simulator):
    n_trials = 1000
    n_games = 200
    Plotter = Candlestick3DPlotter()

    def __init__(self):
        super().__init__(LeaguePlayer, n_trials=self.n_trials, n_games=self.n_games)

# Sweep parameter definitions
delta_wr_param = Parameter("delta_wr", values = np.linspace(0.0, 0.05, 10))
p_troll_param = Parameter("p_troll", values = np.linspace(0.0, .5, 10))

# Sweep with progress bar
simulator = LeagueSimulator()
simulator.simulate(delta_wr_param, p_troll_param)