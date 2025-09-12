import numpy as np
from tqdm import tqdm

from plot_helper import *

class Player:
    def __init__(self):
        self.rating = 0
    
    def step_trial(self):
        raise NotImplementedError()

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

def make_league_player_factory_factory(delta_wr: float, p_troll: float):
    def impl():
        return LeaguePlayer(delta_wr, p_troll)
    return impl 

def simulate_games(
    n_runs=10000,
    n_games=200,
    player_factory=make_league_player_factory_factory(.025, .05) 
) -> "tuple[float, float, float, float, float]":
    """
    Simulates n_runs players over n_games and returns
    (mean, q25, q75, q05, q95).
    """
    ratings_final = np.zeros(n_runs)

    for run in range(n_runs):
        player = player_factory()
        for g in range(n_games):
            player.step_trial()
        ratings_final[run] = player.rating

    return (
        ratings_final.mean(),
        np.percentile(ratings_final, 25),
        np.percentile(ratings_final, 75),
        np.percentile(ratings_final, 5),
        np.percentile(ratings_final, 95),
    )

# Sweep ranges
p_troll_vals = np.linspace(0.0, 1, 10)
delta_wr_vals = np.linspace(0.0, 0.05, 10)

results_matrix = np.zeros((len(delta_wr_vals), len(p_troll_vals)), dtype=object)

# Progress bar over grid
for i, dwr in enumerate(tqdm(delta_wr_vals, desc="Sweeping Δ_wr")):
    for j, pt in enumerate(tqdm(p_troll_vals, desc="Sweeping p_troll", leave=False)):
        results_matrix[i, j] = simulate_games(
            n_runs=100,
            n_games=200,
            player_factory=make_league_player_factory_factory(dwr, pt)
        )

means = np.array([results_matrix[i, j][0] for i in range(len(delta_wr_vals)) for j in range(len(p_troll_vals))])
q25s  = np.array([results_matrix[i, j][1] for i in range(len(delta_wr_vals)) for j in range(len(p_troll_vals))])
q75s  = np.array([results_matrix[i, j][2] for i in range(len(delta_wr_vals)) for j in range(len(p_troll_vals))])
q05s  = np.array([results_matrix[i, j][3] for i in range(len(delta_wr_vals)) for j in range(len(p_troll_vals))])
q95s  = np.array([results_matrix[i, j][4] for i in range(len(delta_wr_vals)) for j in range(len(p_troll_vals))])

# Bar dimensions
bar_sx = 1/10
bar_sy = 0.05/10

X, Y = np.meshgrid(p_troll_vals, delta_wr_vals)
X, Y = (X.flatten(), Y.flatten())

plot_with_mesh(X, Y, q25s, q75s, q05s, q95s, bar_sx=0.1, bar_sy=0.005)
