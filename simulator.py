import numpy as np
from plot_helper import Plotter
from tqdm import tqdm

class Parameter:
    def __init__(self, name, min_val=1, max_val=1, step=None, values=None):
        self.name = name
        if values is not None:
            self.values = np.array(values)
        elif step is not None and min_val != max_val:
            self.values = np.arange(min_val, max_val + step, step)
        else:
            raise ValueError("Invalid combination of min/max and step.")
    
    def __iter__(self):
        return iter(self.values)
    
    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    @property
    def step_scale(self):
        if len(self.values) < 2:
            return 1.0  # fallback
        return (self.values.max() - self.values.min()) / len(self.values)

    def min(self):
        return self.values.min()
    
    def max(self):
        return self.values.max()

class Player:
    def __init__(self):
        self.rating = 0
    
    def step_trial(self):
        raise NotImplementedError()

class Simulator:
    Plotter = Plotter()

    def __init__(self, PlayerClass, n_trials=10000, n_games=200):
        self.PlayerClass= PlayerClass
        self.n_trials= n_trials
        self.n_games= n_games
    
    def simulate(self, x_param, y_param):
        results_matrix = np.zeros((len(x_param), len(y_param)), dtype=object)

        for i, yv in enumerate(tqdm(x_param, desc=f"Sweeping {x_param.name}")):
            for j, xv in enumerate(tqdm(y_param, desc=f"Sweeping {y_param.name}", leave=False)):
                results_matrix[i, j] = self._simulate_one(delta_wr=yv, p_troll=xv)

        self.Plotter.plot(results_matrix, x_param, y_param)

    def _simulate_one(self, **player_kwargs):
        outcomes_final = np.zeros(self.n_trials)
        for trial_idx in range(self.n_trials):
            player = self.PlayerClass(**player_kwargs)
            for _ in range(self.n_games):
                player.step_trial()
            outcomes_final[trial_idx] = player.rating
        return Simulator.summarize(outcomes_final)
    
    @staticmethod
    def summarize(outcomes_final):
        return {
            "q25": np.percentile(outcomes_final, 25),
            "q75": np.percentile(outcomes_final, 75),
            "q05": np.percentile(outcomes_final, 5),
            "q95": np.percentile(outcomes_final, 95),
        }