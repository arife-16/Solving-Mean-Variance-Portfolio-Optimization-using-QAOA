# Smart QAOA short example by ⚛️ Sigma PublishinQ Team ⚛️  
# https://www.linkedin.com/company/sigma-publishinq/about/

import numpy as np
import networkx as nx
from typing import List, Optional, Callable, Dict, Tuple
from scipy.optimize import minimize
import copy
import sys
import os
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/QOKit/')
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN/short_smart_qaoa_example/')
from qokit.qaoa_objective_maxcut import get_qaoa_maxcut_objective
from erdos_renyi_fabric import ErdosRenyiFabric


class QAOAMaxCutEngine:
    """
    QAOA for solving MaxCut problem using QOKit.
    """

    def __init__(self, Graph: ErdosRenyiFabric = None, reps: int = 1) -> None:
        self._G = Graph
        self.p = reps
        self._qaoa_objective = get_qaoa_maxcut_objective(
            N=self._G.number_of_nodes(),
            p=self.p,
            G=self._G,
            simulator="auto"
        )

    def optimize(self,
                 optimizer: str = 'BFGS',
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 maxiter: int = 1000,
                 tol: Optional[float] = None,
                 callback: Optional[Callable] = None) -> Dict:
        """QAOA parameter optimization."""
        try:
            self._init_params
        except AttributeError:
            raise AttributeError('Run determine_initial_point method to determine initial point.')

        result = minimize(
            fun=self._qaoa_objective,
            x0=copy.copy(self._init_params),
            method=optimizer,
            bounds=bounds,
            tol=tol,
            callback=callback,
            options={'maxiter': maxiter, 'disp': True}
        )
        return result

    def _qaoa_cost(self, beta: np.ndarray, gamma: np.ndarray) -> float:
        """Computes QAOA cost for given parameters."""
        params = np.concatenate([gamma, beta])
        return self._qaoa_objective(params)

    def determine_initial_point(self, given_params=None) -> np.ndarray:
        """
        Determines initial parameters for QAOA.
        Uses TQA algorithm if parameters are not given.
        """
        if given_params is None:
            self._init_params = self._tqa_beta_gamma(self._tqa_dt())
        elif isinstance(given_params, float):
            self._init_params = self._tqa_beta_gamma(given_params)
        else:
            try:
                if len(given_params) == 2 * self.p:
                    self._init_params = given_params
                else:
                    self._init_params = self._tqa_beta_gamma(self._tqa_dt())
            except:
                self._init_params = self._tqa_beta_gamma(self._tqa_dt())

    def _tqa_dt(self, time_1: float = 0.35, time_end: float = 1.4, time_steps_num: int = 50) -> float:
        """TQA algorithm for finding optimal dt."""
        dt_list = np.linspace(time_1, time_end, time_steps_num)
        cut_value = 0
        best_dt = time_1
        
        for dt in dt_list:
            params = self._tqa_beta_gamma(dt)
            beta, gamma = params[:self.p], params[self.p:]
            cost = self._qaoa_cost(beta, gamma)
            
            if cost <= cut_value:
                cut_value = cost
                best_dt = dt
        
        return best_dt

    def _tqa_beta_gamma(self, best_dt: float) -> np.ndarray:
        """Computes beta and gamma parameters from best dt."""
        i_list = np.arange(1, self.p + 1) - 0.5
        gamma = (i_list / self.p) * best_dt
        beta = (1 - (i_list / self.p)) * best_dt
        return np.concatenate([gamma, beta])

    @property
    def reps(self):
        return self.p

    @property
    def initial_point(self):
        try:
            return self._init_params
        except AttributeError:
            raise AttributeError('Run determine_initial_point method to determine initial point.')

    def evaluate_energy(self, params: np.ndarray) -> float:
        """Computes energy for given parameters."""
        return self._qaoa_objective(params)