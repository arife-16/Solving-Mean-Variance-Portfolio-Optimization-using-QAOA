import numpy as np
import pickle
import sys
import networkx as nx
from interconnect import route_request
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import os


class RQAOA_agent:
    """RQAOA agent with gradient optimization training."""
    
    def __init__(self, n: int, nc: int, G: nx.Graph, d: int, learning_rates: List[float], 
                 gamma: float, init_beta: List, init_angles: List, batch_size: int, 
                 normalize: bool, reward_fct: str, pickle_path: Optional[str] = None, 
                 run: str = '0', idx: str = '0', G_name: str = ''):
        
        self.n = n
        self.nc = nc
        self.G = G
        self.G_name = G_name
        self.d = d
        self.learning_rates = learning_rates
        self.gamma = gamma
        self.init_beta = init_beta
        self.init_angles = init_angles
        self.batch_size = batch_size
        self.normalize = normalize
        self.reward_fct = reward_fct
        self.run = run
        self.pickle_path = pickle_path or self._generate_pickle_path(idx)
        
        self._initialize_parameters()
        optimizer_result = route_request('adamwoptimizer', 'create_optimizer', {
            'method_params': {'params': [self.all_angles, self.betas], 'learning_rate_init': learning_rates, 'amsgrad': True}
        })
        self.optimizer = optimizer_result['data']
        self.history_rewards = []
        self.batch_gradients = {'xy': [], 'beta': []}
        self.batch_returns = []
    
    def _initialize_parameters(self) -> None:
        """Agent parameters initialization."""
        beta_type, beta_val = self.init_beta
        self.betas = {
            "all": lambda: beta_val * np.ones(self.n - self.nc),
            "one": lambda: np.array([beta_val], dtype=float),
            "one-all": lambda: beta_val * np.ones(int((self.n ** 2 - self.n) / 2)),
            "all-all": lambda: beta_val * np.ones((self.n - self.nc) * int((self.n ** 2 - self.n) / 2))
        }[beta_type]()
        
        angle_type = self.init_angles[0]
        if angle_type == "rqaoa":
            self._initialize_with_rqaoa()
        elif angle_type == "zero":
            self.all_angles = np.random.randn(2 * (self.n - self.nc)) * self.init_angles[1]
            self.ref = None
        elif angle_type == "xtrm":
            self.all_angles = np.array([0.21991149, 0.39443473] * (self.n - self.nc))
            self.ref = None
        else:
            raise ValueError(f"Unsupported angle initialization type: {angle_type}")
    
    def _initialize_with_rqaoa(self) -> None:
        """Initialize angles using classical RQAOA."""
        if self.init_angles[2] is None:
            batch_data = [(lambda i=i: (print(f"RQAOA initialization {i + 1}/{self.init_angles[1]}"), 
                          self._rqaoa_solve(), print(f"Energy: {self._rqaoa_solve()[1]}"))[-2])() 
                         for i in range(self.init_angles[1])]
            best_idx = np.argmax([data[1] for data in batch_data])
            self.all_angles = np.array(batch_data[best_idx][0], dtype=float)
            self.ref = (batch_data[best_idx][1], batch_data[best_idx][2], [data[1] for data in batch_data])
        else:
            self.all_angles, self.ref = pickle.load(open(self.init_angles[2], 'rb'))
    
    def _rqaoa_solve(self):
        """Solve RQAOA through core API."""
        result = route_request('rqaoaalgorithms', 'rqaoa', {
            'method_params': {'n': self.n, 'nc': self.nc, 'graph': self.G}
        })
        return result['data']
    
    def _generate_pickle_path(self, idx: str) -> str:
        """Generate path for agent saving."""
        return (f'./trained_agents/run{self.run}/rqaoa_agent{self.n}-{self.nc}_{self.G_name}'
                f'_lr{self.learning_rates[0]}-{self.learning_rates[1]}_gamma{self.gamma}'
                f'_rwd-{self.reward_fct}_batchsize{self.batch_size}'
                f'_beta-{self.init_beta[0]}-{self.init_beta[1]}'
                f'_ang-{self.init_angles[0]}-{self.init_angles[1]}'
                f'{"_norm" if self.normalize else ""}_{idx}.pckl')
    
    def _compute_policy(self, action_space: List, abs_expectations: np.ndarray, step: int) -> Tuple[np.ndarray, Any]:
        """Compute action selection policy."""
        print(f"Computing policy, step {step + 1}, action_space: {action_space}, abs_expectations: {abs_expectations}")
        if self.init_beta[0] in ["one-all", "all-all"]:
            betas_data = [(self._get_idx_beta(e), self.betas[self._get_idx_beta(e)]) for e in action_space]
            betas = [b[1] for b in betas_data]
            return self._softmax(abs_expectations * betas, 1.), (betas, [b[0] for b in betas_data])
        else:
            beta = self.betas[0] if self.init_beta[0] == "one" else self.betas[step]
            return self._softmax(abs_expectations, beta), beta
    
    def _get_idx_beta(self, edge):
        """Get beta index for edge."""
        i, j = edge
        if j < i:
            i, j = j, i
        return int(i * self.n - i * (i + 1) / 2 + j - i - 1)
    
    def _softmax(self, values, beta):
        """Compute softmax function."""
        vals = np.array(values)
        vals -= np.amax(vals)
        vals = np.exp(beta * vals)
        return vals / np.sum(vals)
    
    def _execute_step(self, step: int, f_s: List, action_space: List, nodelist: np.ndarray, 
                     compute_gradients: bool = True) -> Dict[str, Any]:
        """Execute one training/evaluation step."""
        print(f"Starting step {step + 1}/{self.n - self.nc}")
        angles = self.all_angles[2 * step:2 * step + 2]
        print(f"Angles for step: {angles}")
        
        try:
            result = route_request('computational_core', 'compute_expectations', {
                'method_params': {'f_s': f_s, 'angles': angles}
            })
            expectations, indcs = result['data']
            
            print(f"Expectations: {expectations}, Indices: {indcs}")
            abs_expectations = np.abs(expectations)
            print(f"Absolute expectations: {abs_expectations}")
            
            policy, policy_data = self._compute_policy(action_space, abs_expectations, step)
            print(f"Policy: {policy}, Policy data: {policy_data}")
            
            idx = np.random.choice(range(len(policy)), p=policy)
            edge, sign = action_space[idx], np.sign(expectations[idx])
            print(f"Step {step + 1}/{self.n - self.nc}: Selected action: edge={edge}, sign={sign}")
            
            result = {'action': (edge, sign)}
            
            if compute_gradients:
                gradients = {'xy': [], 'beta': 0}
                
                if self.learning_rates[0] > 0:
                    betas = (policy_data[0] if self.init_beta[0] in ["one-all", "all-all"] 
                            else np.ones(len(f_s)) * policy_data)
                    
                    gradients['xy'] = route_request('computational_core', 'compute_log_pol_diff', {
                        'method_params': {'f_s': f_s, 'indcs': indcs, 'policy': policy, 'expectations': expectations, 'betas': betas, 'idx': idx, 'sign': sign, 'angles': angles}
                    })['data']
                    
                    print(f"Gradients xy: {gradients['xy']}")
                
                if self.init_beta[0] in ["one-all", "all-all"]:
                    grad = np.zeros(int((self.n ** 2 - self.n) / 2))
                    betas_idx = policy_data[1]
                    grad[betas_idx[idx]] += abs_expectations[idx]
                    for i in range(len(policy)):
                        grad[betas_idx[i]] -= policy[i] * abs_expectations[i]
                    gradients['beta'] = grad
                else:
                    gradients['beta'] = abs_expectations[idx] - np.dot(policy, abs_expectations)
                print(f"Gradients beta: {gradients['beta']}")
                
                result.update({'grad_xy': gradients['xy'], 'grad_beta': gradients['beta']})
            
            return result
        except Exception as e:
            print(f"Error in _execute_step: {e}")
            raise
    
    def _run_episode(self, f_s: List, action_space: List, training: bool = True) -> Dict[str, Any]:
        """Execute one training/evaluation episode."""
        print("Starting episode...")
        
        route_request('graph', 'reset', {})['data']
        
        nodelist = np.arange(self.n)
        print(f"Initial nodes: {nodelist}")
        assignments, signs = [], []
        gradients = {'xy': [], 'beta': []} if training else None
        
        for m in range(self.n - self.nc):
            print(f"Step {m + 1}/{self.n - self.nc}")
            try:
                step_data = self._execute_step(m, f_s, action_space, nodelist, training)
                edge, sign = step_data['action']
                print(f"Selected action: edge={edge}, sign={sign}")
                assignments.append(edge)
                signs.append(sign)
                
                if training:
                    gradients['xy'].extend(step_data['grad_xy'])
                    gradients['beta'].append(step_data['grad_beta'])
                
                if m < self.n - self.nc - 1:
                    result = route_request('graph', 'eliminate', {
                        'method_params': {'edge': edge, 'sign': sign}
                    })
                    rmv_edges, updt_edges, add_edges = result['data']
                    nodelist = nodelist[nodelist != edge[1]]
                    J = route_request('graph', 'get_G_numpy', {
                        'method_params': {'nodelist': nodelist}
                    })['data']
                    result = route_request('computational_core', 'update', {
                        'method_params': {'f_s': f_s, 'action_space': action_space, 'J': J, 'nodelist': nodelist, 'rmv_edges': rmv_edges, 'updt_edges': updt_edges, 'add_edges': add_edges}
                    })
                    f_s, action_space = result['data']
                else:
                    route_request('graph', 'eliminate', {
                        'method_params': {'edge': edge, 'sign': sign}
                    })
                    nodelist = nodelist[nodelist != edge[1]]
                    
            except Exception as e:
                print(f"Error in _run_episode on step {m + 1}: {e}")
                raise
        
        J = route_request('graph', 'get_G_numpy', {
            'method_params': {'nodelist': nodelist}
        })['data']
        result = route_request('computational_core', 'bruteforce', {
            'method_params': {'J': J, 'n': self.nc}
        })
        _, z_c = result['data']
        result = route_request('rqaoaalgorithms', 'expand_result_with_energies', {
            'method_params': {'z_c': z_c, 'assignments': assignments, 'signs': signs, 'nodelist': nodelist}
        })
        z_s, ep_energies, ep_contribs = result['data']
        
        result = {'energies': ep_energies, 'contributions': ep_contribs, 'final_state': z_s[0]}
        if training:
            result['gradients'] = gradients
        print(f"Episode completed: {result}")
        return result
    
    def _compute_returns(self, ep_energies: List[float], ep_contribs: List[float]) -> List[float]:
        """Compute returns."""
        if self.reward_fct == "nrg":
            returns = [ep_energies[0] * (self.gamma ** i) for i in range(self.n - self.nc)]
            return returns[::-1]
        elif self.reward_fct == "ep-cntrb":
            returns, discounted_sum = [], 0
            for r in ep_contribs[::-1]:
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)
            return returns
        else:
            return ep_energies
    
    def _update_parameters(self) -> None:
        """Update agent parameters."""
        batch_returns = np.array(self.batch_returns)
        if self.normalize:
            batch_returns = batch_returns - np.mean(batch_returns, axis=0)
        
        avg_grad_xy = np.zeros_like(self.all_angles)
        avg_grad_beta = np.zeros_like(self.betas)
        
        for i in range(self.batch_size):
            returns, grad_xy, grad_beta = batch_returns[i], self.batch_gradients['xy'][i], self.batch_gradients['beta'][i]
            
            for j in range(min(len(grad_xy) // 2, self.n - self.nc)):
                if len(grad_xy) > 2 * j + 1:
                    avg_grad_xy[2 * j:2 * j + 2] += returns[j] * np.array(grad_xy[2 * j:2 * j + 2])
            
            if self.init_beta[0] == "one":
                avg_grad_beta += np.dot(returns, grad_beta)
            elif self.init_beta[0] == "one-all":
                avg_grad_beta += np.sum([returns[j] * grad_beta[j] for j in range(len(returns))], axis=0)
            elif self.init_beta[0] == "all-all":
                avg_grad_beta += np.concatenate([returns[j] * grad_beta[j] for j in range(len(returns))])
            else:
                avg_grad_beta += np.array(returns) * grad_beta
        
        avg_grad_xy /= self.batch_size
        avg_grad_beta /= self.batch_size
        
        updates_result = route_request('adamwoptimizer', 'get_updates', {
            'method_params': {'gradients': [avg_grad_xy, avg_grad_beta]},
            'instance_id': id(self.optimizer)
        })
        updates = updates_result['data']
        self.all_angles += updates[0]
        self.betas += updates[1]
    
    def train_batch(self, f_s0: List, action_space0: List) -> None:
        """Train one batch."""
        self.batch_gradients = {'xy': [], 'beta': []}
        self.batch_returns = []
        
        for episode in range(self.batch_size):
            episode_data = self._run_episode(f_s0.copy(), action_space0.copy(), True)
            final_energy = episode_data['energies'][0]
            self.history_rewards.append(final_energy)
            print(f'Energy: {final_energy}')
            
            returns = self._compute_returns(episode_data['energies'], episode_data['contributions'])
            self.batch_returns.append(returns)
            self.batch_gradients['xy'].append(episode_data['gradients']['xy'])
            self.batch_gradients['beta'].append(episode_data['gradients']['beta'])
        
        self._update_parameters()
    
    def _evaluate_batch(self, f_s0: List, action_space0: List) -> None:
        """Evaluate batch without training."""
        for episode in range(self.batch_size):
            episode_data = self._run_episode(f_s0.copy(), action_space0.copy(), False)
            self.history_rewards.append(episode_data['energies'][0])
            print(f'Energy: {episode_data["energies"][0]}')
    
    def play_train(self, nb_batches: int, train: bool = True, store: bool = True) -> None:
        """Main agent training loop."""
        route_request('graph', 'reset', {})['data']
        nodelist = np.array(sorted(self.G.nodes()))
        J = route_request('graph', 'get_G_numpy', {
            'method_params': {'nodelist': nodelist}
        })['data']
        result = route_request('computational_core', 'generate_fs_h_actions', {
            'method_params': {'J': J, 'nodelist': nodelist}
        })
        f_s0, _, action_space0 = result['data']
        
        for batch in range(nb_batches):
            print(f"Batch {batch + 1}/{nb_batches}")
            
            if train:
                self.train_batch(f_s0.copy(), action_space0.copy())
            else:
                self._evaluate_batch(f_s0.copy(), action_space0.copy())
            
            if store:
                self.store_agent()
    
    def store_agent(self, pickle_path: Optional[str] = None) -> None:
        """Save agent to file."""
        path = pickle_path or self.pickle_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics."""
        if not self.history_rewards:
            return {}
        rewards = np.array(self.history_rewards)
        return {
            'mean_reward': np.mean(rewards), 'std_reward': np.std(rewards),
            'max_reward': np.max(rewards), 'min_reward': np.min(rewards),
            'total_episodes': len(rewards)
        }


def create_default_graph() -> Tuple[int, int, nx.Graph, int, str]:
    """Create default graph."""
    nc, d, n = 8, 8, 14
    G = nx.random_regular_graph(d=d, n=n, seed=6121619833208740511)
    G = _random_weights(graph=G, rs=np.random.RandomState(42), type='bimodal')
    return n, nc, G, d, '8d_14n_bimodaldist_6121619833208740511seed_17'


def parse_graph_name(G_name: str) -> Tuple[int, int, nx.Graph, int]:
    """Parse graph name to extract parameters."""
    parts = G_name.split('_')
    d = int(parts[0][:-1])
    n = int(parts[1][:-1])
    generator_seed = int(parts[3][:-4])
    distribution = parts[2][:-4]
    
    G = nx.random_regular_graph(d=d, n=n, seed=generator_seed)
    return n, 8, _random_weights(graph=G, rs=np.random.RandomState(42), type=distribution), d


def _random_weights(graph: nx.Graph, rs: np.random.RandomState, type: str = 'bimodal'):
    """Add random weights to graph."""
    problem_graph = nx.Graph()
    for n1, n2 in graph.edges:
        if type == 'bimodal':
            problem_graph.add_edge(n1, n2, weight=rs.choice([-1, 1]))
        elif type == 'gaussian':
            problem_graph.add_edge(n1, n2, weight=rs.randn())
        elif type == 'one':
            problem_graph.add_edge(n1, n2, weight=1)
        else:
            raise ValueError(f"Unsupported weight type: {type}")
    return problem_graph


if __name__ == "__main__":
    sys_args = sys.argv
    
    if len(sys_args) > 1:
        n, nc, G, d, G_name = (create_default_graph() if str(sys_args[3]) == 'None' 
                              else (*parse_graph_name(str(sys_args[3])), str(sys_args[3])))
        if str(sys_args[3]) != 'None':
            print(f'Solving problem for {G_name}')
        
        init_angles = [str(sys_args[9]), 
                      None if str(sys_args[10]) == 'None' else float(sys_args[10]),
                      None if str(sys_args[11]) == 'None' else str(sys_args[11])]
        
        agent = RQAOA_agent(
            n=n, nc=nc, G=G, d=d, learning_rates=[float(sys_args[4]), float(sys_args[5])], 
            gamma=float(sys_args[6]), init_beta=[str(sys_args[7]), float(sys_args[8])],
            init_angles=init_angles, batch_size=int(sys_args[12]), 
            normalize=(str(sys_args[13]) == 'True'), reward_fct=str(sys_args[14]), 
            run=str(sys_args[15]), idx=str(sys_args[1]), G_name=G_name
        )
        
        agent.play_train(nb_batches=int(sys_args[2]), train=(str(sys_args[16]) == 'True'))
    else:
        print("Usage: python rqaoa_agent_main.py <idx> <nb_batches> <G_name> ...")
        print("For full list of arguments see documentation.")