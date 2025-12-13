import numpy as np
import sys
import os
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import route_request


class RQAOAAlgorithms:
    """Basic RQAOA algorithms."""
    
    def __init__(self, n, nc, graph):
        self.n = n
        self.nc = nc
        self.graph = graph

    def rqaoa(self):
        """Classical RQAOA algorithm."""
        self.graph.reset()
        nodelist = np.array(sorted(self.graph.G.nodes()))
        J = self.graph.get_G_numpy(nodelist)

        result = route_request('computationalcore', 'generate_fs_h_actions', {'method_params': {'J': J, 'nodelist': nodelist}})
        f_s, h, action_space = result['data']
        assignments = []
        signs = []
        rqaoa_angles = np.array([], dtype=float)
        
        for m in range(self.n - self.nc):
            result = route_request('computationalcore', 'compute_extrema', {'method_params': {'h': h}})
            angles, f_val = result['data']
            rqaoa_angles = np.append(rqaoa_angles, angles)
            result = route_request('computationalcore', 'compute_expectations', {'method_params': {'f_s': f_s, 'angles': angles}})
            expectations, indcs = result['data']
            abs_expectations = np.abs(expectations)
            max_abs = np.flatnonzero(abs_expectations == abs_expectations.max())
            idx = np.random.choice(max_abs)
            edge, sign = action_space[idx], np.sign(expectations[idx])
            rmv_edges, updt_edges, add_edges = self.graph.eliminate(edge, sign)
            assignments.append(edge)
            signs.append(sign)
            nodelist = nodelist[nodelist != edge[1]]
            J = self.graph.get_G_numpy(nodelist)
            result = route_request('computationalcore', 'update', {'method_params': {'f_s': f_s, 'action_space': action_space, 'J': J, 'nodelist': nodelist, 'rmv_edges': rmv_edges, 'updt_edges': updt_edges, 'add_edges': add_edges}})
            f_s, action_space = result['data']
            result = route_request('computationalcore', 'compute_h', {'method_params': {'f_s': f_s, 'action_space': action_space, 'J': J, 'nodelist': nodelist}})
            h = result['data']
        
        result = route_request('computationalcore', 'bruteforce', {'method_params': {'J': J, 'nc': self.nc}})
        _, z_c = result['data']
        z_s, ep_energies, ep_contribs = self.expand_result_with_energies(z_c, assignments, signs, nodelist)

        return rqaoa_angles, ep_energies[0], z_s[0]

    def expand_result(self, z_c, assignments, signs, nodelist):
        """Result expansion."""
        z_s = [route_request('computationalcore', 'get_binary', {'method_params': {'z': z, 'length': len(nodelist)}})['data'] for z in z_c]
        z_s = [np.array([z[nodelist.tolist().index(i)] if i in nodelist else 0 
                        for i in range(self.n)], dtype=np.int32) for z in z_s]
        return z_s

    def expand_result_with_energies(self, z_c, assignments, signs, nodelist):
        """Result expansion with energy and contribution computation."""
        z_s = self.expand_result(z_c, assignments, signs, nodelist)
        
        original_graph = self.graph.G.copy()
        self.graph.reset()
        J = self.graph.get_G_numpy()
        
        ep_energies = []
        for i, assgn in enumerate(assignments[::-1]):
            for j, z in enumerate(z_s):
                z[assgn[1]] = signs[-i - 1] * z[assgn[0]]
                z_s[j] = z
            ep_energies.insert(0, J.dot(z_s[0]).dot(z_s[0]))
        
        ep_contribs = ([ep_energies[i] - ep_energies[i + 1] 
                       for i in range(len(ep_energies) - 1)] + [ep_energies[-1]])
        
        self.graph.G = original_graph
        
        return z_s, ep_energies, ep_contribs