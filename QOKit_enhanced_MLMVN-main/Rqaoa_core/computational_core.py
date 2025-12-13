import numpy as np
import sympy as sym
import scipy
from scipy import optimize
import sys
import os

# Add interconnect to path
sys.path.append('/content/drive/MyDrive/QOKit_enhanced_MLMVN')
from interconnect import register_component, route_request


class ComputationalCore:
    """RQAOA computational core with mathematical operations."""
    
    def __init__(self, n, nc):
        self.n = n
        self.nc = nc
        self.x = sym.Symbol('x', real=True)
        self.y = sym.Symbol('y', real=True)
        self.grid_N = 1000

    def compute_expectations(self, f_s, angles):
        """Compute mathematical expectations."""
        x, y = self.x, self.y
        expectations = []
        indcs = []
        for i, f in enumerate(f_s):
            if f in f_s[:i]:
                indx = f_s.index(f)
                expectations.append(expectations[indx])
                indcs.append(indx)
            else:
                expectations.append(float(f.subs({x: angles[0], y: angles[1]})))
                indcs.append(i)
        return expectations, indcs

    def compute_log_pol2(self, f_s, indcs, policy, expectations, betas):
        """Compute logarithmic policy."""
        gather = np.zeros_like(policy)
        for i in range(len(indcs)):
            gather[indcs[i]] += policy[i] * betas[i]
        log_pol2 = 0.
        for i in range(len(gather)):
            if gather[i]:
                log_pol2 += sym.Float(gather[i] * np.sign(expectations[i]), 10) * f_s[i]
        return log_pol2

    def compute_log_pol_diff(self, f_s, indcs, policy, expectations, betas, idx, sign, angles):
        """Compute logarithmic policy differential."""
        x, y = self.x, self.y
        gather = np.zeros_like(policy)
        for i in range(len(indcs)):
            gather[indcs[i]] += policy[i] * betas[i]
        
        diff_log_pol_x = sign * betas[idx] * float(f_s[idx].diff(x).subs({x: angles[0], y: angles[1]}))
        diff_log_pol_y = sign * betas[idx] * float(f_s[idx].diff(y).subs({x: angles[0], y: angles[1]}))
        
        for i in range(len(gather)):
            if gather[i]:
                diff_log_pol_x -= gather[i] * np.sign(expectations[i]) * float(
                    f_s[i].diff(x).subs({x: angles[0], y: angles[1]}))
                diff_log_pol_y -= gather[i] * np.sign(expectations[i]) * float(
                    f_s[i].diff(y).subs({x: angles[0], y: angles[1]}))
        return [diff_log_pol_x, diff_log_pol_y]

    def get_idx_beta(self, edge):
        """Get beta parameter index."""
        i, j = edge
        if j < i:
            i, j = j, i
        return int(i * self.n - i * (i + 1) / 2 + j - i - 1)

    def get_binary(self, x, n):
        """Convert to binary representation."""
        return 2 * np.array([int(b) for b in bin(x)[2:].zfill(n)], dtype=np.int32) - 1

    def bruteforce(self, J, n):
        """Brute force solution."""
        maxi = -n
        idx = []
        for i in range(2 ** n - 1):
            z = self.get_binary(i, n)
            val = J.dot(z).dot(z)
            if val > maxi:
                maxi = val
                idx = [i]
            elif val == maxi:
                idx.append(i)
        return maxi, idx

    def compute_f(self, J, i, j):
        """Compute function f."""
        x, y = self.x, self.y
        prod1, prod2, prod3, prod4 = 1., 1., 1., 1.
        for k in range(len(J)):
            if k not in [i, j]:
                if J[i, k] - J[j, k]:
                    prod1 *= sym.cos(2 * x * (J[i, k] - J[j, k]))
                if J[i, k] + J[j, k]:
                    prod2 *= sym.cos(2 * x * (J[i, k] + J[j, k]))
                if J[i, k]:
                    prod3 *= sym.cos(2 * x * J[i, k])
                if J[j, k]:
                    prod4 *= sym.cos(2 * x * J[j, k])
        
        term = (sym.Rational(1, 2) * (sym.sin(2 * y) ** 2) * (prod1 - prod2) + 
                sym.cos(2 * y) * sym.sin(2 * y) * sym.sin(2 * x * J[i, j]) * (prod3 + prod4))
        
        return term

    def generate_fs_h_actions(self, J, nodelist):
        """Generate f_s functions, Hamiltonian h and action space."""
        f_s = []
        h = 0.
        action_space = []

        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    action_space.append((nodelist[i], nodelist[j]))
                    term = self.compute_f(J, i, j)
                    f_s.append(term)
                    h += J[i, j] * term
        return f_s, h, action_space

    def compute_h(self, f_s, action_space, J, nodelist):
        """Compute Hamiltonian."""
        h = 0.
        count = 0
        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    if action_space[count] != (nodelist[i], nodelist[j]):
                        raise ValueError("Incorrect index counting")
                    h += J[i, j] * f_s[count]
                    count += 1
        return h

    def compute_h_diff(self, f_s, indcs, action_space, J, nodelist):
        """Compute Hamiltonian differential."""
        x, y = self.x, self.y
        gather = np.zeros_like(f_s)
        count = 0
        for i in range(len(J)):
            for j in range(i + 1, len(J)):
                if J[i, j]:
                    if action_space[count] != (nodelist[i], nodelist[j]):
                        raise ValueError("Incorrect index counting")
                    gather[indcs[count]] += J[i, j]
                    count += 1
        diff_h_x = 0.
        diff_h_y = 0.
        for i in range(len(gather)):
            if gather[i]:
                diff_h_x += gather[i] * f_s[i].diff(x)
                diff_h_y += gather[i] * f_s[i].diff(y)
        return diff_h_x, diff_h_y

    def softmax(self, values, beta):
        """Softmax function."""
        vals = np.array(values)
        vals -= np.amax(vals)
        vals = np.exp(beta * vals)
        return vals / np.sum(vals)

    def compute_extrema(self, h, search_space=[0, np.pi]):
        """Compute extrema."""
        x, y = self.x, self.y

        r = (h.subs({x: x, y: np.pi/8}) + h.subs({x: x, y: -np.pi/8})) / 2
        q = (h.subs({x: x, y: np.pi/8}) - h.subs({x: x, y: -np.pi/8})) / 2
        p = h.subs({x: x, y: 0}) - r

        max_y_fun = r + sym.sqrt(p ** 2 + q ** 2)
        neg_max_y_fun = -max_y_fun
        fun = sym.lambdify([x], neg_max_y_fun, modules=['numpy'])
        
        param_ranges = (slice(search_space[0], search_space[1], 
                             abs(search_space[1] - search_space[0]) / self.grid_N),)
        res_brute = optimize.brute(fun, param_ranges, full_output=True, finish=optimize.fmin)

        solution_x = res_brute[0]
        if hasattr(solution_x, '__len__') and not isinstance(solution_x, str):
            solution_x = float(solution_x[0]) if len(solution_x) > 0 else float(solution_x)
        else:
            solution_x = float(solution_x)
        
        solution = [solution_x]

        q_val = q.subs({x: solution[0]})
        p_val = p.subs({x: solution[0]})
        y_val = 1 / 4 * (sym.atan2(q_val, p_val))
        
        if hasattr(y_val, 'evalf'):
            y_val = float(y_val.evalf())
        else:
            y_val = float(y_val)
        
        solution = np.append(solution, y_val)
        extrema = -res_brute[1]

        return solution, extrema

    def update(self, f_s, action_space, J, nodelist, rmv_edges, updt_edges, add_edges):
        """Update functions and action space."""
        nl = list(nodelist)
        
        for edge in rmv_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            if edge in action_space:
                indx = action_space.index(edge)
                action_space.pop(indx)
                f_s.pop(indx)

        for edge in add_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            inserted = False
            for i in range(len(action_space)):
                edge_i = action_space[i]
                if (edge_i[0] == edge[0] and edge_i[1] > edge[1]) or edge_i[0] > edge[0]:
                    action_space.insert(i, edge)
                    f_s.insert(i, self.compute_f(J, nl.index(edge[0]), nl.index(edge[1])))
                    inserted = True
                    break
            if not inserted:
                action_space.append(edge)
                f_s.append(self.compute_f(J, nl.index(edge[0]), nl.index(edge[1])))

        for edge in updt_edges:
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            if edge in action_space:
                indx = action_space.index(edge)
                f_s[indx] = self.compute_f(J, nl.index(edge[0]), nl.index(edge[1]))

        return f_s, action_space


# Register component in interconnect
register_component('computational_core', ComputationalCore)