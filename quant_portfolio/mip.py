import numpy as np
import math

def relax_mvo_qp(mu, sigma, q, N, K):
    try:
        import cvxpy as cp
    except Exception:
        return None
    x = cp.Variable(N)
    obj = q * cp.quad_form(x, sigma) - mu @ x
    cons = [cp.sum(x) == K, x >= 0, x <= 1]
    prob = cp.Problem(cp.Minimize(obj), cons)
    solvers = ["OSQP", "SCS", "GUROBI", "MOSEK", "CPLEX"]
    res = None
    for s in solvers:
        try:
            res = prob.solve(solver=getattr(cp, s))
            break
        except Exception:
            continue
    if res is None or x.value is None:
        return None
    return np.array(x.value).ravel()

def solve_mad_lp(returns, N, K):
    try:
        import pulp
    except Exception:
        return None
    T = returns.shape[0]
    prob = pulp.LpProblem("mad_lp", pulp.LpMinimize)
    w = [pulp.LpVariable(f"w_{i}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous) for i in range(N)]
    mu = pulp.LpVariable("mu", lowBound=None, upBound=None, cat=pulp.LpContinuous)
    u = [pulp.LpVariable(f"u_{t}", lowBound=0.0, cat=pulp.LpContinuous) for t in range(T)]
    # Portfolio returns per period: r_p(t) = sum_i w_i * r_{t,i}
    # Constraints for absolute deviations: u_t >= r_p(t) - mu and u_t >= -(r_p(t) - mu)
    for t in range(T):
        rp_t = pulp.lpSum([w[i] * float(returns[t, i]) for i in range(N)])
        prob += u[t] >= rp_t - mu
        prob += u[t] >= -(rp_t - mu)
    # Sum of weights equals 1 (normalized budget). To map to K selections later, we will pick top-K.
    prob += pulp.lpSum(w) == 1.0
    # Objective: minimize MAD minus mean return (risk-return trade-off)
    mean_ret = (1.0 / float(T)) * pulp.lpSum([pulp.lpSum([w[i] * float(returns[t, i]) for i in range(N)]) for t in range(T)])
    mad = (1.0 / float(T)) * pulp.lpSum(u)
    prob += mad - mean_ret
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    w_sol = np.array([pulp.value(w[i]) for i in range(N)], dtype=float)
    # Construct a K-hot bitstring by selecting top-K weights
    idx = np.argsort(-w_sol)[:K]
    z = 0
    for i in idx:
        z |= (1 << int(i))
    # Compute objective value for selected discrete portfolio for reporting
    rp = returns @ (w_sol)
    mu_p = rp.mean()
    mad_val = np.abs(rp - mu_p).mean()
    val = float(mad_val - mu_p)
    return float(val), int(z)

def solve_mvo_milp(mu, sigma, q, N, K, tc=None, lam_tc=0.0):
    try:
        import pulp
    except Exception:
        return None
    prob = pulp.LpProblem("mvo_milp", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary) for i in range(N)]
    y = {}
    for i in range(N):
        for j in range(i, N):
            y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpContinuous)
            prob += y[(i, j)] <= x[i]
            prob += y[(i, j)] <= x[j]
            prob += y[(i, j)] >= x[i] + x[j] - 1
    quad = 0
    for i in range(N):
        quad += sigma[i, i] * y[(i, i)]
        for j in range(i + 1, N):
            quad += 2.0 * sigma[i, j] * y[(i, j)]
    lin = 0
    for i in range(N):
        lin += -mu[i] * x[i]
    if tc is not None and lam_tc != 0.0:
        for i in range(N):
            lin += lam_tc * tc[i] * x[i]
    prob += q * quad + lin
    prob += pulp.lpSum(x) == K
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    x_sol = np.array([pulp.value(x[i]) for i in range(N)])
    z = 0
    for i in range(N):
        if x_sol[i] >= 0.5:
            z |= (1 << i)
    val = q * x_sol @ sigma @ x_sol - mu @ x_sol
    if tc is not None and lam_tc != 0.0:
        val += lam_tc * (tc @ x_sol)
    return float(val), int(z)

def solve_mvo_miqp(mu, sigma, q, N, K, tc=None, lam_tc=0.0):
    try:
        import cvxpy as cp
    except Exception:
        return None
    x = cp.Variable(N, boolean=True)
    quad = cp.quad_form(x, sigma)
    lin = -mu @ x
    if tc is not None and lam_tc != 0.0:
        lin = lin + lam_tc * (tc @ x)
    obj = q * quad + lin
    constraints = [cp.sum(x) == K]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    solvers = ["GUROBI", "MOSEK", "CPLEX", "SCIP", "ECOS_BB"]
    res = None
    for s in solvers:
        try:
            res = prob.solve(solver=getattr(cp, s))
            break
        except Exception:
            continue
    if res is None or x.value is None:
        return None
    x_sol = np.array(x.value).ravel()
    z = 0
    for i in range(N):
        if x_sol[i] >= 0.5:
            z |= (1 << i)
    val = float(q * x_sol @ sigma @ x_sol - mu @ x_sol)
    if tc is not None and lam_tc != 0.0:
        val += float(lam_tc * (tc @ x_sol))
    return float(val), int(z)
