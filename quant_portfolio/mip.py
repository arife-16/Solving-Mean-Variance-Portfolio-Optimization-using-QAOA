import numpy as np

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
