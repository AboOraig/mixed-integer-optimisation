"""
Deterministic mixed-integer formulation of the EV charging allocation
problem, framed as a resource-constrained scheduling problem:

    - "jobs"      = EVs, each with a release time (arrival), deadline,
                    and processing volume (energy demand in charger-slots)
    - "resources" = chargers, unary resources (serve one EV per time slot)

Decision variables
-------------------
x[i,j,t] in {0,1}   EV i is charged by charger j during slot t
u[i]     >= 0        unmet energy demand for EV i (slack, penalized)

Formulation
-----------
minimize   sum_{i,j,t} x[i,j,t] * price[t] * power[j]         (energy cost)
         + beta * sum_{i,j,t} x[i,j,t] * (t - arrival[i])      (waiting proxy)
         + gamma * sum_i u[i]                                  (unmet demand)

subject to
    sum_{j,t feasible for i} x[i,j,t] + u[i] >= demand_slots[i]   for all i
    sum_i x[i,j,t] <= 1                                            for all j,t
    x[i,j,t] = 0 for t outside [arrival[i], deadline[i]]  (enforced by
                                                            simply not
                                                            creating the var)
    x binary, 0 <= u[i] <= demand_slots[i]

This is solved with scipy.optimize.milp (HiGHS branch-and-cut backend --
no external solver / license required).
"""
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds
import time


def _valid_slots(instance, i):
    return range(instance['arrival'][i], instance['deadline'][i] + 1)


def build_variable_index(instance):
    """Map every feasible (i,j,t) triple to a column index; u[i] follow after."""
    n, m = instance['n_evs'], instance['n_chargers']
    x_index = {}
    col = 0
    for i in range(n):
        for j in range(m):
            for t in _valid_slots(instance, i):
                x_index[(i, j, t)] = col
                col += 1
    n_x = col
    u_index = {i: n_x + i for i in range(n)}
    n_vars = n_x + n
    return x_index, u_index, n_vars


def solve_deterministic(instance, beta=0.5, gamma=50.0, fixed_x=None,
                         deviation_penalty=None, time_limit=60):
    """
    Solve the deterministic charging-allocation MILP.

    If `fixed_x` (a dict of the SAME instance's variable index -> value in
    {0,1}, typically a first-stage plan from the stochastic model) and
    `deviation_penalty` are both given, the model instead solves a
    *recourse* problem: it may deviate from fixed_x but pays
    `deviation_penalty` per slot changed (added/removed). This is used to
    evaluate a fixed first-stage schedule against a realized scenario.

    Returns a dict with the solution and diagnostics.
    """
    x_index, u_index, n_vars = build_variable_index(instance)
    n, m, H = instance['n_evs'], instance['n_chargers'], instance['horizon']
    price, power = instance['price'], instance['charger_power']

    n_x = len(x_index)
    extra_dev_vars = 0
    dev_index = {}
    if fixed_x is not None and deviation_penalty is not None:
        # one auxiliary deviation variable z[i,j,t] >= |x[i,j,t]-fixed_x[i,j,t]|
        for key in x_index:
            dev_index[key] = n_vars + extra_dev_vars
            extra_dev_vars += 1
    n_total = n_vars + extra_dev_vars

    # ---- objective ----
    c = np.zeros(n_total)
    for (i, j, t), col in x_index.items():
        c[col] = price[t] * power[j] + beta * (t - instance['arrival'][i])
    for i, col in u_index.items():
        c[col] = gamma
    for key, col in dev_index.items():
        c[col] = deviation_penalty

    # ---- constraints ----
    rows, cols, data = [], [], []
    lb_list, ub_list = [], []
    row = 0

    # (1) energy demand per EV
    for i in range(n):
        for j in range(m):
            for t in _valid_slots(instance, i):
                rows.append(row); cols.append(x_index[(i, j, t)]); data.append(1.0)
        rows.append(row); cols.append(u_index[i]); data.append(1.0)
        lb_list.append(instance['demand_slots'][i]); ub_list.append(np.inf)
        row += 1

    # (2) charger capacity per (j,t): at most 1 EV per charger per slot
    for j in range(m):
        for t in range(H):
            cols_here = [x_index[(i, j, t)] for i in range(n)
                         if (i, j, t) in x_index]
            if not cols_here:
                continue
            for col in cols_here:
                rows.append(row); cols.append(col); data.append(1.0)
            lb_list.append(-np.inf); ub_list.append(1.0)
            row += 1

    # (3) deviation linking constraints: z >= x - x0 ; z >= x0 - x
    if dev_index:
        for key, xcol in x_index.items():
            x0 = fixed_x.get(key, 0)
            zcol = dev_index[key]
            # z - x >= -x0   =>  -x + z >= -x0
            rows.append(row); cols.append(xcol); data.append(-1.0)
            rows.append(row); cols.append(zcol); data.append(1.0)
            lb_list.append(-x0); ub_list.append(np.inf)
            row += 1
            # z + x >= x0
            rows.append(row); cols.append(xcol); data.append(1.0)
            rows.append(row); cols.append(zcol); data.append(1.0)
            lb_list.append(x0); ub_list.append(np.inf)
            row += 1

    A = sparse.csr_matrix((data, (rows, cols)), shape=(row, n_total))
    constraints = LinearConstraint(A, lb_list, ub_list)

    lb = np.zeros(n_total)
    ub = np.ones(n_total)
    for i, col in u_index.items():
        ub[col] = instance['demand_slots'][i]
    for key, col in dev_index.items():
        ub[col] = 1.0
    bounds = Bounds(lb, ub)

    integrality = np.zeros(n_total)
    for col in x_index.values():
        integrality[col] = 1
    for col in dev_index.values():
        integrality[col] = 1
    # u[i] left continuous

    t0 = time.time()
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
               options={'time_limit': time_limit, 'disp': False})
    solve_time = time.time() - t0

    x_sol = {}
    u_sol = {}
    if res.x is not None:
        for key, col in x_index.items():
            x_sol[key] = int(round(res.x[col]))
        for i, col in u_index.items():
            u_sol[i] = res.x[col]

    energy_cost = sum(price[t] * power[j] * x_sol.get((i, j, t), 0)
                       for (i, j, t) in x_index) if res.x is not None else None
    waiting_cost = sum(beta * (t - instance['arrival'][i]) * x_sol.get((i, j, t), 0)
                        for (i, j, t) in x_index) if res.x is not None else None
    unmet_cost = gamma * sum(u_sol.values()) if res.x is not None else None

    return dict(
        status=res.status, success=res.success, message=res.message,
        objective=res.fun, x=x_sol, u=u_sol, x_index=x_index,
        energy_cost=energy_cost, waiting_cost=waiting_cost, unmet_cost=unmet_cost,
        solve_time=solve_time, n_vars=n_total, n_constraints=row,
    )
