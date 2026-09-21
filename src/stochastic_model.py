"""
Two-stage stochastic MILP for EV charging under uncertain arrival times and
energy demand, solved in extensive form (a.k.a. the deterministic
equivalent) with scipy's HiGHS backend.

Stage 1 (here-and-now): a planned schedule x0[i,j,t] is committed BEFORE
the true arrival times / energy demands are known, using the nominal
(point-forecast) instance.

Stage 2 (wait-and-see, per scenario s): once (arrival_i^s, demand_i^s) is
observed, the operator may deviate from the plan -- reassigning slots --
but pays a recourse penalty rho per unit of deviation, on top of the usual
energy, waiting and unmet-demand costs evaluated under the REALIZED data.

This is the standard SAA (sample average approximation) construction:
    min_{x0}  E_s[ Q(x0, scenario_s) ]
where Q is the scenario's recourse cost including the deviation penalty.
Solving the extensive form jointly (rather than x0 then independent
recourse) is what makes x0 genuinely hedge against the scenario spread,
rather than just being the nominal deterministic solution.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds
import time


def _valid_slots_generic(arrival, deadline, i):
    return range(arrival[i], deadline[i] + 1)


def _all_possible_xkeys(n, m, H):
    """Stage-1 variables must be defined over the UNION of all slots any
    scenario could plausibly use, since the plan is committed before the
    scenario is known. We use the full horizon per EV to be safe (small
    instances keep this cheap)."""
    return [(i, j, t) for i in range(n) for j in range(m) for t in range(H)]


def solve_two_stage(nominal, scenarios, probs=None, beta=0.5, gamma=50.0,
                     rho=2.0, time_limit=120):
    """
    Build and solve the extensive-form two-stage stochastic MILP.

    nominal   : instance dict (defines n_evs, n_chargers, horizon, prices...)
    scenarios : list of instance dicts (same n/m/H/price, different
                arrival/deadline/demand_slots), e.g. from generate_scenarios
    probs     : optional list of scenario probabilities (default: uniform)
    rho       : per-slot recourse (deviation) penalty
    """
    n, m, H = nominal['n_evs'], nominal['n_chargers'], nominal['horizon']
    price, power = nominal['price'], nominal['charger_power']
    S = len(scenarios)
    probs = np.array(probs) if probs is not None else np.full(S, 1.0 / S)

    # ---- stage-1 variable index: x0[i,j,t] over full horizon ----
    x0_keys = _all_possible_xkeys(n, m, H)
    x0_index = {key: idx for idx, key in enumerate(x0_keys)}
    n_x0 = len(x0_keys)

    # ---- stage-2 variables per scenario: x^s[i,j,t] (only within that
    #      scenario's [arrival,deadline]), u^s[i], z^s[i,j,t] (deviation) ----
    col = n_x0
    x_s_index = []   # list over s of dict (i,j,t)->col
    u_s_index = []   # list over s of dict i->col
    z_s_index = []   # list over s of dict (i,j,t)->col  (only for keys that exist in x_s)
    for s, sc in enumerate(scenarios):
        xi = {}
        for i in range(n):
            for j in range(m):
                for t in _valid_slots_generic(sc['arrival'], sc['deadline'], i):
                    xi[(i, j, t)] = col; col += 1
        x_s_index.append(xi)
        ui = {}
        for i in range(n):
            ui[i] = col; col += 1
        u_s_index.append(ui)
        zi = {}
        for key in xi:
            zi[key] = col; col += 1
        z_s_index.append(zi)
    n_total = col

    # ---- objective ----
    c = np.zeros(n_total)
    for s, sc in enumerate(scenarios):
        w = probs[s]
        for (i, j, t), cidx in x_s_index[s].items():
            c[cidx] = w * (price[t] * power[j] + beta * (t - sc['arrival'][i]))
        for i, cidx in u_s_index[s].items():
            c[cidx] = w * gamma
        for key, cidx in z_s_index[s].items():
            c[cidx] = w * rho
    # x0 itself carries no direct cost (it's a plan, not a physical action);
    # all realized cost flows through the scenario recourse variables.

    rows, cols_, data = [], [], []
    lb_list, ub_list = [], []
    row = 0

    for s, sc in enumerate(scenarios):
        xi, ui, zi = x_s_index[s], u_s_index[s], z_s_index[s]

        # (1) energy demand per EV under realized scenario
        for i in range(n):
            for j in range(m):
                for t in _valid_slots_generic(sc['arrival'], sc['deadline'], i):
                    rows.append(row); cols_.append(xi[(i, j, t)]); data.append(1.0)
            rows.append(row); cols_.append(ui[i]); data.append(1.0)
            lb_list.append(sc['demand_slots'][i]); ub_list.append(np.inf)
            row += 1

        # (2) charger capacity per (j,t) under realized scenario
        for j in range(m):
            for t in range(H):
                cols_here = [xi[(i, j, t)] for i in range(n) if (i, j, t) in xi]
                if not cols_here:
                    continue
                for cidx in cols_here:
                    rows.append(row); cols_.append(cidx); data.append(1.0)
                lb_list.append(-np.inf); ub_list.append(1.0)
                row += 1

        # (3) deviation linking: z >= x^s - x0 ; z >= x0 - x^s
        for key, xcol in xi.items():
            x0col = x0_index[key]
            zcol = zi[key]
            rows.append(row); cols_.append(xcol); data.append(1.0)
            rows.append(row); cols_.append(x0col); data.append(-1.0)
            rows.append(row); cols_.append(zcol); data.append(1.0)
            lb_list.append(0.0); ub_list.append(np.inf)
            row += 1
            rows.append(row); cols_.append(xcol); data.append(-1.0)
            rows.append(row); cols_.append(x0col); data.append(1.0)
            rows.append(row); cols_.append(zcol); data.append(1.0)
            lb_list.append(0.0); ub_list.append(np.inf)
            row += 1

    A = sparse.csr_matrix((data, (rows, cols_)), shape=(row, n_total))
    constraints = LinearConstraint(A, lb_list, ub_list)

    lb = np.zeros(n_total)
    ub = np.ones(n_total)
    for s, sc in enumerate(scenarios):
        for i, cidx in u_s_index[s].items():
            ub[cidx] = sc['demand_slots'][i]

    bounds = Bounds(lb, ub)

    integrality = np.zeros(n_total)
    for cidx in x0_index.values():
        integrality[cidx] = 1
    for s in range(S):
        for cidx in x_s_index[s].values():
            integrality[cidx] = 1
        for cidx in z_s_index[s].values():
            integrality[cidx] = 1
    # u variables stay continuous

    t0 = time.time()
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
               options={'time_limit': time_limit, 'disp': False})
    solve_time = time.time() - t0

    x0_sol = {}
    if res.x is not None:
        for key, cidx in x0_index.items():
            x0_sol[key] = int(round(res.x[cidx]))

    return dict(
        status=res.status, success=res.success, message=res.message,
        objective=res.fun, x0=x0_sol, x0_index=x0_index,
        solve_time=solve_time, n_vars=n_total, n_constraints=row, n_scenarios=S,
    )
