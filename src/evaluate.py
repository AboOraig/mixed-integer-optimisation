"""
Out-of-sample evaluation of fixed first-stage schedules, and the classic
stochastic-programming quality metrics:

    VSS  (Value of the Stochastic Solution) = cost(EV policy) - cost(RP policy)
         evaluated on the SAME fresh out-of-sample scenarios via recourse.
         VSS >= 0 shows the stochastic plan hedges better than a plan based
         only on the nominal/expected instance.

    EVPI (Expected Value of Perfect Information) = cost(RP policy) - cost(WS)
         where WS ("wait-and-see") is the average cost if the scheduler
         magically knew each scenario in advance. EVPI >= 0 quantifies the
         remaining gap to a clairvoyant scheduler and bounds how much
         better forecasting could ever help.
"""
import numpy as np
from deterministic_model import solve_deterministic


def evaluate_policy_out_of_sample(fixed_x, test_scenarios, beta, gamma, rho, time_limit=30):
    """Evaluate a fixed first-stage plan against each test scenario via a
    recourse re-solve (may deviate from the plan at cost rho/slot).
    Returns an array of realized (recourse) costs, one per scenario."""
    costs = []
    for sc in test_scenarios:
        res = solve_deterministic(sc, beta=beta, gamma=gamma,
                                   fixed_x=fixed_x, deviation_penalty=rho,
                                   time_limit=time_limit)
        costs.append(res['objective'] if res['objective'] is not None else np.nan)
    return np.array(costs)


def wait_and_see_costs(test_scenarios, beta, gamma, time_limit=30):
    """Best achievable cost per scenario if the scenario were known in
    advance (no deviation penalty -- this IS the plan)."""
    costs = []
    for sc in test_scenarios:
        res = solve_deterministic(sc, beta=beta, gamma=gamma, time_limit=time_limit)
        costs.append(res['objective'] if res['objective'] is not None else np.nan)
    return np.array(costs)


def compute_vss_evpi(ev_costs, rp_costs, ws_costs):
    vss = np.mean(ev_costs) - np.mean(rp_costs)
    evpi = np.mean(rp_costs) - np.mean(ws_costs)
    return dict(
        ev_mean=np.mean(ev_costs), ev_std=np.std(ev_costs),
        rp_mean=np.mean(rp_costs), rp_std=np.std(rp_costs),
        ws_mean=np.mean(ws_costs), ws_std=np.std(ws_costs),
        vss=vss, evpi=evpi,
    )
