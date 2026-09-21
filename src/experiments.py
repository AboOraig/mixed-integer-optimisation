import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

from data_gen import generate_instance, generate_scenarios
from deterministic_model import solve_deterministic
from stochastic_model import solve_two_stage
from evaluate import evaluate_policy_out_of_sample, wait_and_see_costs, compute_vss_evpi

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def experiment_1_scalability():
    """How solve time and cost scale with problem size (number of EVs)."""
    rows = []
    for n in [5, 10, 15, 20, 30, 40, 50, 70, 100]:
        inst = generate_instance(n_evs=n, n_chargers=4, horizon=24, seed=42)
        res = solve_deterministic(inst, beta=0.5, gamma=50.0, time_limit=30)
        rows.append(dict(n_evs=n, solve_time=res['solve_time'],
                          objective=res['objective'], n_vars=res['n_vars'],
                          n_constraints=res['n_constraints'],
                          unmet_total=sum(res['u'].values()) if res['u'] else None))
        print(f"[exp1] n_evs={n:3d}  time={res['solve_time']:.3f}s  obj={res['objective']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp1_scalability.csv'), index=False)
    return df


def experiment_2_tradeoff():
    """Pareto trade-off between energy cost and waiting cost as the waiting
    weight (beta) is varied, at fixed resource level; repeated for two
    charger counts to show how added resources shift the frontier."""
    rows = []
    for n_chargers in [2, 4]:
        inst = generate_instance(n_evs=15, n_chargers=n_chargers, horizon=24, seed=7)
        for beta in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            res = solve_deterministic(inst, beta=beta, gamma=100.0, time_limit=30)
            rows.append(dict(n_chargers=n_chargers, beta=beta,
                              energy_cost=res['energy_cost'], waiting_cost=res['waiting_cost'],
                              unmet_cost=res['unmet_cost'], objective=res['objective']))
            print(f"[exp2] chargers={n_chargers} beta={beta:>4} "
                  f"energy={res['energy_cost']:.2f} wait={res['waiting_cost']:.2f} "
                  f"unmet={res['unmet_cost']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp2_tradeoff.csv'), index=False)
    return df


def experiment_3_uncertainty(n_evs=8, n_chargers=2, horizon=12,
                              n_train_scenarios=10, n_test_scenarios=30,
                              beta=0.5, gamma=50.0, rho=2.0):
    """Hero experiment: compare EV (deterministic-on-nominal), RP (two-stage
    stochastic) and WS (perfect information) policies, all evaluated on the
    SAME fresh out-of-sample scenarios. Reports VSS and EVPI."""
    nominal = generate_instance(n_evs=n_evs, n_chargers=n_chargers, horizon=horizon, seed=100)
    train_scenarios = generate_scenarios(nominal, n_scenarios=n_train_scenarios, seed=1,
                                          arrival_jitter_slots=2, energy_std_frac=0.25)
    test_scenarios = generate_scenarios(nominal, n_scenarios=n_test_scenarios, seed=999,
                                         arrival_jitter_slots=2, energy_std_frac=0.25)

    print("[exp3] solving EV (deterministic on nominal)...")
    ev_res = solve_deterministic(nominal, beta=beta, gamma=gamma, time_limit=60)
    x_ev = ev_res['x']

    print(f"[exp3] solving RP (two-stage, {n_train_scenarios} scenarios)...")
    rp_res = solve_two_stage(nominal, train_scenarios, beta=beta, gamma=gamma, rho=rho, time_limit=90)
    x_rp = rp_res['x0']

    print("[exp3] evaluating EV policy out-of-sample...")
    ev_costs = evaluate_policy_out_of_sample(x_ev, test_scenarios, beta, gamma, rho)
    print("[exp3] evaluating RP policy out-of-sample...")
    rp_costs = evaluate_policy_out_of_sample(x_rp, test_scenarios, beta, gamma, rho)
    print("[exp3] computing wait-and-see (perfect information) costs...")
    ws_costs = wait_and_see_costs(test_scenarios, beta, gamma)

    metrics = compute_vss_evpi(ev_costs, rp_costs, ws_costs)
    print("[exp3] metrics:", metrics)

    pd.DataFrame({'EV': ev_costs, 'RP': rp_costs, 'WS': ws_costs}).to_csv(
        os.path.join(RESULTS_DIR, 'exp3_out_of_sample_costs.csv'), index=False)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULTS_DIR, 'exp3_metrics.csv'), index=False)
    return dict(metrics=metrics, ev_costs=ev_costs, rp_costs=rp_costs, ws_costs=ws_costs,
                nominal=nominal, x_ev=x_ev, x_rp=x_rp, ev_res=ev_res, rp_res=rp_res)


def experiment_4_saa_convergence(n_evs=8, n_chargers=2, horizon=12,
                                  scenario_counts=(4, 6, 8, 10),
                                  n_test_scenarios=25, beta=0.5, gamma=50.0, rho=2.0):
    """Sample-average-approximation stability: does the RP policy's
    out-of-sample performance stabilize as the number of training
    scenarios grows?"""
    nominal = generate_instance(n_evs=n_evs, n_chargers=n_chargers, horizon=horizon, seed=100)
    test_scenarios = generate_scenarios(nominal, n_scenarios=n_test_scenarios, seed=999,
                                         arrival_jitter_slots=2, energy_std_frac=0.25)
    rows = []
    for S in scenario_counts:
        train_scenarios = generate_scenarios(nominal, n_scenarios=S, seed=1,
                                              arrival_jitter_slots=2, energy_std_frac=0.25)
        t0 = time.time()
        rp_res = solve_two_stage(nominal, train_scenarios, beta=beta, gamma=gamma, rho=rho, time_limit=60)
        rp_costs = evaluate_policy_out_of_sample(rp_res['x0'], test_scenarios, beta, gamma, rho)
        rows.append(dict(S=S, train_solve_time=time.time() - t0,
                          oos_mean=np.mean(rp_costs), oos_std=np.std(rp_costs),
                          train_objective=rp_res['objective'], status=rp_res['status']))
        print(f"[exp4] S={S:2d}  oos_mean={np.mean(rp_costs):.2f}  oos_std={np.std(rp_costs):.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'exp4_saa_convergence.csv'), index=False)
    return df


if __name__ == '__main__':
    experiment_1_scalability()
    experiment_2_tradeoff()
    experiment_3_uncertainty()
    experiment_4_saa_convergence()
