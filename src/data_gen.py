"""
Synthetic data generation for the EV charging allocation problem.

An "instance" is a dict describing a deterministic (point-forecast) problem:
    n_evs, n_chargers, horizon (number of time slots), slot_minutes,
    arrival[i]      -> earliest slot EV i can start charging
    deadline[i]     -> last slot by which EV i must finish
    demand[i]       -> energy required, in "slot-units" of a full-power charger
                        (i.e. number of slots of charging needed at that
                        charger's power rating -- kept simple/dimension-free
                        so the MILP stays linear and easy to audit)
    charger_power[j]-> relative power of charger j (kWh delivered per slot)
    price[t]         -> energy price ($/kWh) at slot t (time-of-use tariff)

A "scenario" perturbs arrival[i] and raw energy demand (kWh) around the
nominal instance, keeping the physical resources (chargers, prices, horizon)
fixed -- this is the standard setup for scenario-based stochastic
programming applied to scheduling under uncertain job arrivals/sizes.
"""
import numpy as np


def time_of_use_prices(horizon, low=0.12, high=0.35, peak_start_frac=0.55, peak_width_frac=0.25):
    """Simple two-tier time-of-use price curve with a peak window."""
    t = np.arange(horizon)
    peak_start = int(peak_start_frac * horizon)
    peak_end = int((peak_start_frac + peak_width_frac) * horizon)
    prices = np.full(horizon, low, dtype=float)
    prices[peak_start:peak_end] = high
    return prices


def generate_instance(n_evs, n_chargers, horizon, seed=0,
                       slot_minutes=30, mean_energy_kwh=10.0, std_energy_kwh=3.0,
                       charger_power_kwh_per_slot=3.3, min_stay_slots=3):
    """
    Generate one nominal (point-forecast) instance.

    Arrivals are spread across the first 2/3 of the horizon (Poisson-process
    like), each EV gets a deadline min_stay_slots to horizon-1 slots after
    arrival, and energy demand is drawn from a Gamma-ish (kept simple as
    truncated normal) distribution converted into charger-slots-needed.
    """
    rng = np.random.default_rng(seed)

    arrival_window = max(1, int(horizon * 2 / 3))
    arrival = np.sort(rng.integers(0, arrival_window, size=n_evs))

    max_stay = horizon - arrival
    stay = np.array([
        rng.integers(min_stay_slots, max(min_stay_slots + 1, ms)) if ms > min_stay_slots
        else min_stay_slots
        for ms in max_stay
    ])
    deadline = np.minimum(arrival + stay, horizon - 1)

    energy_kwh = np.clip(rng.normal(mean_energy_kwh, std_energy_kwh, size=n_evs), 2.0, None)
    # energy demand expressed as number of full-power charger-slots needed
    demand_slots = np.ceil(energy_kwh / charger_power_kwh_per_slot).astype(int)
    # never require more slots than the EV's own stay window allows exactly-
    # infeasible instances are fine (that's what the unmet-demand slack is for)

    charger_power = np.full(n_chargers, charger_power_kwh_per_slot)
    price = time_of_use_prices(horizon)

    return dict(
        n_evs=n_evs, n_chargers=n_chargers, horizon=horizon,
        slot_minutes=slot_minutes,
        arrival=arrival, deadline=deadline,
        energy_kwh=energy_kwh, demand_slots=demand_slots,
        charger_power=charger_power, price=price,
    )


def perturb_scenario(nominal, seed, arrival_jitter_slots=2, energy_std_frac=0.25):
    """
    Draw one realized scenario around a nominal instance: arrival times
    shift by a small integer jitter (clipped to stay within [0, horizon-1]
    and not after original deadline), and energy demand is re-sampled with
    relative noise.
    """
    rng = np.random.default_rng(seed)
    n = nominal['n_evs']
    horizon = nominal['horizon']

    jitter = rng.integers(-arrival_jitter_slots, arrival_jitter_slots + 1, size=n)
    arrival = np.clip(nominal['arrival'] + jitter, 0, horizon - 1)
    # deadline shifts together with arrival so the stay duration is preserved,
    # but never exceeds the horizon
    stay = nominal['deadline'] - nominal['arrival']
    deadline = np.minimum(arrival + stay, horizon - 1)
    arrival = np.minimum(arrival, deadline)  # safety

    energy_kwh = np.clip(
        nominal['energy_kwh'] * rng.normal(1.0, energy_std_frac, size=n), 2.0, None
    )
    demand_slots = np.ceil(energy_kwh / nominal['charger_power'][0]).astype(int)

    scenario = dict(nominal)
    scenario['arrival'] = arrival
    scenario['deadline'] = deadline
    scenario['energy_kwh'] = energy_kwh
    scenario['demand_slots'] = demand_slots
    return scenario


def generate_scenarios(nominal, n_scenarios, seed=0, arrival_jitter_slots=2, energy_std_frac=0.25):
    return [
        perturb_scenario(nominal, seed=seed * 10_000 + s,
                          arrival_jitter_slots=arrival_jitter_slots,
                          energy_std_frac=energy_std_frac)
        for s in range(n_scenarios)
    ]
