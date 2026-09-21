# Mixed-Integer Optimisation for EV Charging Allocation — Report

## 1. Problem statement

We allocate a fleet of electric vehicles (EVs), each arriving with a
release time, a deadline and an energy demand, to a limited set of
chargers over a discretised time horizon. This is a **resource-constrained
scheduling problem**: chargers are unary resources (each serves at most one
EV per time slot), EVs are jobs with release times, deadlines and
processing volume, and the objective balances **energy cost** (chargers
draw power under a time-of-use tariff), **waiting time** (how late, on
average, an EV's charging happens relative to its arrival) and **service
level** (unmet energy demand, penalized as a soft constraint so the model
stays feasible under resource scarcity instead of simply failing).

## 2. Deterministic mixed-integer formulation

**Sets.** EVs `i ∈ I`, chargers `j ∈ J`, time slots `t ∈ T = {0,...,H-1}`.

**Parameters.** `arrival_i`, `deadline_i` (feasible window), `demand_i`
(energy requirement, in units of one charger-slot at rated power),
`price_t` (energy price per slot), `power_j` (charger rating).

**Decision variables.**
- `x[i,j,t] ∈ {0,1}` — EV *i* is charged by charger *j* in slot *t*
  (only defined for `t ∈ [arrival_i, deadline_i]`).
- `u[i] ≥ 0` — unmet energy demand for EV *i* (continuous slack).

**Objective.**

```
minimize   Σ x[i,j,t] · price_t · power_j                  (energy cost)
         + β · Σ x[i,j,t] · (t − arrival_i)                  (waiting proxy)
         + γ · Σ u[i]                                        (unmet demand)
```

The waiting term is a linear proxy for lateness: it sums, over every
charging slot actually used, how far that slot sits after the EV's
arrival — cheap to encode (no auxiliary "start time" variable needed) and
it still rewards front-loading a vehicle's charging close to arrival.

**Constraints.**

```
Σ_{j,t} x[i,j,t] + u[i]  ≥  demand_i          for all i         (energy)
Σ_i x[i,j,t]             ≤  1                 for all j,t       (capacity)
```

`x` binary, `0 ≤ u[i] ≤ demand_i`. This is solved to global optimality
with `scipy.optimize.milp` (HiGHS branch-and-cut, open source, no license
required).

## 3. Uncertainty-aware extension: two-stage stochastic MILP

Real arrival times and energy demands are only known approximately ahead
of time. We extend the model to a **two-stage stochastic program with
recourse**, solved in extensive (deterministic-equivalent) form:

- **Stage 1 (here-and-now).** A plan `x0[i,j,t]` is committed *before* the
  true arrivals/demands are known, using nominal (point-forecast) data.
- **Stage 2 (recourse, per scenario `s`).** Once `(arrival_i^s, demand_i^s)`
  is realized, the operator may deviate from the plan — reassigning
  slots — but pays a recourse penalty `ρ` per slot changed, on top of the
  usual energy/waiting/unmet costs evaluated under the *realized* data.
  Deviation is linearized with auxiliary variables
  `z[i,j,t,s] ≥ |x[i,j,t,s] − x0[i,j,t]|`.

The objective is the sample-average approximation (SAA) of the expected
total cost across `S` sampled scenarios:

```
minimize_{x0}   (1/S) Σ_s  [ energy_cost(x^s) + β·waiting(x^s)
                             + γ·unmet(x^s) + ρ·deviation(x^s, x0) ]
```

Solving stage 1 and stage 2 *jointly* (rather than optimizing `x0` on the
nominal instance alone) is what makes the plan genuinely hedge against the
scenario spread rather than just reproducing the deterministic solution.

**Scenario generation.** Arrival times are jittered by a small integer
number of slots and energy demand is resampled with ±25% relative noise
around the nominal instance (see `data_gen.perturb_scenario`).

## 4. Evaluation methodology (why this is more than "it runs")

To make claims about the stochastic model being "better", we do not just
compare in-sample objective values (which is not a fair comparison — the
stochastic model has seen more information). Instead we use the standard
stochastic-programming protocol:

- **EV policy** — solve the deterministic model once on the nominal
  (expected-value) instance.
- **RP policy** — solve the two-stage stochastic model on `S` training
  scenarios.
- **WS (wait-and-see)** — for each test scenario, solve the deterministic
  model *knowing that scenario exactly* — an unattainable clairvoyant
  upper bound on performance.
- All three are evaluated on a **fresh, disjoint set of out-of-sample test
  scenarios**, with EV/RP plans evaluated via the same recourse model
  (fixed first-stage plan + realized data + deviation penalty). This gives
  an apples-to-apples realized cost distribution per policy.

From these we compute:
- **VSS** (Value of the Stochastic Solution) = mean(EV) − mean(RP). VSS ≥
  0 shows hedging against uncertainty genuinely pays off.
- **EVPI** (Expected Value of Perfect Information) = mean(RP) − mean(WS).
  Bounds how much a perfect forecaster could still improve on the
  stochastic solution.

## 5. Experiments and results

All code is in `src/experiments.py` / `src/visualize.py`; raw numbers are
in `results/*.csv`, figures in `figures/*.png`.

### Experiment 1 — Scalability (`figures/exp1_scalability.png`)

Deterministic MILP solve time vs. number of EVs (4 chargers, 24 slots),
from 5 to 100 EVs. Solve time stays under **0.1 s** even at 100 EVs
(≈4,500 variables) — HiGHS handles this problem class comfortably.
Total cost grows roughly linearly with fleet size until chargers
saturate (~20-25 EVs on 4 chargers over the horizon used), after which
**unmet demand appears and cost grows super-linearly** — a clean,
interpretable resource-saturation effect rather than an artefact.

### Experiment 2 — Cost / waiting-time trade-off (`figures/exp2_tradeoff.png`)

Sweeping the waiting-weight `β` at fixed resources (15 EVs) shows the
expected trade-off, but its *shape depends heavily on resource
availability*: with only 2 chargers there is almost no schedule flexibility
(waiting cost scales almost linearly with `β` while energy cost is nearly
flat, because the schedule is essentially forced by capacity) until, at a
high enough `β` (=8), it becomes cheaper to shed demand entirely (unmet
cost turns on). With 4 chargers there is real slack: energy cost rises
from **$18.9 to $27.3** as `β` grows, buying a large reduction in waiting
cost. This is a good illustration of how resource-constrained scheduling
trade-off curves are not resource-independent.

### Experiment 3 — Value of the stochastic solution (`figures/exp3_uncertainty.png`)

On an 8-EV / 2-charger / 12-slot instance (chosen so the two-stage model
solves to proven optimality), trained on 10 scenarios and evaluated on 30
fresh out-of-sample scenarios:

| Policy | Mean realized cost | Std |
|---|---|---|
| EV (deterministic) | 496.08 | 96.13 |
| RP (stochastic) | 490.73 | 98.48 |
| WS (perfect information) | 477.51 | 97.93 |

**VSS = 5.35, EVPI = 13.22** (both positive, as theory requires). The
stochastic plan modestly but consistently outperforms the plan built from
a single point forecast when both are stress-tested under realistic
variability, and there remains a ~13-cost-unit gap to a clairvoyant
scheduler — i.e. most of the achievable benefit from "knowing the future"
is about the deviation/recourse structure, not the forecast itself.

### Experiment 4 — SAA convergence (`figures/exp4_saa_convergence.png`)

Out-of-sample cost of the RP policy as the number of training scenarios
`S` grows from 4 to 10: mean cost decreases and stabilizes (496.8 → 490.3)
while training solve time grows from 0.3 s to ~1.9 s — a standard SAA
diminishing-returns curve, and evidence the 10-scenario setting used in
Experiment 3 is a reasonable convergence point rather than an arbitrary
choice.

### Illustrative schedule (`figures/example_schedule_gantt.png`)

A Gantt-style rendering of one solved instance (8 EVs, 3 chargers),
showing each EV's arrival/deadline window and its actual assigned
charger/slots, for sanity-checking the model visually.

## 6. Limitations and honest scope

- The extensive-form two-stage MILP scales as `O(S)` in both variables and
  constraints; beyond ~15 scenarios or ~10 EVs at this horizon, HiGHS no
  longer closes the optimality gap within the time limits used here (this
  is expected and is exactly why real deployments use scenario reduction,
  Benders decomposition, or progressive hedging instead of the extensive
  form — noted as a natural extension, not implemented here).
- Energy demand is modelled in discrete "charger-slot" units rather than
  continuous kWh with modulated charging power — a simplification standard
  in scheduling-style MILP formulations, traded off against tractability.
- Uncertainty is scenario-based (SAA); a robust-optimization (worst-case)
  variant was considered but not implemented, and would be a natural
  follow-up for a "distributionally cautious" operating mode.

## 7. Reproducing

```bash
cd src
python experiments.py
python visualize.py
```
All random seeds are fixed in `experiments.py` for reproducibility.
