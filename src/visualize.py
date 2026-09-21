import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
})
COLORS = dict(EV='#4C72B0', RP='#DD8452', WS='#55A868', c2='#4C72B0', c4='#C44E52')


def fig_scalability():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp1_scalability.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].plot(df.n_evs, df.solve_time, marker='o', color='#4C72B0')
    axes[0].set_xlabel('Number of EVs'); axes[0].set_ylabel('Solve time (s)')
    axes[0].set_title('MILP solve time vs. problem size')

    axes[1].plot(df.n_evs, df.objective, marker='o', color='#C44E52')
    axes[1].set_xlabel('Number of EVs'); axes[1].set_ylabel('Objective ($)')
    axes[1].set_title('Total cost vs. problem size')

    axes[2].plot(df.n_evs, df.unmet_total, marker='o', color='#8172B2')
    axes[2].set_xlabel('Number of EVs'); axes[2].set_ylabel('Unmet demand (charger-slots)')
    axes[2].set_title('Resource saturation (4 chargers)')
    axes[2].axvspan(df.n_evs[df.unmet_total > 0].min() - 2, df.n_evs.max(), color='red', alpha=0.05)

    fig.suptitle('Experiment 1 — Scalability of the deterministic MILP (HiGHS)', y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp1_scalability.png'), bbox_inches='tight')
    plt.close(fig)


def fig_tradeoff():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp2_tradeoff.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)

    for ax, nch, color in zip(axes, [2, 4], [COLORS['c2'], COLORS['c4']]):
        sub = df[df.n_chargers == nch].sort_values('beta')
        ax.plot(sub.beta, sub.energy_cost, marker='o', color=color, label='Energy cost')
        ax2 = ax.twinx()
        ax2.plot(sub.beta, sub.waiting_cost, marker='s', color='gray', linestyle='--',
                  label='Waiting cost', alpha=0.8)
        ax.set_xscale('symlog', linthresh=0.1)
        ax.set_xlabel('Waiting-weight β (log scale)')
        ax.set_ylabel('Energy cost ($)', color=color)
        ax2.set_ylabel('Waiting cost (lateness proxy)', color='gray')
        ax.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='y', labelcolor='gray')
        ax.set_title(f'{nch} chargers')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    fig.suptitle('Experiment 2 — Energy cost vs. waiting-time trade-off as β increases\n'
                 '(15 EVs; note how 2 chargers leaves little room to trade, unlike 4)', y=1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp2_tradeoff.png'), bbox_inches='tight')
    plt.close(fig)


def fig_uncertainty():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp3_out_of_sample_costs.csv'))
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'exp3_metrics.csv')).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    data = [df['EV'].dropna(), df['RP'].dropna(), df['WS'].dropna()]
    bp = axes[0].boxplot(data, labels=['EV\n(deterministic)', 'RP\n(stochastic)', 'WS\n(perfect info)'],
                          patch_artist=True, widths=0.55)
    for patch, key in zip(bp['boxes'], ['EV', 'RP', 'WS']):
        patch.set_facecolor(COLORS[key]); patch.set_alpha(0.6)
    axes[0].set_ylabel('Out-of-sample realized cost ($)')
    axes[0].set_title('Experiment 3 — Policy comparison\n(30 out-of-sample scenarios)')

    bars = axes[1].bar(['VSS', 'EVPI'], [metrics['vss'], metrics['evpi']],
                        color=['#DD8452', '#55A868'])
    axes[1].set_ylabel('$')
    axes[1].set_title('Value of stochastic solution &\nexpected value of perfect information')
    for b in bars:
        h = b.get_height()
        axes[1].annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                          textcoords="offset points", xytext=(0, 4), ha='center')
    axes[1].axhline(0, color='black', linewidth=0.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp3_uncertainty.png'), bbox_inches='tight')
    plt.close(fig)


def fig_saa_convergence():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp4_saa_convergence.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].errorbar(df.S, df.oos_mean, yerr=df.oos_std, marker='o', capsize=4, color='#DD8452')
    axes[0].set_xlabel('Number of training scenarios (S)')
    axes[0].set_ylabel('Out-of-sample cost ($)')
    axes[0].set_title('SAA stability: out-of-sample cost vs. S')

    axes[1].plot(df.S, df.train_solve_time, marker='o', color='#4C72B0')
    axes[1].set_xlabel('Number of training scenarios (S)')
    axes[1].set_ylabel('Training solve time (s)')
    axes[1].set_title('Cost of hedging: solve time vs. S')

    fig.suptitle('Experiment 4 — Sample Average Approximation convergence', y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'exp4_saa_convergence.png'), bbox_inches='tight')
    plt.close(fig)


def fig_schedule_gantt():
    """Illustrative Gantt-style chart of one solved schedule (nominal EV policy)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from data_gen import generate_instance
    from deterministic_model import solve_deterministic

    # A slightly less resource-starved instance than the Experiment-3 nominal,
    # chosen purely for a clean illustrative picture of what the MILP outputs.
    inst = generate_instance(n_evs=8, n_chargers=3, horizon=14, seed=11)
    res = solve_deterministic(inst, beta=0.5, gamma=50.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    charger_colors = plt.cm.Set2(np.linspace(0, 1, inst['n_chargers']))
    for (i, j, t), val in res['x'].items():
        if val == 1:
            ax.barh(i, 1, left=t, height=0.6, color=charger_colors[j], edgecolor='white')
    for i in range(inst['n_evs']):
        ax.plot([inst['arrival'][i], inst['arrival'][i]], [i - 0.4, i + 0.4], color='black', lw=1.5)
        ax.plot([inst['deadline'][i] + 1, inst['deadline'][i] + 1], [i - 0.4, i + 0.4],
                 color='black', lw=1.5, linestyle='--')
    ax.set_yticks(range(inst['n_evs']))
    ax.set_yticklabels([f"EV {i}" for i in range(inst['n_evs'])])
    ax.set_xlabel('Time slot')
    ax.set_title('Example optimal schedule (solid=arrival, dashed=deadline)\ncolor = assigned charger')
    handles = [plt.Rectangle((0, 0), 1, 1, color=charger_colors[j]) for j in range(inst['n_chargers'])]
    ax.legend(handles, [f"Charger {j}" for j in range(inst['n_chargers'])],
              loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'example_schedule_gantt.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_scalability()
    fig_tradeoff()
    fig_uncertainty()
    fig_saa_convergence()
    fig_schedule_gantt()
    print("All figures written to", FIG_DIR)
