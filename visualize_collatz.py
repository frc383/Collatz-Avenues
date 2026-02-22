import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# ==========================================
# 1. CORE MATHEMATICAL ENGINE
# ==========================================

def get_v2(n):
    """Calculates the 2-adic valuation (altitude) of n."""
    if n == 0: return 0
    count = 0
    # Use bitwise operation for efficiency
    while n > 0 and (n & 1) == 0:
        count += 1
        n >>= 1
    return count

def lift_operator(n):
    """Applies the lifted operator A(n) = 3n + 2^v2(n)."""
    return 3 * n + (1 << get_v2(n))

def get_lifted_trajectory(n, steps=50):
    """Generates a trajectory under the lifted operator A(n)."""
    traj = [n]
    for _ in range(steps):
        n = lift_operator(n)
        traj.append(n)
    return traj

def get_convergence_trajectory(n, max_steps=500):
    """Tracks a trajectory until it hits a power of two (odd kernel = 1)."""
    traj = [n]
    for _ in range(max_steps):
        if n > 0 and (n / (2**get_v2(n))) == 1:
            break
        n = lift_operator(n)
        traj.append(n)
    return traj

def get_syracuse_trajectory(m, steps=50):
    """Generates the Syracuse map trajectory (odd numbers only)."""
    m = m // (2**get_v2(m)) if m > 0 else m
    traj = [m]
    for _ in range(steps):
        # Syracuse Map: T(m) = (3m + 1) / 2^k
        m = (3 * m + 1)
        m = m // (2**get_v2(m))
        traj.append(m)
        if m == 1: break
    return traj

# ==========================================
# 2. VISUALIZATION ENGINE
# ==========================================

def plot_figure_1(seed=27, L=41):
    """Figure 1: Syracuse vs. Altitude (Matched Scales)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Syracuse Plot (Left)
    syr_traj = get_syracuse_trajectory(seed, L)
    ax1.plot(syr_traj, color='#e67e22', marker='s', markersize=3, linewidth=1.2)
    ax1.set_title(f"Syracuse Magnitude (n={seed})")
    ax1.set_yscale('log')
    ax1.set_xlabel("Odd Step (L)")
    ax1.set_ylabel("Value (Log Scale)")
    ax1.grid(True, which="both", alpha=0.2)

    # Altitude Plot (Right)
    lifted_traj = get_lifted_trajectory(seed, L)
    altitudes = [get_v2(x) for x in lifted_traj]
    ax2.plot(altitudes, color='#3498db', marker='o', markersize=4)
    ax2.set_title("2-adic Altitude Flow (v2)")
    ax2.set_yscale('log')
    ax2.set_xlabel("Odd Step (L)")
    ax2.grid(True, which="both", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('arrow_of_time.png', dpi=300)
    plt.show()

def plot_figure_2(num_cycles=4):
    """Figure 2: 3n+5 Helix (Multi-Cycle Rational Orbit)."""
    base_kernels, base_rel_alts = [19, 31, 49], [0, 1, 2] 
    shift_per_cycle = 5
    colors = ['#8e44ad', '#2980b9', '#27ae60', '#f39c12']
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(num_cycles):
        seg_kernels = base_kernels + [base_kernels[0]]
        seg_alts = [a + (i * shift_per_cycle) for a in base_rel_alts] + [(i + 1) * shift_per_cycle]
        z, x = np.array(seg_alts), np.log(seg_kernels) / np.log(3)
        t = np.linspace(i * 2 * np.pi, (i + 1) * 2 * np.pi, len(seg_kernels))
        y = np.sin(t)
        ax.plot(x, y, z, marker='o', color=colors[i % len(colors)], linewidth=2.5, label=f'Cycle {i+1}')
    
    ax.set_title("3n+5 Rational Helix ($19/5$ Orbit)")
    ax.set_xlabel("mu (Kernel)"); ax.set_ylabel("tau (Phase)"); ax.set_zlabel("k (Altitude)")
    ax.legend(); ax.view_init(elev=25, azim=45)
    plt.savefig('19_5_helix.png', dpi=300)
    plt.show()

def plot_figure_3(target_L=12, count=50):
    """Figure 3: Dual Dyadic Ribbon (Vortex vs. Global Drain)."""
    def find_convergence_seeds(target, num_required):
        found, queue = [], [(1, 0)]
        while len(found) < num_required and queue:
            curr, l = queue.pop(0)
            if l == target: found.append(curr); continue
            for k in range(1, 40):
                num = curr * (2**k) - 1
                if num > 0 and num % 3 == 0:
                    x = num // 3
                    if x > 1 and x % 2 != 0: queue.append((x, l + 1))
                    if len(queue) > 500: break
        return found
        
    vortex_seeds = find_convergence_seeds(target_L, count)
    full_conv_seeds = [27 + i*4 for i in range(count)]
    
    fig = plt.figure(figsize=(20, 10))
    ax1, ax2 = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')
    start_col, stop_col = np.array([255, 0, 127])/255.0, np.array([0,0,0])/255.0
    
    def draw_vortex(ax, seeds, is_fixed=True, title=""):
        for idx, seed in enumerate(seeds):
            traj = get_lifted_trajectory(seed, target_L) if is_fixed else get_convergence_trajectory(seed)
            mu = np.log([x/(2**get_v2(x)) for x in traj]) / np.log(3)
            theta = np.linspace(0, 2 * np.pi, len(traj))
            x, y, z = mu * np.cos(theta), mu * np.sin(theta), [get_v2(n) for n in traj]
            
            for i in range(len(traj)-1):
                prog = i/(len(traj)-1)
                c = (1-prog)*start_col + prog*stop_col
                ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=c, linewidth=1, alpha=0.6)
            
            ax.scatter(x[0], y[0], z[0], color=start_col, s=15, marker='o', label="Start" if idx==0 else "")
            ax.scatter(x[-1], y[-1], z[-1], color=stop_col, s=40, marker='*', label="Stop" if idx==0 else "")
        
        ax.set_title(title); ax.view_init(elev=20, azim=30)
        ax.legend(loc='upper right', fontsize=7)

    draw_vortex(ax1, vortex_seeds, True, "Vortex (Fixed L=12 Convergence)")
    draw_vortex(ax2, full_conv_seeds, False, "Global Drain (Full Path to n=1)")
    plt.tight_layout(); plt.savefig('dyadic_ribbon.png', dpi=300); plt.show()

def plot_figure_4(seed1=27, seed2=871):
    """Figure 4: Parity Overhead Side-by-Side Comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def plot_overhead(ax, seed):
        n, l = seed, 0
        while n > 1 and l < 500:
            v = get_v2(n)
            if v > 0: n >>= v
            if n == 1: break
            n = 3 * n + 1; l += 1
            
        traj = get_lifted_trajectory(seed, l)
        s_obs = np.cumsum([get_v2(x) for x in traj])
        steps = np.arange(len(s_obs))
        s_min = steps * np.log2(3)
        
        ax.plot(steps, s_obs, label="Observed Shift (S_obs)", color='#2c3e50', lw=2)
        ax.plot(steps, s_min, label="Diophantine Min (S_min)", color='#bdc3c7', ls='--')
        ax.fill_between(steps, s_min, s_obs, color='#ecf0f1', alpha=0.5, label="Parity Overhead")
        ax.set_title(f"Parity Overhead (n={seed}, L={l})")
        ax.legend(); ax.grid(True, alpha=0.3)

    plot_overhead(ax1, seed1); plot_overhead(ax2, seed2)
    plt.tight_layout(); plt.savefig('parity_overhead.png', dpi=300); plt.show()

# ==========================================
# 3. DATA & EXECUTION
# ==========================================

def generate_table(seeds=[3, 7, 27, 31, 73, 871, 937, 159487, 1000001, 31948311]):
    """Generates the experimental data table for parity overhead analysis."""
    results = []
    for n_start in seeds:
        n, l, s = n_start, 0, 0
        while n > 1 and l < 1000:
            v = get_v2(n)
            if v > 0: s += v; n >>= v
            if n == 1: break
            n = 3 * n + 1; l += 1
        s_min = int(np.ceil(l * np.log2(3)))
        gap = abs(2**s - 3**l)
        results.append({
            "Seed": n_start, "L": l, "S_min": s_min, "S_obs": s, 
            "Overhead": s-s_min, "Interval": round((3**l)/gap, 4) if gap != 0 else 0
        })
    df = pd.DataFrame(results)
    print("\nParity Overhead Data Analysis:\n", df.to_string(index=False))
    return df

if __name__ == "__main__":
    generate_table()
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
