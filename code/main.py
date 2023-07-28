##############################################################################
# Main code to run all the results from the paper
##############################################################################
import argparse
import plot_figures
import setParams
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "PROTOCOL", help="which protocol to run", choices=[0, 1, 2, 3, 4], type=int
)
parser.add_argument(
    "NTHREADS", help="number of threads to use", choices=range(1, 41), type=int
)
args = parser.parse_args()

# Which protocol to run
p = args.PROTOCOL
# Number of local threads to use
lnt = args.NTHREADS

if p == 0:
    import fig2

    ##########################################################################
    # Simulation parameters
    ##########################################################################
    # Time resolution
    dt = 0.2
    # Simulation time
    simtime = 600.0
    # Transient time
    transient = 0.0
    ###########################################################
    # Figure 1B
    ###########################################################
    t, Rstrong = fig2.simulate(
        lnt=lnt,
        simtime=simtime,
        dt=dt,
        tau=30.0,
        transient=transient,
        Wee=6.00,
        Wei=6.70,
    )
    t, Rweak = fig2.simulate(
        lnt=lnt,
        simtime=simtime,
        dt=dt,
        tau=20.0,
        transient=transient,
        Wee=4.45,
        Wei=4.70,
    )

    plot_figures.fig2b(t, Rweak, Rstrong)

    ###########################################################
    # Figure 1C
    ###########################################################
    Wee = np.linspace(4, 7, 70)  # Coupling E -> E
    Wei = np.linspace(4.5, 7.5, 70)  # Coupling I -> E
    Fmax = np.zeros([Wee.shape[0], Wei.shape[0]])

    for i in range(Wee.shape[0]):
        for j in range(Wei.shape[0]):
            wee = Wee[i]
            wei = Wei[j]
            t, R = fig2.simulate(
                simtime=simtime, dt=dt, transient=transient, Wee=Wee[i], Wei=Wei[j]
            )
            Fmax[j, i] = R.max()

    plot_figures.fig2c(Fmax, [Wee.min(), Wee.max(), Wei.min(), Wei.max()])

if p == 1:
    import fig3

    ##########################################################################
    # Simulation parameters
    ##########################################################################
    simtime = 10000.0  # ms
    trans = 0.2  # ms
    dt = 0.2  # ms
    seed = 100

    ##########################################################################
    # Model parameters (default is Weak GBA)
    ##########################################################################
    params = setParams.get_params_rate_model()

    ##########################################################################
    # Simulating Weak GBA
    ##########################################################################
    tidx, r_w, max_freq_w = fig3.simulate(
        lnt=lnt, simtime=simtime, dt=dt, params=params, max_cond=False, seed=10
    )

    ##########################################################################
    # Simulating Strong GBA
    ##########################################################################
    params = setParams.get_params_rate_model(gba="strong-gba")
    tidx, r_s, max_freq_s = fig3.simulate(
        lnt=lnt, simtime=simtime, dt=dt, params=params, max_cond=False, seed=10
    )

    plot_figures.fig3b_d(tidx, r_w, max_freq_w, r_s, max_freq_s)

    ##########################################################################
    # Varying \mu_{ee} for weak/strong GBA (with and withoud max condition)
    ##########################################################################
    muee_vec = np.arange(20, 52, 2, dtype=float)
    max_r_24c_f = np.zeros([muee_vec.shape[0], 2])
    max_r_24c_t = np.zeros([muee_vec.shape[0], 2])

    params = setParams.get_params_rate_model()

    G = 55.0 / 178.0

    for uu in [0, 1]:
        for i in range(muee_vec.shape[0]):
            if uu == 1:
                params["weights"]["wei"] = 19.7 + (muee_vec[i] - 33.7) * G
            else:
                params["weights"]["wei"] = 19.7
            params["weights"]["muee"] = muee_vec[i]
            t, r, max_freq = fig3.simulate(
                lnt=lnt, simtime=simtime, dt=dt, params=params, seed=10, max_cond=False
            )
            max_r_24c_f[i, uu] = max_freq[-1] - 10
            if max_r_24c_f[i, uu] > 500.0:
                max_r_24c_f[i, uu] = 500.0

    for uu in [0, 1]:
        for i in range(muee_vec.shape[0]):
            if uu == 1:
                params["weights"]["wei"] = 19.7 + (muee_vec[i] - 33.7) * G
            else:
                params["weights"]["wei"] = 19.7
            params["weights"]["muee"] = muee_vec[i]
            t, r, max_freq = fig3.simulate(
                lnt=lnt, simtime=simtime, dt=dt, params=params, seed=10, max_cond=True
            )
            max_r_24c_t[i, uu] = max_freq[-1] - 10
            if max_r_24c_t[i, uu] > 500.0:
                max_r_24c_t[i, uu] = 500.0

    plot_figures.fig3_f(muee_vec, max_r_24c_t, max_r_24c_f)

if p == 2:
    import fig4_5_6

    ##########################################################################
    # Simulation parameters
    ##########################################################################
    simtime = 1000.0  # Simulation time (ms)
    dt = 0.1  # Simulation resolution (ms)
    np.random.seed(10)

    ##########################################################################
    # Weak GBA / Asynchronous
    ##########################################################################
    i_aw, _, t_aw, _, maxf_aw, _, _ = fig4_5_6.simulate(
        lnt=lnt,
        seed=10,
        simtime=simtime,
        reg="async",
        gba="weak-gba",
        transient=0,
        dt=dt,
    )

    ##########################################################################
    # Strong GBA / Asynchronous
    ##########################################################################
    i_as, _, t_as, _, maxf_as, _, _ = fig4_5_6.simulate(
        lnt=lnt,
        seed=10,
        simtime=simtime,
        reg="async",
        gba="strong-gba",
        transient=0,
        dt=dt,
    )
    plot_figures.fig4_5_6(
        t_aw, i_aw, t_as, i_as, maxf_aw, maxf_as, "async", simtime, "figures/fig4.png"
    )

    ##########################################################################
    # Weak GBA / Synchronous
    ##########################################################################
    i_sw, _, t_sw, _, maxf_sw, _, _ = fig4_5_6.simulate(
        lnt=lnt,
        seed=10,
        simtime=simtime,
        reg="sync",
        gba="weak-gba",
        transient=0,
        dt=dt,
    )

    ##########################################################################
    # Strong GBA / Synchronous
    ##########################################################################
    i_ss, _, t_ss, _, maxf_ss, _, _ = fig4_5_6.simulate(
        lnt=lnt,
        seed=10,
        simtime=simtime,
        reg="sync",
        gba="strong-gba",
        transient=0,
        dt=dt,
    )
    plot_figures.fig4_5_6(
        t_sw, i_sw, t_ss, i_ss, maxf_sw, maxf_ss, "sync", simtime, "figures/fig5.png"
    )


if p == 3:
    import fig4_5_6

    ##########################################################################
    # Simulation parameters
    ##########################################################################
    simtime = 1000.0  # Simulation time (ms)
    dt = 0.1  # Simulation resolution (ms)
    np.random.seed(10)

    ##########################################################################
    # Strong GBA / Synchronous
    ##########################################################################
    i_ss, _, t_ss, _, maxf_ss, _, _ = fig4_5_6.simulate(
        lnt=lnt,
        seed=100,
        simtime=simtime,
        reg="sync",
        gba="strong-gba",
        transient=0,
        dt=dt,
    )
    plot_figures.plot_raster(t_ss, i_ss, None, simtime, "figures/fig6.png", save=True)

# Compute average frequencie per population in each stage
if p == 4:
    import fig4_5_6

    ##########################################################################
    # Simulation parameters
    ##########################################################################
    simtime = 6000.0  # Simulation time (ms)
    trans = 1000.0
    dt = 0.1  # Simulation resolution (ms)
    np.random.seed(10)

    Ie = np.linspace(220, 320, 20)
    Ii = np.linspace(220, 320, 20)

    ##########################################################################
    # Weak GBA / Asynchronous
    ##########################################################################
    _, _, _, _, _, rewa, riwa = fig4_5_6.simulate(
        lnt=lnt,
        seed=100,
        simtime=simtime,
        reg="async",
        gba="weak-gba",
        transient=trans,
        dt=dt,
        input_to_v1=False,
        use_default_noise=False,
        Ie_mean=284.0,
        Ii_mean=294.0,
    )

    ##########################################################################
    # Strong GBA / Asynchronous
    ##########################################################################
    _, _, _, _, _, resa, risa = fig4_5_6.simulate(
        lnt=lnt,
        seed=100,
        simtime=simtime,
        reg="async",
        gba="strong-gba",
        transient=trans,
        dt=dt,
        input_to_v1=False,
        use_default_noise=False,
        Ie_mean=284.0,
        Ii_mean=293.0,
    )

    ##########################################################################
    # Weak GBA / Synchronous
    ##########################################################################
    _, _, _, _, _, rews, riws = fig4_5_6.simulate(
        lnt=lnt,
        seed=100,
        simtime=simtime,
        reg="sync",
        gba="weak-gba",
        transient=trans,
        dt=dt,
        input_to_v1=False,
        use_default_noise=False,
        Ie_mean=298.0,
        Ii_mean=270.0,
    )

    ##########################################################################
    # Strong GBA / Synchronous
    ##########################################################################
    _, _, _, _, _, ress, riss = fig4_5_6.simulate(
        lnt=lnt,
        seed=100,
        simtime=simtime,
        reg="sync",
        gba="strong-gba",
        transient=trans,
        dt=dt,
        input_to_v1=False,
        use_default_noise=False,
        Ie_mean=310.0,
        Ii_mean=270.0,
    )

    print(f"          Weak GBA           ")
    print(f"           | Async |  Sync  |")
    print(f"   E       | {rewa}|  {rews}|")
    print(f"   I       | {riwa}|  {riws}|")
    print("------------------------------")
    print(f"          Strong GBA         ")
    print(f"           | Async |  Sync  |")
    print(f"   E       | {resa}|  {ress}|")
    print(f"   I       | {risa}|  {riss}|")
