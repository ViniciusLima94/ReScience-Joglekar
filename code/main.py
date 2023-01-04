#########################################################################################
# Main code to run all the results from the paper
#########################################################################################
import sys
import argparse
import plot_figures
import setParams
import numpy        as np

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("PROTOCOL", help="which protocol to run", choices=[0, 1, 2, 3],
                    type=int)
parser.add_argument("NTHREADS", help="number of threads to use", choices=range(1,41),
                    type=int)
args   = parser.parse_args()
# Which protocol to run
p      = args.PROTOCOL
# Number of local threads to use
lnt    = args.NTHREADS

# Will run the code fig1.py, to generate the first figure from the original paper
if p == 0:

    import fig1

    #########################################################################################
    # Simulation parameters
    #########################################################################################
    # Time resolution
    dt        = 0.2
    # Simulation time
    simtime   = 600.0
    # Transient time
    transient = 0.0
    ###########################################################
    # Figure 1B
    ###########################################################
    t, Rstrong = fig1.simulate(lnt = lnt, simtime = simtime, dt = dt, tau = 30.0, transient = transient, Wee = 6.00, Wei = 6.70)
    t, Rweak   = fig1.simulate(lnt = lnt, simtime = simtime, dt = dt, tau = 20.0, transient = transient, Wee = 4.45, Wei = 4.70)

    plot_figures.fig1b(t, Rweak, Rstrong)

    ###########################################################
    # Figure 1C
    ###########################################################
    Wee = np.linspace(4, 7, 70)     # Coupling E -> E
    Wei = np.linspace(4.5, 7.5, 70) # Coupling I -> E
    Fmax = np.zeros([Wee.shape[0], Wei.shape[0]])

    for i in range(Wee.shape[0]):
        for j in range(Wei.shape[0]):
            wee = Wee[i]
            wei = Wei[j]
            t, R = fig1.simulate(simtime = simtime, dt = dt, transient = transient, Wee = Wee[i], Wei = Wei[j])
            Fmax[j,i] = R.max()

    plot_figures.fig1c(Fmax, [Wee.min(), Wee.max(), Wei.min(), Wei.max()])

# Will run the third fig3.py, to generate the third figure from the original paper
if p == 1:

    import fig3

    #########################################################################################
    # Simulation parameters
    #########################################################################################
    simtime = 10000.  # ms
    trans   = 0.2     # ms
    dt      = 0.2     # ms
    seed    = 100

    #########################################################################################
    # Model parameters (default is Weak GBA)
    #########################################################################################
    params = setParams.get_params_rate_model()

    #########################################################################################
    # Simulating Weak GBA
    #########################################################################################
    tidx, r_w, max_freq_w = fig3.simulate(simtime = simtime, dt = dt, params=params, max_cond = False, seed = 10)

    #########################################################################################
    # Simulating Strong GBA
    #########################################################################################
    params = setParams.get_params_rate_model(gba='strong-gba') # Getting parameters for strong
    tidx, r_s, max_freq_s = fig3.simulate(simtime = simtime, dt = dt, params=params, max_cond = False, seed = 10)

    plot_figures.fig3b_d(tidx, r_w, max_freq_w, r_s, max_freq_s)

    #########################################################################################
    # Varying \mu_{ee} for weak and strong GBA (with and withoud max condition)
    #########################################################################################
    muee_vec    = np.arange(20,52,2, dtype=float)
    max_r_24c_f = np.zeros([muee_vec.shape[0],2]) # Store max freq in 24c with max_cond = False
    max_r_24c_t = np.zeros([muee_vec.shape[0],2]) # Store max freq in 24c with max_cond = True
    #params['Iext']            = 22.05*1.9
    params = setParams.get_params_rate_model()

    for uu in [0,1]:
        for i in range(muee_vec.shape[0]):
            if uu == 1:
                params['weights']['wei']  = 19.7 + (muee_vec[i]-33.7)*55.0/178.0
            else:
                params['weights']['wei']  = 19.7
            params['weights']['muee'] = muee_vec[i]
            t,r,max_freq = fig3.simulate(simtime = simtime, dt = dt, params = params, seed = 10, max_cond = False)
            max_r_24c_f[i,uu] = max_freq[-1]-10
            if max_r_24c_f[i,uu] > 500.:
                max_r_24c_f[i,uu] = 500.

    for uu in [0,1]:
        for i in range(muee_vec.shape[0]):
            if uu == 1:
                params['weights']['wei']  = 19.7 + (muee_vec[i]-33.7)*55.0/178.0
            else:
                params['weights']['wei']  = 19.7
            params['weights']['muee'] = muee_vec[i]
            t,r,max_freq = fig3.simulate(simtime = simtime, dt = dt, params = params, seed = 10, max_cond = True)
            max_r_24c_t[i,uu] = max_freq[-1]-10
            if max_r_24c_t[i,uu] > 500.:
                max_r_24c_t[i,uu] = 500.

    plot_figures.fig3_f(muee_vec, max_r_24c_t, max_r_24c_f)

# Will run the third fig5_6.py, to generate figure 5 and 6 from the original paper
if p == 2:

    import fig5_6

    #########################################################################################
    # Simulation parameters
    #########################################################################################
    simtime = 1000.0           # Simulation time (ms)
    dt      = 0.1              # Simulation resolution (ms)
    np.random.seed(10)

    #########################################################################################
    # Weak GBA / Asynchronous
    #########################################################################################
    i_aw, _, t_aw, _, maxf_aw,_,_ = fig5_6.simulate(lnt = lnt, seed = 10, simtime = simtime, reg = 'async', gba = 'weak-gba', transient = 0, dt = dt)

    #########################################################################################
    # Strong GBA / Asynchronous
    #########################################################################################
    i_as, _, t_as, _, maxf_as,_,_ = fig5_6.simulate(lnt = lnt, seed = 10, simtime = simtime, reg = 'async', gba = 'strong-gba', transient = 0, dt = dt)
    plot_figures.fig5_6(t_aw, i_aw, t_as, i_as, maxf_aw, maxf_as, 'async', simtime, 'figures/fig5.png')

    #########################################################################################
    # Weak GBA / Synchronous
    #########################################################################################
    i_sw, _, t_sw, _, maxf_sw,_,_ = fig5_6.simulate(lnt = lnt, seed = 10, simtime = simtime, reg = 'sync', gba = 'weak-gba', transient = 0, dt = dt)

    #########################################################################################
    # Strong GBA / Synchronous
    #########################################################################################
    i_ss, _, t_ss, _, maxf_ss,_,_ = fig5_6.simulate(lnt = lnt, seed = 100, simtime = simtime, reg = 'sync', gba = 'strong-gba', transient = 0, dt = dt)
    plot_figures.fig5_6(t_sw, i_sw, t_ss, i_ss, maxf_sw, maxf_ss, 'sync', simtime, 'figures/fig6.png')

    #########################################################################################
    # Strong GBA / Synchronous (bad seed)
    #########################################################################################
    i_ss, _, t_ss, _, maxf_ss,_,_ = fig5_6.simulate(lnt = lnt, seed = 10, simtime = simtime, reg = 'sync', gba = 'strong-gba', transient = 0, dt = dt)
    plot_figures.plot_raster(t_ss, i_ss, None, simtime,  'figures/fig7.png', save=True)

# Compute average frequencie per population in each stage
if p == 3:

    import fig5_6

    #########################################################################################
    # Simulation parameters
    #########################################################################################
    simtime = 6000.0           # Simulation time (ms)
    trans   = 1000.0
    dt      = 0.1               # Simulation resolution (ms)
    np.random.seed(10)

    Ie = np.linspace(220, 320, 20)
    Ii = np.linspace(220, 320, 20)

    #########################################################################################
    # Weak GBA / Asynchronous
    #########################################################################################
    _, _, _, _, _, rewa, riwa = fig5_6.simulate(lnt = lnt, seed = 100, simtime = simtime, reg = 'async', 
                                                gba = 'weak-gba', transient = trans, dt = dt, input_to_v1 = False, 
                                                use_default_noise=False, Ie_mean=284.0, Ii_mean=294.0)

    #########################################################################################
    # Strong GBA / Asynchronous
    #########################################################################################
    _, _, _, _, _, resa, risa = fig5_6.simulate(lnt = lnt, seed = 100, simtime = simtime, reg = 'async', 
                                               gba = 'strong-gba', transient = trans, dt = dt, input_to_v1 = False, 
                                               use_default_noise=False, Ie_mean=284.0, Ii_mean=293.)

    #########################################################################################
    # Weak GBA / Synchronous
    #########################################################################################
    _, _, _, _, _, rews, riws = fig5_6.simulate(lnt = lnt, seed = 100, simtime = simtime, reg = 'sync', 
                                                gba = 'weak-gba', transient = trans, dt = dt, input_to_v1 = False, 
                                                use_default_noise=False, Ie_mean=298.0, Ii_mean=270.0)

    #########################################################################################
    # Strong GBA / Synchronous
    #########################################################################################
    _, _, _, _, _, ress, riss = fig5_6.simulate(lnt = lnt, seed = 100, simtime = simtime, reg = 'sync', 
                                                gba = 'strong-gba', transient = trans, dt = dt, input_to_v1 = False, 
                                                use_default_noise=False, Ie_mean=310.0, Ii_mean=270.0)
