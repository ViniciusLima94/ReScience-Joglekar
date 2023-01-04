# import NEST & NEST rasterplot
import nest
import nest.raster_plot
import sys
import scipy.io
import numpy            as np
import setParams

#########################################################################################
# Reading connectivity data
#########################################################################################
area_names = ['V1','V2','V4','DP','MT','8m','5','8l','TEO','2','F1','STPc','7A','46d','10','9/46v',\
        '9/46d','F5','TEpd','PBr','7m','7B','F2','STPi','PROm','F7','8B','STPr','24c']

data            = np.load('interareal/markov2014.npy', allow_pickle=True).item()
netwParams_hier = data['Hierarchy']
flnMat          = data['FLN']
M               = (flnMat > 0).astype(int)
delayMat        = data['Distances'] / 3.5

def simulate(lnt = 1, seed = 100, simtime = 1000.0, reg = 'async', gba = 'weak-gba', transient = 0, dt = 0.2, input_to_v1 = True, use_default_noise=True, Ie_mean=284.0, Ii_mean=294.0):

    
    # seed = 0 - Ruim; seed = 100 - bão (for lnt=20)

    #########################################################################################
    # Getting parameters
    #########################################################################################
    Nareas, NE, NI, N, alpha, tau_ex, tau_in, d, \
    p, R, sigma, exc_Cm, in_Cm, std_ex, std_in, \
    params, I_ext, inputs = setParams.get_params_spiking_model(dt,reg=reg,gba=gba)
    
    #########################################################################################
    # Configure NEST
    #########################################################################################
    # Configure kernel
    nest.ResetKernel()
    nest.SetKernelStatus({
        'resolution': dt,         # Set simulation resolution
        'print_time': True,       # Enable printing of simulation progress (-> terminal)
        'local_num_threads': lnt, # Use two threads to build & simulate the network
        'rng_seed' : seed
    })

    # Excitatory neurons
    exc_params = {
        'C_m': exc_Cm,         # Membrane capacity (pF)
        'E_L': -70.,           # Resting membrane potential (mV)
        'I_e': 0.,             # External input current (pA)
        'V_reset': -60.0,      # Reset membrane potential after a spike (mV)
        'V_th': -50.0,         # Spike threshold (mV)
        't_ref': 2.0,          # Refractory period (ms)
        'tau_m': tau_ex,       # Membrane time constant (ms)
    }

    # Inhibitory neurons
    in_params = {
        'C_m': in_Cm,          # Membrane capacity (pF)
        'E_L': -70.,           # Resting membrane potential (mV)
        'I_e': 0.,             # External input current (pA)
        'V_reset': -60.0,      # Reset membrane potential after a spike (mV)
        'V_th': -50.0,         # Spike threshold (mV)
        't_ref': 2.0,          # Refractory period (ms)
        'tau_m': tau_in,       # Membrane time constant (ms)
    }

    # set defaults for neuron models
    nest.CopyModel("iaf_psc_delta", "exc_iaf_psc_delta", exc_params)
    nest.CopyModel("iaf_psc_delta", "in_iaf_psc_delta", in_params)

    #########################################################################################
    # Create neurons and devices
    #########################################################################################

    # Creating neuronal populations
    pop_e = []
    pop_i = []
    for pop in range(Nareas):
        pop_e.append(nest.Create('exc_iaf_psc_delta', NE))
        pop_i.append(nest.Create('in_iaf_psc_delta',  NI))

    # Creating spike detectors
    spikes_e = nest.Create('spike_recorder')
    spikes_i = nest.Create('spike_recorder')

    #########################################################################################
    # Create white noise devices
    #########################################################################################
    xi_e = nest.Create('noise_generator')
    xi_i = nest.Create('noise_generator')

    if use_default_noise == True:

        nest.SetStatus(xi_e, [{'mean': I_ext[reg][gba]['Ie'],
                               'std': std_ex, 
                               'dt':0.1}])
        nest.SetStatus(xi_i, [{'mean': I_ext[reg][gba]['Ii'],
                               'std': std_in,
                               'dt':0.1}])

    else:

        nest.SetStatus(xi_e, [{'mean': Ie_mean,
                               'std': std_ex, 
                               'dt':0.1}])
        nest.SetStatus(xi_i, [{'mean': Ii_mean,
                               'std': std_in,
                               'dt':0.1}])

    #########################################################################################
    # Input current devices
    #########################################################################################
    if input_to_v1 == True:
        input_v1 = nest.Create('step_current_generator')
        nest.SetStatus(input_v1, [{'amplitude_times': [inputs[reg][gba]['t_on'], inputs[reg][gba]['t_off']],
                                   'amplitude_values':[inputs[reg][gba]['I'], 0.]}])
        nest.Connect(input_v1, pop_e[0])


    #########################################################################################
    # Setting weights
    #########################################################################################
    # Local synaptic weights wij (from j->i)
    wee, wie, wii, wei = params[reg][gba]['wee'], params[reg][gba]['wie'], params[reg][gba]['wii'], params[reg][gba]['wei']
    # Global synaptic weights muij (from j->i)
    muie, muee         = params[reg][gba]['muie'], params[reg][gba]['muee']

    #########################################################################################
    # Inter-areal weights parameters
    #########################################################################################
    pos, pre = np.where(M==1)
    W_ee_mat = np.zeros([Nareas, Nareas])
    W_ei_mat = np.zeros([Nareas, Nareas])
    for i,j in zip(pre,pos):  
        W_ee_mat[j,i] = (1.0 + alpha * netwParams_hier[j][0]) * muee * flnMat[j,i]
        W_ei_mat[j,i] = (1.0 + alpha * netwParams_hier[j][0]) * muie * flnMat[j,i]

    #########################################################################################
    # Create intra-areal connections
    #########################################################################################

    # Connection specification (test every pair with probability p)
    conn_exc = {'rule': 'pairwise_bernoulli', 'p': p}
    conn_inh = {'rule': 'pairwise_bernoulli', 'p': p}

    # Intra-connections
    for pop in range(Nareas):
        # Create excitatory connections
        nest.Connect(pop_e[pop], pop_e[pop], conn_exc, {'delay': d, 'weight': (1.0+alpha*netwParams_hier[pop][0])*wee})
        nest.Connect(pop_e[pop], pop_i[pop], conn_exc, {'delay': d, 'weight': (1.0+alpha*netwParams_hier[pop][0])*wie})

        # Create inhibitory connections
        nest.Connect(pop_i[pop], pop_e[pop], conn_inh, {'delay': d, 'weight': -wei})
        nest.Connect(pop_i[pop], pop_i[pop], conn_inh, {'delay': d, 'weight': -wii})

        # Connect spike detectors
        nest.Connect(pop_e[pop], spikes_e)
        nest.Connect(pop_i[pop], spikes_i)
        # Connecting noise
        nest.Connect(xi_e, pop_e[pop])
        nest.Connect(xi_i, pop_i[pop])

    #########################################################################################
    # Create inter-areal connections
    #########################################################################################
    for i,j in zip(pre,pos):
        nest.Connect(pop_e[i], pop_e[j], conn_exc, 
            {'delay': nest.random.normal(mean=delayMat[j,i], std=delayMat[j,i]*0.1), 
             'weight': W_ee_mat[j,i]})
        nest.Connect(pop_e[i], pop_i[j], conn_exc, 
            {'delay':nest.random.normal(mean=delayMat[j,i], std=delayMat[j,i]*0.1),
             'weight': W_ei_mat[j,i]})

    #########################################################################################
    # Set input and background
    #########################################################################################

    # Setting neurons' initial parameters
    for pop in range(Nareas):
        vinit_e = exc_params['E_L']#exc_params['E_L'] + np.random.rand(NE) * (exc_params['V_th'] - exc_params['E_L'])
        vinit_i = exc_params['E_L']#exc_params['E_L'] + np.random.rand(NI) * (exc_params['V_th'] - exc_params['E_L'])
        nest.SetStatus(pop_e[pop],{'I_e': 0.0})
        nest.SetStatus(pop_i[pop],{'I_e': 0.0})
        nest.SetStatus(pop_e[pop], 'V_m' , vinit_e)
        nest.SetStatus(pop_i[pop], 'V_m' , vinit_i)

    #########################################################################################
    # Simulate
    #########################################################################################
    nest.Simulate(simtime)

    # calculate mean firing rate in spikes per second
    rate_ex = np.sum(nest.GetStatus(spikes_e)[0]['events']['times']>=transient)/(simtime-transient)/(NE*Nareas)*1e3
    rate_in = np.sum(nest.GetStatus(spikes_i)[0]['events']['times']>=transient)/(simtime-transient)/(NI*Nareas)*1e3
    
    times_ex = nest.GetStatus(spikes_e)[0]['events']['times']
    times_in = nest.GetStatus(spikes_i)[0]['events']['times']

    index_ex = nest.GetStatus(spikes_e)[0]['events']['senders']
    index_in = nest.GetStatus(spikes_i)[0]['events']['senders']

    # Computing maximum frequency for each layer excitatory populations
    Na = [NE+NI]*Nareas
    Na = [0] + np.cumsum(Na).tolist()

    t_on  = 0#inputs[reg][gba]['t_on']
    t_off = simtime#inputs[reg][gba]['t_off'].
    max_fr = []

    for i in range(len(Na)-1):
        # Index of excitatory neurons for each population
        i_d, i_u = Na[i], Na[i+1]-NI
        # Get the spikes of excitatory neurons
        idx_ex = (index_ex>i_d)*(index_ex<i_u)
        t_ex   = times_ex[idx_ex]
        c, x   = np.histogram(t_ex, bins=np.arange(t_on, t_off,1))
        c      = c / (NE*1e-3)
        max_fr.append(c.max())

    return index_ex, index_in, times_ex, times_in, max_fr, rate_ex, rate_in
