#########################################################################################
# Code to reproduce Figure 1 from the original paper 
# doi: https://doi.org/10.1016/j.neuron.2018.02.031
#########################################################################################

import nest
import matplotlib
import matplotlib.pyplot as plt
import numpy             as np

def simulate(lnt = 1, simtime = 600.0, dt = 0.2, transient = 0, tau = 20.0, Wee = 6.00, Wei = 6.70):

    #########################################################################################
    # Configure NEST kernel and parameters
    #########################################################################################
    nest.ResetKernel()
    nest.SetKernelStatus({
        'resolution': dt,          # Set simulation resolution
        'print_time': False,       # Enable printing of simulation progress (-> terminal)
        'local_num_threads': lnt,  # Use two threads to build & simulate the network
        'rng_seed' : lnt
    })
    nest.SetDefaults('threshold_lin_rate_ipn',  {'theta': 0.0, 
                                                 'alpha': 1e10,
                                                 'mu': 0.0, 
                                                 'lambda': 1.0,
                                                 'sigma': 0.0,
                                                 'linear_summation': True})


    #########################################################################################
    # Neuron parameters
    #########################################################################################
    tau_ex = tau_in = tau

    #########################################################################################
    # Synapsis parameters
    #########################################################################################
    #if LBA == 'strong':
    #   Wee, Wei = 6.00, 6.70
    #elif LBA == 'weak':
    #   Wee, Wei = 4.45, 4.70
    Wie, Wii = 4.29, 4.71

    # Excitatory population
    E = nest.Create('threshold_lin_rate_ipn', 1)
    # Inhibitory population
    I = nest.Create('threshold_lin_rate_ipn', 1)
    # Setting parameter for E/I populations
    nest.SetStatus(E, {'tau': tau_ex, 'g': 1.0, 'rate': 1.0, 'sigma': 0.0})
    nest.SetStatus(I, {'tau': tau_in, 'g': 1.0, 'rate': 0.0, 'sigma': 0.0})

    multi = nest.Create("multimeter")
    #  nest.SetStatus(multi, {"record_from":["rate"], 'withgid':True, 'interval':dt})
    nest.SetStatus(multi, {"record_from":["rate"], 'interval':dt})
    nest.Connect(multi, E)

    #########################################################################################
    # Initial input to set E0 = 1
    #########################################################################################
    #Iext = nest.Create('step_rate_generator')
    #nest.SetStatus(Iext, {'amplitude_times': [999., 1000.], 'amplitude_values': [tau+1.0 , 0.]}) 
    #conn = {'rule': 'one_to_one'}  
    #syn_e  = {'weight': 1.0, 'delay': 0.1,  'model': 'rate_connection_delayed'}
    #nest.Connect(Iext, E, conn, syn_e)

    #########################################################################################
    # Connecting populations
    #########################################################################################
    conn   = {'rule': 'one_to_one'}                        
    # E->E
    syn  = {'weight': Wee, 'synapse_model': 'rate_connection_instantaneous'}
    nest.Connect(E, E, conn, syn)
    # E->I
    syn  = {'weight': Wie, 'synapse_model': 'rate_connection_instantaneous'}
    nest.Connect(E, I, conn, syn)
    # I->E
    syn  = {'weight': -Wei, 'synapse_model': 'rate_connection_instantaneous'}
    nest.Connect(I, E, conn, syn)
    # I->I
    syn  = {'weight': -Wii, 'synapse_model': 'rate_connection_instantaneous'}
    nest.Connect(I, I, conn, syn)

    nest.Simulate(simtime)

    index = nest.GetStatus(multi)[0]['events']['senders']
    times = nest.GetStatus(multi)[0]['events']['times']
    rates = nest.GetStatus(multi)[0]['events']['rate']

    return times[int(transient/dt):]-times[int(transient/dt):].min(), rates[int(transient/dt):]

