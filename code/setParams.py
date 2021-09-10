#########################################################################################
# Parameters to use in the rate based and spikin neuron network simulations
#########################################################################################
import numpy    as np  

#########################################################################################
# Return the parameters for the rate based network model
#########################################################################################
def get_params_rate_model(gba='weak-gba'):
    params = {}

    # Time constants
    params['tau_ex'] = 20.0 # ms
    params['tau_in'] = 10.0 # ms
    # Scale for excitatory weights
    params['eta']    = 0.68
    # Intrinsic time constants
    params['beta_e'] = 0.066
    params['beta_i'] = 0.351
    # Local/Long-range circuit connection weights
    params['weights'] = {'wee':24.3, 'wie':12.2, 'wii':12.5, 'wei':19.7, 'muie':25.3, 'muee':33.7} # in pA/Hz
    # External inputs to excitatory and inhibitory populations
    params['I']       = {'exc': 10.0, 'inh': 35.0} # in Hz
    # Input to V1
    params['Iext']    = 22.05*1.9

    if gba == 'strong-gba':
        params['weights']['wei']  = 25.2
        params['weights']['muee'] = 51.5
        params['Iext']            = 11.54*1.9

    return params

#########################################################################################
# Return the parameters for the spiking neuron network model
#########################################################################################
def get_params_spiking_model(dt, reg = 'async', gba = 'weak-gba'):

    np.random.seed(10)

    #########################################################################################
    # Network parameters
    #########################################################################################
    Nareas = 29      # Number of populations (29 in reference)
    NE     = 1600    # Number of excitatory neurons in one population
    NI     = 400     # Number of inhibitory neurons in one population
    N      = NE+NI   # Total number of neurons
    p      = 0.1     # Connectivity density

    #########################################################################################
    # Synapse parameters
    #########################################################################################
    alpha  = 4.0        # So deus sabe
    tau_ex = 20.0       # Excitatory time constant
    tau_in = 10.0       # Inhibitory time constant
    d      = 2.0        # Intra-areal delay in ms

    params = {}
    params['async'] = {'weak-gba'  : {'wee':0.01, 'wie':0.075, 'wii':0.075, 'wei':0.0375, 'muie': 0.19/4, 'muee': 0.0375}, 
                       'strong-gba': {'wee':0.01, 'wie':0.075, 'wii':0.075, 'wei':0.0500, 'muie': 0.19/4, 'muee': 0.0500}
                       }
    params['sync']  = {'weak-gba'  : {'wee':0.04, 'wie':0.3, 'wii':0.3, 'wei':0.56, 'muie': 0.19, 'muee': 0.16}, 
                       'strong-gba': {'wee':0.04, 'wie':0.3, 'wii':0.3, 'wei':0.98, 'muie': 0.19, 'muee': 0.25}
                       }

    inputs = {}
    inputs['async'] = {'weak-gba': {'t_on': 500., 't_off': 650., 'I' : 300.}, 'strong-gba': {'t_on': 500., 't_off': 650., 'I' : 126.}}
    inputs['sync']  = {'weak-gba': {'t_on': 500., 't_off': 508., 'I' : 200.}, 'strong-gba': {'t_on': 500., 't_off': 508., 'I' : 200.}}

    #########################################################################################
    # Neuron parameters
    #########################################################################################
    R     = 50.0   # Membrane resistence in Mohm
    sigma = 2.12   # Noise variance   

    exc_Cm = (tau_ex/R)*1000.0  # tau_m(ms)/R_m(Mohm): excitatory capacitance in pF
    in_Cm  = (tau_in/R)*1000.0  # tau_m(ms)/R_m(Mohm): inhibitory capacitance in pF

    x_ex = np.exp(-dt/tau_ex)          
    x_in = np.exp(-dt/tau_in)       
    # Conversion sigma in mV to current: sigma * exc_Cm/tau_ex * np.sqrt( (1+x_ex)/(1-x_ex) )                                             
    std_ex = sigma * exc_Cm/tau_ex * np.sqrt( (1+x_ex)/(1-x_ex) )
    std_in = sigma * in_Cm/tau_in * np.sqrt( (1+x_in)/(1-x_in) )

    #########################################################################################
    # external input parameters
    #########################################################################################
    I_ext = {}
    #  I_ext['async'] = {'weak-gba': {'Ie' : (14.2/R)*1000, 'Ii': (14.7/R)*1000}, 'strong-gba': {'Ie' : (14.2/R)*1000, 'Ii': (14.7/R)*1000}}
    #  I_ext['sync']  = {'weak-gba': {'Ie' : (15.4/R)*1000, 'Ii': (14.0/R)*1000}, 'strong-gba': {'Ie' : (16.0/R)*1000, 'Ii': (14.0/R)*1000}}
    I_ext['async'] = {'weak-gba': {'Ie' : (14.2/R)*1000, 'Ii': (15.7/R)*1000}, 'strong-gba': {'Ie' : (14.2/R)*1000, 'Ii': (15.7/R)*1000}}
    I_ext['sync']  = {'weak-gba': {'Ie' : (15.4/R)*1000, 'Ii': (14.5/R)*1000}, 'strong-gba': {'Ie' : (16.0/R)*1000, 'Ii': (14.5/R)*1000}}

    return Nareas, NE, NI, N, alpha, tau_ex, tau_in, d, p, R, sigma, exc_Cm, in_Cm, std_ex, std_in, params, I_ext, inputs
