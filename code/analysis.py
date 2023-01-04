import numpy             as np
import matplotlib.pyplot as plt
from   setParams         import *

# Total number of neurons
N = [NE+NI]*Nareas
# Cumulative sum of neurons in each region
N = [0] + np.cumsum(N).tolist()

t_on  = 500
t_off = 650 

def compute_max_freq(N, NE, NI, index_ex, times_ex, t_on, t_off):
    ############################################################
    # Compute the peak frequencie per region
    ###########################################################
    max_fr = []

    for i in range(len(N)-1):
        # Index of excitatory neurons for each population
        i_d, i_u = N[i], N[i+1]-NI
        # Get the spikes of excitatory neurons
        idx_ex = (index_ex>i_d)*(index_ex<i_u)
        t_ex   = times_ex[idx_ex]
        c, x   = np.histogram(t_ex, bins=np.arange(t_on, t_off,1))
        c      = c / (NE*1e-3)
        max_fr.append(c.max())

    return max_fr

#########################################################################################
# Measuring firing rate for each region
#########################################################################################
data     = np.load('data/weak-gba_data.npy', allow_pickle=True).item()
times_ex = data['times_ex']
index_ex = data['index_ex']

max_fr_w = compute_max_freq(N, NE, NI, index_ex, times_ex, t_on, t_off)

data     = np.load('data/strong-gba_data.npy', allow_pickle=True).item()
times_ex = data['times_ex']
index_ex = data['index_ex']

max_fr_s = compute_max_freq(N, NE, NI, index_ex, times_ex, t_on, t_off)

plt.figure(figsize=(9,3))
plt.semilogy(range(len(max_fr_w)), max_fr_w, 'o-', color = 'g', label='weak GBA')
plt.semilogy(range(len(max_fr_s)), max_fr_s, 'o-', color = 'm', label='strong GBA')

plt.legend()
plt.ylabel('Maximum firing rates [Hz]')
plt.xticks(range(len(max_fr_s)), area_names)
plt.tight_layout()
plt.savefig('figures/max_freq_async.png', dpi = 600)
plt.show()
