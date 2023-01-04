import pandas            as pd 
import numpy             as np
import matplotlib.pyplot as plt 
import scipy.io
import matplotlib

#plt.rcParams.update({'font.size': 24}) 


area_names = ['V1','V2','V4','DP','MT','8m','5','8l','TEO','2','F1','STPc','7A','46d','10','9/46v',\
        '9/46d','F5','TEpd','PBr','7m','7B','F2','STPi','PROm','F7','8B','STPr','24c']

def read_data(path='connectivity.pickle'):
	data = pd.read_pickle(path)
	fln_bin, sln, fln = data['fln_bin'], data['sln'], data['fln']
	#fln_bin = np.delete(fln_bin, 21, 0); fln_bin = np.delete(fln_bin, 21, 1)
	#sln     = np.delete(sln, 21, 0); sln = np.delete(sln, 21, 1) 
	#fln = np.delete(fln, 21, 0); fln = np.delete(fln, 21, 1) 
	return fln_bin, sln, fln

hierVals = scipy.io.loadmat('hierValspython.mat')
hierValsnew = hierVals['hierVals'][:]
netwParams_hier=hierValsnew/max(hierValsnew) # Hierarchy normalized. 

#fln values file 
flnMatp = scipy.io.loadmat('efelenMatpython.mat')
flnMat=flnMatp['flnMatpython'][:][:]         # FLN values..Cij is strength from j to i 
M = (flnMat > 0).astype(int)

distMatp  = scipy.io.loadmat('subgraphWiring29.mat')
distMat   = distMatp['wiring'][:][:]         # Distances between areas values..
delayMat  = distMat / 3.5


plt.figure()
lFLN = np.log(flnMat)
cmap = matplotlib.cm.hot
cmap.set_bad('black',1.0)
plt.figure()
plt.imshow(lFLN, aspect='auto', cmap=cmap)
plt.colorbar()
plt.title('log(FLN)')
plt.ylabel('to')
plt.xlabel('from')
plt.xticks(range(29), area_names, rotation=90)
plt.yticks(range(29), area_names)
plt.savefig('../figures/FLNmap.pdf', dpi=300)
plt.close()

plt.figure()
plt.imshow(distMat, aspect='auto', cmap='copper', vmin=0)
plt.colorbar()
plt.title('Wiring distance [mm]')
plt.ylabel('to')
plt.xlabel('from')
plt.xticks(range(29), area_names, rotation=90)
plt.yticks([])
plt.savefig('../figures/Dmap.pdf', dpi=300)
plt.close()

plt.figure(figsize=(6,2))
plt.plot(range(29),netwParams_hier, 'o')
plt.ylabel('Normalized Hierarchy')
plt.xticks(range(29), area_names, rotation = 90)
plt.savefig('../figures/hier.pdf', dpi=300)
plt.close()

'''
plt.subplot(1,2,1)
plt.imshow(fln, aspect='auto', cmap='jet', origin='lower')
plt.subplot(1,2,2)
plt.imshow(fln2, aspect='auto', cmap='jet', origin='lower')
plt.show()
'''

np.save('markov2014.npy', {'FLN': flnMat, 'Distances': distMat, 'Hierarchy': netwParams_hier})