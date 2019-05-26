import numpy as np
import scipy
import matplotlib.pyplot as plt
from soapack import interfaces
from pixell import enmap
from pixell import enplot

def eshow(x,**kwargs): enplot.show(enplot.plot(x, downgrade=4,**kwargs))


# We seek to plot some ACTxACT, PlanckxPlanck, and ACTxPlanck cross spectra. Here we get the coadds from the two Planck half-missions.
def get_planck_coadd(freq, dmp):
    psplit0 = dmp.get_split(freq, splitnum=0, ncomp=1, srcfree=True)
    psplit1 = dmp.get_split(freq, splitnum=1, ncomp=1, srcfree=True)
    psplit0_i = dmp.get_split_ivar(freq, splitnum=0, ncomp=1, srcfree=True)
    psplit1_i = dmp.get_split_ivar(freq, splitnum=1, ncomp=1, srcfree=True)
    weighted = (psplit0_i * psplit0 + psplit1 * psplit1_i) / (psplit0_i + psplit1_i)
    weighted[~np.isfinite(weighted)] = 0.0
    return weighted


# Here is where we do a crappy job of estimating power spectra.

def bin(data,modlmap,bin_edges):
    digitized = np.digitize(np.ndarray.flatten(modlmap), bin_edges,right=True)
    return np.bincount(digitized,(data).reshape(-1))[1:-1]/np.bincount(digitized)[1:-1]

def compute_ps(map1, map2, mask, beamf1, beamf2):
    """Compute the FFTs, multiply, bin
    """
    kmap1 = enmap.fft(map1*mask, normalize="phys")
    kmap2 = enmap.fft(map2*mask, normalize="phys")
    power = (kmap1*np.conj(kmap2)).real
    
    bin_edges = np.arange(0,8000,40)
    centers = (bin_edges[1:] + bin_edges[:-1])/2.
    w2 = np.mean(mask**2.)
    modlmap = enmap.modlmap(map1.shape,map1.wcs)
    binned_power = bin(power/w2/beamf1(modlmap)/beamf2(modlmap),modlmap,bin_edges)
    return centers, binned_power


# # ACTxPlanck

# coadded ACT x coadded Planck

ACT_planck = {}

for patch in ['deep56', 'boss']:
    mask = interfaces.get_act_mr3_crosslinked_mask(patch)
    dma = interfaces.ACTmr3(region=mask)
    dmp = interfaces.PlanckHybrid(region=mask)
    # we loop over all pairs of Planck x ACT
    for planckfreq in ['030','044','070','100','143','217','353','545']: # no '857'
        planckbeam = lambda x: dmp.get_beam(x, planckfreq)
        planckmap = get_planck_coadd(planckfreq, dmp)[0,:,:]

        for actseason in ['s14','s15']:
            for array in ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150']:
                try:
                    actbeam = lambda x: dma.get_beam(x, actseason, 
                                           patch, array)
                    actmap = dma.get_coadd(actseason, 
                                           patch, array, ncomp=1, 
                                           srcfree=True)[0,:,:] # just want T
                    lb, Cb = compute_ps(planckmap, actmap, mask, planckbeam, actbeam)
                    ACT_planck[(patch, planckfreq, actseason, array)] = (lb, Cb)
                except (OSError,KeyError):
                    print("Can't find this ACT map:", actseason, array)



import pickle

with open('for_mat/ACT_planck.pickle', 'wb') as handle:
    pickle.dump(ACT_planck, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # Planck x Planck (different freqs)
# I use coadded planck x coadded planck

planck_planck = {}

for patch in ['deep56', 'boss']:
    mask = interfaces.get_act_mr3_crosslinked_mask(patch)
    dmp = interfaces.PlanckHybrid(region=mask)
    for planckfreq0 in ['030','044','070','100','143','217','353','545']:
        # we loop over all pairs of Planck x Planck
        planckbeam0 = lambda x: dmp.get_beam(x, planckfreq0)
        planckmap0 = get_planck_coadd(planckfreq0, dmp)[0,:,:]
        for planckfreq1 in ['030','044','070','100','143','217','353','545']:
            if float(planckfreq0) < float(planckfreq1):
                planckbeam1 = lambda x: dmp.get_beam(x, planckfreq1)
                planckmap1 = get_planck_coadd(planckfreq1, dmp)[0,:,:]
                lb, Cb = compute_ps(planckmap0, planckmap1, mask, planckbeam0, planckbeam1)
                planck_planck[(patch,planckfreq0, planckfreq1)] = (lb, Cb)


# # Planck x Planck (same freq)
# These are spectra for which $f_0 = f_1$, so I use half missions.


for patch in ['deep56', 'boss']:
    mask = interfaces.get_act_mr3_crosslinked_mask(patch)
    dmp = interfaces.PlanckHybrid(region=mask)
    for planckfreq in ['030','044','070','100','143','217','353','545']:
        # we loop over all pairs of Planck x Planck
        planckbeam = lambda x: dmp.get_beam(x, planckfreq)
        planckmap0 = dmp.get_split(planckfreq, splitnum=0, ncomp=1, srcfree=True)
        planckmap1 = dmp.get_split(planckfreq, splitnum=1, ncomp=1, srcfree=True)

        lb, Cb = compute_ps(planckmap0, planckmap1, mask, planckbeam, planckbeam)
        planck_planck[(patch,planckfreq, planckfreq)] = (lb, Cb)


# In[37]:


with open('for_mat/Planck_Planck.pickle', 'wb') as handle:
    pickle.dump(planck_planck, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:





# # ACT x ACT
# Different seasons/arrays - can use just the coadds

# In[19]:


act_act = {}

for patch in ['deep56', 'boss']:
    mask = interfaces.get_act_mr3_crosslinked_mask(patch)
    dma = interfaces.ACTmr3(region=mask)
    
    # we loop over all pairs of ACT x ACT
    for actseason0 in ['s14','s15']: # s13 doesn't have these patches
        for array0 in ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150']:
            
            for actseason1 in ['s14','s15']: # s13 doesn't have these patches
                for array1 in ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150']:
                    
                    if (actseason0 != actseason1 ) or (array0 != array1):
                        try:
                            actbeam0 = lambda x: dma.get_beam(x, actseason0, patch, array0)
                            actbeam1 = lambda x: dma.get_beam(x, actseason1, patch, array1)

                            actmap0 = dma.get_coadd(actseason0, patch, array0, 
                                                    ncomp=1, srcfree=True)[0,:,:] # just want T
                            actmap1 = dma.get_coadd(actseason1, patch, array1, 
                                                    ncomp=1, srcfree=True)[0,:,:] # just want T
                            lb, Cb = compute_ps(actmap0, actmap1, mask, actbeam0, actbeam1)
                            act_act[(patch, actseason0, array0, actseason1, array1)] = (lb, Cb)
                        except (OSError,KeyError):
                            print("Can't find this ACT map:", actseason0, array0, actseason1, array1)


# Same season, same array: cross spectra over the splits.

# In[23]:


import itertools


# In[33]:


nsplits = 4
for patch in ['deep56', 'boss']:
    mask = interfaces.get_act_mr3_crosslinked_mask(patch)
    dma = interfaces.ACTmr3(region=mask)
    
    # we loop over all pairs of ACT x ACT
    for actseason0 in ['s14','s15']: # s13 doesn't have these patches
        for array0 in ['pa1_f150', 'pa2_f150', 'pa3_f090', 'pa3_f150']:
            
            try:
                actbeam = lambda x: dma.get_beam(x, actseason0, patch, array0)
                actmaps = dma.get_splits(actseason0, patch, array0, 
                                            ncomp=1, srcfree=True)
                Cb_list = []
                lb_list = []
                for s0, s1 in itertools.combinations(range(nsplits),r=2):
                    actmap0 = actmaps[0, s0, 0, :, :]
                    actmap1 = actmaps[0, s1, 0, :, :]
                    lb, Cb = compute_ps(actmap0, actmap1, mask, actbeam, actbeam)
                    lb_list.append(lb)
                    Cb_list.append(Cb)
                Cb_list = np.array(Cb_list)
                act_act[(patch, actseason0, array0, actseason0, array0)] = (lb_list[0], 
                                                                            np.sum(Cb_list,axis=0)/Cb_list.shape[0])
            except (OSError,KeyError):
                print("Can't find this ACT map:", actseason0, array0, actseason1, array1)


# In[34]:


with open('for_mat/ACT_ACT.pickle', 'wb') as handle:
    pickle.dump(act_act, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




