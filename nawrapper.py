import numpy as np
import pymaster as nmt

class NaWrapper:
    """Decouple your modes in TT concisely."""
    
    def __init__(self, map1, map2, mask1, mask2, beam, lmax=6925,
                 nlb=None, binleft=None, binright=None, nside=None, is_Dell=True):
        """Compute fields and mode coupling matrix.

        The arguments map1, map2, mask1, and mask2 are arrays representing
        the maps and masks, and can be either enmap (default) or healpix. The
        maps and masks must all be either enmap or healpix!

        To use healpix maps, set nside to the correct number, instead of None.

        You can use custom bins by setting binleft and binright to be arrays
        containing the left and right bin edges. This function will check for
        if they are both not None.

        Parameters
        ----------
        map1 : array
            The first map you want to cross-correlate. 
        map2 : array
            The second map you want to cross-correlate. 
        mask1 : array
            The mask on the first map.
        mask2 : array
            The mask on the second map.
        beam : 1D array
            array containing the beam transfer function B_l.
        nlb : int
            size of ell bin
        binleft : 1D array
            left edge of ell bins
        binright : 1D array
            right edge of ell bins
        nside : int
            optional argument, set this to your map nside if you're using healpix
        is_Dell : bool
            flag for weighting by l(l+1) when binning
        """
        
        # bins
        if nlb is None:
            ells = np.arange(lmax)
            bpws=-1+np.zeros_like(ells) #Array of bandpower indices
            for i, (bl, br) in enumerate(zip(binleft[1:], binright[1:])):
                bpws[bl:br+1] = i
            weights = np.array([1.0 / np.sum(bpws == bpws[l]) for l in range(lmax)])
            b = nmt.NmtBin(2048, bpws=bpws, ells=ells,
                           weights=weights, lmax=lmax, is_Dell=is_Dell)
        else:
            b = nmt.NmtBin(2048, nlb=nlb)
        
        lb = b.get_effective_ells()
        
        niter = 0 # NaMaster-CAR only supports niter=0
        field0 = nmt.NmtField(mask1, [map1], beam=beam, wcs=mask1.wcs, n_iter=niter)
        field1 = nmt.NmtField(mask2, [map2], beam=beam, wcs=mask2.wcs, n_iter=niter)
        
        w0 = nmt.NmtWorkspace()
        w0.compute_coupling_matrix(field0,field1, b, n_iter=niter)
        
        # store references to everything
        cl_coupled = nmt.compute_coupled_cell(field0, field1)
        self.Cb =  w0.decouple_cell(cl_coupled)
        self.lb = b.get_effective_ells()
        self.mask1 = mask1
        self.mask2 = mask2
        self.w0 = w0
        self.b = b
        
    def decouple(self, map1, map2):
        niter = 0
        field0 = nmt.NmtField(self.mask1, [map1], 
                              beam=self.beam, wcs=self.mask1.wcs, n_iter=niter)
        field1 = nmt.NmtField(self.mask2, [map2], 
                              beam=self.beam, wcs=self.mask2.wcs, n_iter=niter)
        
        cl_coupled = nmt.compute_coupled_cell(field0, field1)
        return self.w0.decouple_cell(cl_coupled)