import kfilter_share as kfilter
from pixell import enmap, enplot

apopath = '../apo_mask/deep56_c7v5_car_190220_rect_master_apo_w0.fits'
mpath = '../maps/ACTPol_148_D56_pa1_f150_s14_4way_split0_srcadd_I.fits'
apo = enmap.read_map(apopath)
box = enmap.box(apo.shape,apo.wcs)
m = enmap.read_map(mpath).submap(box,mode='round')

kx = 90
ky = 50
kx_apo = 0
ky_apo = 0
unpixwin = 1
d_th = 1/120.

filtered = kfilter.get_map_kx_ky_filtered_pyfftw(m*apo,d_th,kx,kx_apo,ky,ky_apo,unpixwin=unpixwin,zero_pad=False)

filtered = enmap.samewcs(filtered,m)

enplot.write('raw_map',
	enplot.plot(m, mask=0,min=-300,max=300,ticks=5,downgrade=5))
enplot.write('filt_map',
	enplot.plot(filtered, mask=0,min=-150,max=150,ticks=5,downgrade=5))



