# coding: utf-8
'''
# G-MODE KPFM with Fast Free Force Recovery (F3R)
### Oak Ridge National Laboratory
### *Liam Collins, Anugrah Saxena, Rama Vasudevan and Chris Smith*
### *Edits Raj Giridharagopal, University of Washington*

This notebook will allow  fast KPFM by recovery of the electrostatic foce directly from the photodetector response. Information on the procedure can be found in Collins et al. ([DOI: 10.1021/acsnano.7b02114](http://pubs.acs.org/doi/abs/10.1021/acsnano.7b02114)) In this notebook the following procedured are performed. <br>

#### (1) Models the Cantilever Transfer Function (H(w))
**(1a)** Translates Tune file to H5 <br>
**(1b)** Fits Cantilever resonances to SHO Model <br>
**(1c)** Constructs the effective cantilever transfer function (H(w)) from SHO fits of the tune. <br>
#### (2)Load, Translate and Denoize the G-KPFM data
**(2a)** Loads and translates the .mat file containing the image data to .H5 file format. <br>
**(2b)** Fourier Filters data. <br>
**(2bii)** Checks Force Recovery for 1 pixel...here you need to find the phase offset used in 3. <br>
**(2c- optional)** PCA Denoizing.<br>

#### (3) Fast Free Force Reconstruction
**(3a)** Divides filtered displacement Y(w) by the effective transfer function (H(w)). <br>
<font color=red>This takes some time, can we parallelize it?. One option would be to incorperate it into the FFT filtering **step (2b** <br></font>
**(3b)** iFFT the response above a user defined noise floor to recovery Force in time domain.<br>
**(3c)** Phase correction (from step 2biii).<br>
<font color=red>I havent settled on the best way to find the phase shift required, but I see ye are working towards incorperating a phase shift into the filtering <br></font>

#### (4) Data Analysis
**(4a)** Parabolic fitting to extract CPD.<br>
<font color=red>Needs to be parallelized and output written to H5 file correctly. <br></font>

#### (5) Data Visualization
**(5a)** Visualization and clustering of fitting parameters and CPD.<br>
<font color=red>GIF movies and Kmeans clustering will be added. <br></font>

'''
#%%Change the filepath below, used for storing images

import os

output_filepath = r'E:\ORNL\20191221_BAPI\BAPI20_4ms_700mA__0007'
save_figure = True

# to automatically set light_on times
a = output_filepath.find('ms')
b = output_filepath.find('us')
if a != -1:
    tm = int(output_filepath[a-1])
    light_on_time = [1, 1+tm]  # ms   
elif b != -1:
    tm = int(output_filepath[b-3:b])
    light_on_time = [1, 1+tm/1000]  # ms
del(a)
del(b)

# Avoid prompts when loading data
pre_load_files = True

if pre_load_files is True:
    output_filepath = os.path.expanduser(output_filepath)
    idx = output_filepath.rfind("\\")
    data_file = os.path.join(output_filepath, output_filepath[idx+1:] + '_bigtime_00.dat')
    
    tune_path = os.path.abspath(r'E:\ORNL\20191221_BAPI\BAPI20_TUNE__0009')
    tune_path = os.path.expanduser(tune_path)
    idx = tune_path.rfind("\\")
    tune_file = os.path.join(tune_path, tune_path[idx+1:] + '_bigtime_00.dat')
    
    del(idx)

#%% Installing required packages

# Checks Python Version
import sys
if sys.version_info < (3, 5):
    print('''This notebook was optimized to work on Python 3.5.
    While it may also run on other Python versions,
    functionality and performance are not guaranteed
    Please consider upgrading your python version.''')

# ## Configure Notebook

'''Import necessary libraries'''
# Visualization:
import matplotlib.pyplot as plt

# General utilities:
import os
import sys
from scipy.signal import correlate
from scipy.optimize import curve_fit

# Interactive Value picker
import ipywidgets as widgets

# Computation:
import numpy as np
import numpy.polynomial.polynomial as npPoly

# Parallel computation library:
try:
    import joblib
except ImportError:
    warn('joblib not found.  Will install with pip.')
    import pip
    pip.main(['install', 'joblib'])
import joblib

import h5py

# multivariate analysis:
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# Finally, pycroscopy itself
import pycroscopy as px

# Define Layouts for Widgets
lbl_layout=dict(
    width='15%'
)
widget_layout=dict(
    width='15%',margin='0px 0px 5px 12px'
)
button_layout=dict(
    width='15%',margin='0px 0px 0px 5px'
)

#%% Define Cantilever Parameters
'''
Here you should input the calibrated parameters of the tip from your experiment.
In particular the lever sensitivity (m/V) and Spring Constant (N/m)
    (will be used to convert signals to displacement and force respectively)
'''

# 'k', 'invols', 'Thermal_Q', 'Thermal_resonance'
cantl_parms = dict()
cantl_parms['k'] = 1.7 # N/M
cantl_parms['invols'] = 82.76e-9 # m/V
cantl_parms['Thermal_Q'] = 80
cantl_parms['Thermal_res'] = 57076 #Hz

print(cantl_parms)

#%% Step 1) Model the Cantilever Transfer Function

'''
First we need to read in the tune file for the cantilever you used to perform your measurment with.
This tune show capture the "free" SHO parameters of the cantilever.
If you have previously translated this data you can change the data type in the bottom right corner to .h5, others click the parms file.txt
'''
#%% Step 1A) Translate Tune file to HF5 format

if pre_load_files is False:
    input_file_path = px.io_utils.uiGetFile(caption='Select translated .h5 file or tune data',
                                            file_filter='Parameters for raw G-Line tune (*.dat);; \
                                            Translated file (*.h5)')

    tune_path, _ = os.path.split(input_file_path)
    tune_file_base_name=os.path.basename(tune_path)
else:
    input_file_path = tune_file

i =1
if input_file_path.endswith('.dat'):
    print(i)
    print('Translating raw data to h5. Please wait')
    tran = px.GTuneTranslator()
    h5_path = tran.translate(input_file_path)
    print(input_file_path)
    
else:
    h5_path = input_file_path
    
#%% Step 1B) Extract the Resonance Modes Considered in the Force Reconstruction

#define number of eigenmodes to consider
num_bandsVal=2

#define bands (center frequency +/- bandwith)
center_freq = cantl_parms['Thermal_res']
#MB0_w1 = 57E3 - 20E3
#MB0_w2 = 57E3 + 20E3
#MB1_w1 = 370E3 - 20E3
#MB1_w2 = 370E3 + 20E3
MB0_w1 = center_freq - 20E3
MB0_w2 = center_freq + 20E3
MB1_w1 = center_freq*6.25 - 20E3
MB1_w2 = center_freq*6.25 + 20E3

MB1_amp = 30E-9
MB2_amp = 1E-9

MB_parm_vec = np.array([MB1_amp,MB0_w1,MB0_w2,MB1_amp,MB1_w1,MB1_w2])
MB_parm_vec.resize(2,3)
band_edge_mat = MB_parm_vec[:,1:3]

#%% Step 1B.i) Get response

# [0] and [1] are the DAQ channels, use HDFView for better understanding
hdf = px.ioHDF5(h5_path)
h5_file = hdf.file
h5_resp = px.hdf_utils.getDataSet(hdf.file, 'Raw_Data')[0]  # from tip
h5_main = px.hdf_utils.getDataSet(hdf.file, 'Raw_Data')[-1] # chirp to tip

parms_dict = h5_main.parent.parent.attrs

ex_freq = parms_dict['BE_center_frequency_[Hz]']
samp_rate = parms_dict['IO_rate_[Hz]']
N_points = parms_dict['num_bins']
N_points_per_line = parms_dict['points_per_line']
N_points_per_pixel = parms_dict['num_bins']

dt = 1/samp_rate #delta-time in seconds
df = 1/dt #delta-frequency in Hz

# Used in plotting
w_vec2 = np.linspace(-0.5*samp_rate,
                     0.5*samp_rate - 1.0*samp_rate / N_points_per_line,
                     N_points_per_line)

# Response
A_pd = np.mean(h5_resp, axis=0)
yt0_tune = A_pd - np.mean(A_pd)
Yt0_tune = np.fft.fftshift(np.fft.fft(yt0_tune,N_points_per_line)*dt)

# BE_wave_train
BE_pd = np.mean(h5_main, axis=0)
f0 = BE_pd - np.mean(BE_pd)
F0 = np.fft.fftshift(np.fft.fft(f0,N_points_per_line)*dt)

# The value here on right represents the excited bins
excited_bin_ind = np.where(np.abs(F0) > 0.01e-3)

# Transfer Function!
TF_vec = Yt0_tune/F0

#%% Step 1B.ii) Plot tune

plt.figure(2)
plt.subplot(2,1,1)
#plt.semilogy(np.abs(w_vec2[excited_bin_ind])*1E-6,
#             np.abs(TF_vec[excited_bin_ind]))

plt.semilogy(np.abs(w_vec2[excited_bin_ind])*1E-6,
             np.abs(TF_vec[excited_bin_ind]), 'r')
#plt.semilogy(np.abs(w_vec2[excited_bin_ind])*1E-6,
#             np.abs(Yt0_tune[excited_bin_ind]), 'b')
#plt.semilogy(np.abs(w_vec2[excited_bin_ind])*1E-6,
#             np.abs(F0[excited_bin_ind]), 'k')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude (a.u.)')
plt.xlim([band_edge_mat[0,0]*1e-6, band_edge_mat[0,1]*1e-6])
plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
plt.subplot(2,1,2)
plt.semilogy(np.abs(w_vec2[excited_bin_ind])*1E-6,
             np.angle(TF_vec[excited_bin_ind]))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Phase (Rad)')
plt.xlim([band_edge_mat[0,0]*1e-6, band_edge_mat[0,1]*1e-6])
#plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

#%% Step 1C) Construct an effective Transfer function (TF_Norm) from SHO fits

TunePhase = -np.pi
num_bands = band_edge_mat.shape[0]
coef_mat = np.zeros((num_bands,4))
coef_guess_vec = np.zeros((4))

# wb is an array of frequency points only where F0 above noise floor
wb = w_vec2[excited_bin_ind]

# Fit function for transfer function
TF_fit_vec = np.zeros((w_vec2.shape))
TFb_vec = TF_vec[excited_bin_ind]

# k1 = eigenmodes of cantilever to evaluate. Default = 2
Q_guesses = [120, 500, 700]

for k1 in range(num_bandsVal):
    
    # locate the fitting region
        # bin_ind1 is where band_edge is in the wb array
        # wbb is an array that spans this region for fitting purposes
    w1 = band_edge_mat[k1][0]
    w2 = band_edge_mat[k1][1]
    bin_ind1 = np.where(np.abs(w1-wb) == np.min(np.abs(w1-wb)))[0][0]
    bin_ind2 = np.where(np.abs(w2-wb) == np.min(np.abs(w2-wb)))[0][0]
    wbb = wb[bin_ind1:bin_ind2+1].T/1e6

    response_vec = TFb_vec[bin_ind1:bin_ind2+1].T
    response_mat = np.array([np.real(response_vec), np.imag(response_vec)]).T
    
    # initial guesses    
    A_max_ind = np.argmax(np.abs(response_vec))
    A_max = response_vec[A_max_ind]
    Q_guess = Q_guesses[k1]
    A_guess = A_max/Q_guess
    wo_guess = wbb[A_max_ind]
    phi_guess = TunePhase
    coef_guess_vec = [np.real(A_guess),
                      wo_guess,
                      Q_guess,
                      phi_guess]

#    LL_vec = [0, w1/1E6,1, np.pi]
#    UL_vec = [np.inf, w2/1E6, 10000, -np.pi]

    coef_vec = px.be_sho.SHOestimateGuess(response_vec, wbb, 10)

    response_guess_vec = px.be_sho.SHOfunc(coef_guess_vec, wbb)
    response_fit_vec = px.be_sho.SHOfunc(coef_vec, wbb)

    # Saves the response ni MHz, not used anywhere else
    coef_vec[1] = coef_vec[1]*1E6 #convert to MHz
    coef_mat[k1,:] = coef_vec
    response_fit_full_vec = px.be_sho.SHOfunc(coef_vec,w_vec2)
    TF_fit_vec = TF_fit_vec + response_fit_full_vec # check for length and dimension

    # Plot: blue = data, green = initial guess, red = fit
    fig= plt.figure(10, figsize=(9,9))
    plt.subplot(num_bands,2,k1+1)
    plt.plot(wbb,np.abs(response_vec),'.-')
    plt.plot(wbb,np.abs(response_guess_vec),c='g')
    plt.plot(wbb,np.abs(response_fit_vec),c='r')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Amplitude (nm)')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    #plt.figure(11)
    plt.subplot(num_bands,2,(k1+1)+2)
    plt.plot(wbb,np.angle(response_vec),'.-')
    plt.plot(wbb,np.angle(response_guess_vec),'g')
    plt.plot(wbb,np.angle(response_fit_vec),'r')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Phase (Rad)')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)


    if save_figure == True:
        fig.savefig(output_filepath+'\SHOFitting.eps', format='eps')
        fig.savefig(output_filepath+'\SHOFitting.tif', format='tiff')

Q = coef_mat[0,2]
TF_norm = ((TF_fit_vec- np.min(np.abs(TF_fit_vec)))/ np.max(np.abs(TF_fit_vec))-
           np.min(np.abs(TF_fit_vec))) * Q

#%% Separate close file to allow debugging without errors
hdf.close()

################# TUNE ANALYSIS DONE #######################

#%% Step 2) Load, Translate and Denoize the G-KPFM data

#Step 2A) Load and Translates image file to .H5 file format
pre_load_files = False
if pre_load_files is False:
    input_file_path = px.io_utils.uiGetFile(caption='Select translated .h5 file or raw experiment data',
                                            file_filter='Parameters for raw G-Line data (*.dat);; \
                                            Translated file (*.h5)')
else:
    input_file_path = data_file
    
folder_path, _ = os.path.split(input_file_path)

if input_file_path.endswith('.dat'):
    print('Translating raw data to h5. Please wait')
    tran = px.GLineTranslator()
    h5_path = tran.translate(input_file_path)
else:
    h5_path = input_file_path

#%% Step 2A.i) Extract some relevant parameters

hdf = px.ioHDF5(h5_path)

# Getting ancillary information and other parameters
h5_main = px.hdf_utils.getDataSet(hdf.file,'Raw_Data')[0]
h5_spec_vals = px.hdf_utils.getAuxData(h5_main, auxDataName='Spectroscopic_Values')[0]
h5_spec_inds=px.hdf_utils.getAuxData(h5_main, auxDataName='Spectroscopic_Indices')[0]

# General parameters
parms_dict = h5_main.parent.parent.attrs
samp_rate = parms_dict['IO_rate_[Hz]']
ex_freq = parms_dict['BE_center_frequency_[Hz]']
num_rows = parms_dict['grid_num_rows']
num_cols = parms_dict['grid_num_cols']
h5_pos_vals=px.hdf_utils.getAuxData(h5_main, auxDataName='Position_Values')[0]
h5_pos_inds=px.hdf_utils.getAuxData(h5_main, auxDataName='Position_Indices')[0]
num_pts = h5_main.shape[1]
pnts_per_pix=int(num_pts/num_cols)

N_points = parms_dict['num_bins']
N_points_per_pixel = parms_dict['num_bins']
time_per_osc = (1/parms_dict['BE_center_frequency_[Hz]'])
IO_rate = parms_dict['IO_rate_[Hz]']     #sampling_rate
pnts_per_period = IO_rate * time_per_osc #points per oscillation period
pxl_time = N_points_per_pixel/IO_rate    #seconds per pixel
num_periods = int(pxl_time/time_per_osc) #total # of periods per pixel, should be an integer

# Excitation waveform for a single pixel
pixel_ex_wfm = h5_spec_vals[0, :int(h5_spec_vals.shape[1]/num_cols)]

# Excitation waveform for a single line / row of data
excit_wfm = h5_spec_vals.value

# Preparing the frequency axis:
w_vec = 1E-3*np.linspace(-0.5*samp_rate, 0.5*samp_rate - samp_rate/num_pts, num_pts)
w_vec_pix = 1E-3*np.linspace(-0.5*samp_rate, 0.5*samp_rate - samp_rate/pnts_per_pix, pnts_per_pix)

# Preparing the time axis:
t_vec_line = 1E3*np.linspace(0, num_pts/samp_rate, num_pts)
t_vec_pix = 1E3*np.linspace(0, pnts_per_pix/samp_rate, pnts_per_pix)

#%% Step 2B) Fourier Filter data
'''
Define filter parameters in first cell
Then test on a single row
Finally perform on full dataset

**Here you can play with Noise tolerance**
'''

# Set Filter parameters here:
num_spectral_pts = h5_main.shape[1]
#hpf = px.processing.fft.HarmonicPassFilter(num_pts, samp_rate, ex_freq, 1E+3, 10)

#default filtering, note the bandwidths --> DC filtering and certain noise peaks
lpf = px.processing.fft.LowPassFilter(num_pts, samp_rate, 200E+3)
nbf = px.processing.fft.NoiseBandFilter(num_pts, samp_rate, 
                                        [5E3, 50E3, 100E3, 150E3, 200E3],
                                        [10E3, 1E3, 1E3, 1E3, 1E3])

 #no DC filtering
#nbf = px.processing.fft.NoiseBandFilter(num_pts, samp_rate, 
#                                        [50E3, 100E3, 125E3],
#                                        [1E3, 1E3, 1.5E3])

freq_filts = [lpf, nbf]
noise_tolerance = .000005

# Test filter on a single line:
row_ind = 40
filt_line, fig_filt, axes_filt = px.processing.gmode_utils.test_filter(h5_main[row_ind],
                                                                       frequency_filters=freq_filts,
                                                                       noise_threshold=noise_tolerance,
                                                                       show_plots=True)

#filt_line, fig_filt, axes_filt = px.processing.gmode_utils.test_filter(h5_filt[row_ind],
#                                                                       frequency_filters=freq_filts,
#                                                                       noise_threshold=noise_tolerance,
#                                                                       show_plots=True)

if save_figure == True:
    fig = fig_filt
    fig.savefig(output_filepath+'\FFTFiltering.eps', format='eps')
    fig.savefig(output_filepath+'\FFTFiltering.tif', format='tiff')

filt_row = filt_line.reshape(-1, pixel_ex_wfm.size)
fig, axes = px.plot_utils.plot_loops(pixel_ex_wfm, filt_row,use_rainbow_plots=True, 
                                     x_label='Bias (V)', title='FFT Filtering',
                                     plots_on_side=4, y_label='Deflection (a.u.)')

#%% Step 2B.i) Testing F3R and finding phase on the Filtered row data from previous step
'''
We need to find the phase offset between the measured response and drive voltage.
Adjust phase to close the parabola in the second set of images

This segment does two things:
    1) Tests whether the above filters are effective (FFT plot) and shows the result
    2) Tests is the phase-offset is correct to account for cable pathlenghts
'''
# Try Force Conversion on Filtered data

# Phase Offset
ph = -0.30    # phase from cable delays between excitation and response

# Calculates NoiseLimit
fft_h5row = np.fft.fftshift(np.fft.fft(h5_main[row_ind]))
noise_floor = px.processing.fft.getNoiseFloor(fft_h5row, noise_tolerance)[0]
print('Noise floor = ', noise_floor)
Noiselimit = np.ceil(noise_floor)

# Try Force Conversion on Filtered data of single line (row_ind above)
G_line = np.zeros(w_vec2.size,dtype=complex)         # G = raw
G_wPhase_line = np.zeros(w_vec2.size,dtype=complex)  # G_wphase = phase-shifted

signal_ind_vec = np.arange(w_vec2.size)
ind_drive = (np.abs(w_vec2-ex_freq)).argmin()

# filt_line is from filtered data above
test_line = filt_line-np.mean(filt_line)
test_line = np.fft.fftshift(np.fft.fft(test_line))
signal_kill = np.where(np.abs(test_line) < Noiselimit)
signal_ind_vec = np.delete(signal_ind_vec, signal_kill)

# Original/raw data; TF_norm is from the Tune file transfer function
G_line[signal_ind_vec] = test_line[signal_ind_vec]
G_line = (G_line/TF_norm)
G_time_line = np.real(np.fft.ifft(np.fft.ifftshift(G_line))) #time-domain 

# Phase-shifted data
test_shifted = (test_line)*np.exp(-1j*w_vec2/(w_vec2[ind_drive])*ph)
G_wPhase_line[signal_ind_vec] = test_shifted[signal_ind_vec]
G_wPhase_line = (G_wPhase_line/TF_norm)
G_wPhase_time_line = np.real(np.fft.ifft(np.fft.ifftshift(G_wPhase_line)))

# On a single line, row_ind is above in previous cell
FRaw_resp = np.fft.fftshift(np.fft.fft(h5_main[row_ind]))

#%%
# Plotting F3R
fig, ax = plt.subplots(figsize=(12, 7))
plt.semilogy(w_vec, (np.abs(FRaw_resp)), '^b' ,label='Response')
#plt.semilogy(w_vec[signal_ind_vec], (np.abs(G[signal_ind_vec])), 'og')
plt.semilogy(w_vec[signal_ind_vec], (np.abs(FRaw_resp[signal_ind_vec])),'.r', label='F3r')
plt.semilogy(w_vec[signal_ind_vec], np.abs(TF_norm[signal_ind_vec]), 'k', label='Tune TF')
ax.set_xlabel('Frequency (kHz)', fontsize=16)
ax.set_ylabel('Amplitude (a.u.)', fontsize=16)
ax.legend(fontsize=14)
ax.set_yscale('log')
ax.set_xlim(0, 200)
ax.set_title('Noise Spectrum for row ' + str(row_ind), fontsize=16)
px.plot_utils.set_tick_font_size(ax, 14)

# In time domain again, compare pre/post-phase-shift-corrected versions
unshifted = G_time_line.reshape(-1, pixel_ex_wfm.size)
phaseshifted = G_wPhase_time_line.reshape(-1, pixel_ex_wfm.size)

# Unshifted phase, pre-FFT filter
raw = np.real(np.fft.ifft(np.fft.ifftshift(FRaw_resp)))
raw = raw.reshape(-1, pixel_ex_wfm.size)

#fig, axes = px.plot_utils.plot_loops(pixel_ex_wfm, raw, use_rainbow_plots=True, 
#                                     x_label='Voltage (Vac)', title='Raw',
#                                     plots_on_side=2, y_label='Deflection (a.u.)')


# Unshifted phases, post-FFT filter
fig, axes = px.plot_utils.plot_loops(pixel_ex_wfm, unshifted, use_rainbow_plots=True, 
                                     x_label='Voltage (Vac)', title='Raw',
                                     plots_on_side=2, y_label='Deflection (a.u.)')

# Shifted phase; ideally parabolas should overlap
fig, axes = px.plot_utils.plot_loops(pixel_ex_wfm, phaseshifted, use_rainbow_plots=True, 
                                     x_label='Voltage (Vac)', title='Phase Shifted',
                                     plots_on_side=2, y_label='Deflection (a.u.)')

fig.savefig(output_filepath+r'\PostFilter_Displacements.tif', format='tiff')


#%% Filter the full data set; this process is quite slow

h5_filt_grp = px.hdf_utils.check_for_old(h5_main, 'FFT_Filtering')#, new_parms=filter_parms)

if h5_filt_grp == None:
    
    sig_filt = px.processing.SignalFilter(h5_main, frequency_filters=freq_filts, 
                                          noise_threshold=noise_tolerance,
                                          write_filtered=True, write_condensed=False, 
                                          num_pix=1,verbose=True, cores=2, max_mem_mb=512)
    h5_filt_grp = sig_filt.compute()
    
else:
    print('Taking previously computed results')

h5_filt = h5_filt_grp['Filtered_Data']

# Reshapes the filtered response into a matrix per-pixel instead of in lines (as recorded by NI box)

print('\n','========= Done! Now reshaping... =========')
h5_main_filt = px.hdf_utils.getDataSet(hdf.file,'Filtered_Data')[0]

scan_width=1
h5_resh = px.processing.gmode_utils.reshape_from_lines_to_pixels(h5_filt, pixel_ex_wfm.size,
                                                                 scan_width / num_cols)
h5_resh_grp = h5_resh.parent
h5_resh.shape

#%% Does PCA on the filtered response

h5_svd = px.processing.svd_utils.SVD(h5_resh, num_components=256)
h5_svd_group = h5_svd.compute()

h5_U = h5_svd_group['U']
h5_V = h5_svd_group['V']
h5_S = h5_svd_group['S']

# Since the two spatial dimensions (x, y) have been collapsed to one, we need to reshape the abundance maps:
# The "25" is how many of the eigenvectors to keep
abun_maps = np.reshape(h5_U[:,:25], (num_rows, num_cols,-1))

# Visualize the variance / statistical importance of each component:
fig, axes =px.plot_utils.plot_scree(h5_S, title='Skree plot')

if save_figure == True:
    fig.savefig(output_filepath+'\PCARaw_Skree.eps', format='eps')
    fig.savefig(output_filepath+'\PCARaw_Skree.tif', format='tiff')

# Visualize the eigenvectors; 
first_evecs = h5_V[:9, :]

fig, axes =px.plot_utils.plot_loops(pixel_ex_wfm, first_evecs, use_rainbow_plots=True, 
                                    x_label='Voltage (Vac)', y_label='Displacement (a.u.)', 
                                    plots_on_side=3, subtitle_prefix='Component', 
                                    title='SVD Eigenvectors (F3R)', evenly_spaced=False)

if save_figure == True:
    fig.savefig(output_filepath+'\PCARaw_Eig.eps', format='eps')
    fig.savefig(output_filepath+'\PCARaw_Eig.tif', format='tiff')

# Visualize the abundance maps:
fig, axes =px.plot_utils.plot_map_stack(abun_maps, num_comps=9, heading='SVD Abundance Maps',
                                        color_bar_mode='single', cmap='inferno')

if save_figure == True:
    fig.savefig(output_filepath+r'\PCARaw_Loading.eps', format='eps')
    fig.savefig(output_filepath+r'\PCARaw_Loading.tif', format='tiff')

#%% PCA_Clean prior to F3R Reconstruction?
PCA_pre_reconstruction_clean = False

# Filters out the components specified from h5_resh (the reshaped h5 data)
if PCA_pre_reconstruction_clean == True:
    
    clean_components = np.array([0,1,2,3,4, 5, 6, 7,8]) # np.append(range(5,9),(17,18))
    test = px.svd_utils.rebuild_svd(h5_resh, components=clean_components)
    PCA_clean_data_prerecon = test[:,:].reshape(num_rows,-1)

#%% Step 3) Fast Free Force Reconstruction
'''
Below we perform fast free recovery of the electrostatic force by
dividing the filtered response by the effective transfer function.

We further set a noise treshold, above which is included in the iFFT transform into the time domain
'''
#%% Step 3A) Divide Filtered displacement Y(w) by effective transfer function H(w)

# Divide Image-data h5_main by Tune-data TF_norm
ind_drive = (np.abs(w_vec2-ex_freq)).argmin()

G = np.zeros(w_vec2.size,dtype=complex)
G_time = np.zeros(shape=h5_filt.shape, dtype=h5_filt.dtype)

signal_ind_vec = np.arange(w_vec2.size)

NoiseLimit = np.ceil(noise_floor)

#for i in range(num_rows):
for i in np.array([50]):
    
    print('Row = ', i)
    row_ind = i
    filt_line, fig_filt, axes_filt = px.processing.gmode_utils.test_filter(h5_main[row_ind],
                                                                           frequency_filters=freq_filts,
                                                                           noise_threshold=noise_tolerance,
                                                                           show_plots=False)

    signal_ind_vec=np.arange(w_vec2.size)
    
    G = np.zeros(w_vec2.size,dtype=complex)         # G = raw
    G_line = np.zeros(w_vec2.size,dtype=complex)         # G = raw
    G_wPhase_line = np.zeros(w_vec2.size,dtype=complex)  # G_wphase = phase-shifted
    
    signal_ind_vec = np.arange(w_vec2.size)
    ind_drive = (np.abs(w_vec2-ex_freq)).argmin()


    # Step 3B) Phase correction; ph value is defined way above in Step 2B.i
    
    if PCA_pre_reconstruction_clean == True:
        test_data = PCA_clean_data_prerecon[i,:] - np.mean(PCA_clean_data_prerecon[i,:])   
    else:
        test_data = h5_filt[i,:] - np.mean(h5_filt[i,:])
  
#    print('Test data: ', test_data)
#    print('Filt Line: ', filt_line - np.mean(filt_line))
    
    # filt_line is from filtered data above  
    test_data = np.fft.fftshift(np.fft.fft(test_data))
    signal_kill = np.where(np.abs(test_data) < NoiseLimit)
    signal_ind_vec = np.delete(signal_ind_vec,signal_kill)
    test_data_ph = (test_data) * np.exp(-1j*w_vec2/(w_vec2[ind_drive])*ph)

    # Step 3C)  iFFT the response above a user defined noise floor to recover Force in time domain.
    G[signal_ind_vec] = test_data_ph[signal_ind_vec]
    G = G/TF_norm
    G_time[i,:] = np.real(np.fft.ifft(np.fft.ifftshift(G)))
    G_time_debug_ph = G_time[i,:]
    
    G[signal_ind_vec] = test_data[signal_ind_vec]
    G = G/TF_norm
    G_time_debug = np.real(np.fft.ifft(np.fft.ifftshift(G)))
    
    ##
    signal_ind_vec = np.arange(w_vec2.size)
    ind_drive = (np.abs(w_vec2-ex_freq)).argmin()
    
    # filt_line is from filtered data above
    test_line = filt_line-np.mean(filt_line)
    test_line = np.fft.fftshift(np.fft.fft(test_line))
    signal_kill = np.where(np.abs(test_line) < Noiselimit)
    signal_ind_vec = np.delete(signal_ind_vec, signal_kill)
    
    # Original/raw data; TF_norm is from the Tune file transfer function
    G_line[signal_ind_vec] = test_line[signal_ind_vec]
    G_line = (G_line/TF_norm)
    G_time_line = np.real(np.fft.ifft(np.fft.ifftshift(G_line))) #time-domain 
    
    # Phase-shifted data
    test_shifted = (test_line)*np.exp(-1j*w_vec2/(w_vec2[ind_drive])*ph)
    G_wPhase_line[signal_ind_vec] = test_shifted[signal_ind_vec]
    G_wPhase_line = (G_wPhase_line/TF_norm)
    G_wPhase_time_line = np.real(np.fft.ifft(np.fft.ifftshift(G_wPhase_line)))
    ##  
    print('G_wPhase_time_line: ', G_wPhase_time_line)
    print('G_time phase shift: ', G_time_debug_ph)

    print('G_time_line: ', G_time_line)
    print('G_time: ', G_time_debug)

    FRaw_resp = np.fft.fftshift(np.fft.fft(h5_main[i]))

# Saves as backup in Python for later analysis
if PCA_pre_reconstruction_clean == False:
    G_time_noPCA = G_time
else:
    G_time_PCA = G_time

fig, ax = plt.subplots(figsize=(12, 7))
plt.semilogy(w_vec, (np.abs(FRaw_resp)), label='Response')
plt.semilogy(w_vec[signal_ind_vec], (np.abs(G[signal_ind_vec])), 'og')
plt.semilogy(w_vec[signal_ind_vec], (np.abs(FRaw_resp[signal_ind_vec])),'.r', label='F3r')
ax.set_xlabel('Frequency (kHz)', fontsize=16)
ax.set_ylabel('Amplitude (a.u.)', fontsize=16)
ax.legend(fontsize=14)
ax.set_yscale('log')
ax.set_xlim(0, 200)
ax.set_title('Noise Spectrum for row ' + str(i), fontsize=16)
px.plot_utils.set_tick_font_size(ax, 14)

if save_figure == True:
    if PCA_pre_reconstruction_clean == False:
        fig.savefig(output_filepath+r'\Noise_Spectra_noPCA.eps', format='eps')
        fig.savefig(output_filepath+r'\Noise_Spectra_noPCA.tif', format='tiff')
    else:
        fig.savefig(output_filepath+r'\Noise_Spectra_PCA.eps', format='eps')
        fig.savefig(output_filepath+r'\Noise_Spectra_PCA.tif', format='tiff')        

phaseshifted = G_time[i].reshape(-1, pixel_ex_wfm.size)
fig, axes = px.plot_utils.plot_loops(pixel_ex_wfm, phaseshifted, use_rainbow_plots=True, 
                                     x_label='Voltage (Vac)', title='Phase Shifted',
                                     plots_on_side=2, y_label='Deflection (a.u.)')

if PCA_pre_reconstruction_clean == False:
    fig.savefig(output_filepath+r'\PostFilter_Displacement_noPCA.tif', format='tiff')
else:
    fig.savefig(output_filepath+r'\PostFilter_Displacement_PCA.tif', format='tiff')

#%% Reshaping and Storing  Results

# copies h5_filt over to H5_F3R; if necessary can come back here to reprocess
h5_F3R = px.hdf_utils.create_empty_dataset(source_dset=h5_filt,
                                           dtype=h5_filt.dtype,
                                           dset_name='h5_F3R',
                                           new_attrs=dict(),
                                           skip_refs=False)
px.hdf_utils.copy_main_attributes(h5_filt, h5_F3R)
h5_F3R[:,:] = G_time[:,:]
h5_F3R.file.flush()

px.hdf_utils.link_as_main(h5_main=h5_F3R, h5_pos_inds=h5_pos_inds,
                          h5_pos_vals=h5_pos_vals, h5_spec_inds=h5_spec_inds,
                          h5_spec_vals=h5_spec_vals, anc_dsets=[])

h5_F3rresh_grp = px.hdf_utils.findH5group(h5_F3R, 'Reshape')

scan_width = 1
h5_F3rresh = px.processing.gmode_utils.reshape_from_lines_to_pixels(h5_F3R, pixel_ex_wfm.size, scan_width / num_cols)
h5_F3rresh_grp = h5_F3rresh.parent

print('Data was reshaped from shape', h5_F3R.shape,
      'reshaped to ', h5_F3rresh.shape)

raw = np.reshape(h5_F3rresh, [-1, pixel_ex_wfm.size])
fig, axes = px.plot_utils.plot_loops(pixel_ex_wfm, raw[128:256],use_rainbow_plots=True, 
                                     x_label='Voltage (Vac)', title='Raw',
                                     plots_on_side=2, y_label='Deflection (a.u.)')

#%% Do PCA on F3R recovered data

# SVD and save results
h5_svd = px.processing.svd_utils.SVD(h5_F3rresh, num_components=256)
h5_svd_group = h5_svd.compute()

h5_u = h5_svd_group['U']
h5_v = h5_svd_group['V']
h5_s = h5_svd_group['S']

# Since the two spatial dimensions (x, y) have been collapsed to one, we need to reshape the abundance maps:
abun_maps = np.reshape(h5_u[:,:25], (num_rows, num_cols,-1))

#%% Visualize the variance / statistical importance of each component:
fig, axes =px.plot_utils.plot_scree(h5_s, title='Skree plot')

if save_figure == True:
    if PCA_pre_reconstruction_clean == False:
        fig.savefig(output_filepath+'\PCAF3R_Skree_noPrePCA.eps', format='eps')
        fig.savefig(output_filepath+'\PCF3R_Skree_noPrePCa.tif', format='tiff')
    else:
        fig.savefig(output_filepath+'\PCAF3R_Skree_withPrePCA.eps', format='eps')
        fig.savefig(output_filepath+'\PCF3R_Skree_withPrePCA.tif', format='tiff')


# Visualize the eigenvectors:
first_evecs = h5_v[:9, :]

fig, axes =px.plot_utils.plot_loops(pixel_ex_wfm, first_evecs, x_label='Voltage (Vac)', use_rainbow_plots=True, y_label='Displacement (a.u.)', plots_on_side=3,
                         subtitle_prefix='Component', title='SVD Eigenvectors (F3R)', evenly_spaced=False)

if save_figure == True:
    if PCA_pre_reconstruction_clean == False:
        fig.savefig(output_filepath+'\PCAF3R_Eig_noPrePCA.eps', format='eps')
        fig.savefig(output_filepath+'\PCF3R_Eig_noPrePCA.tif', format='tiff')
    else:
        fig.savefig(output_filepath+'\PCAF3R_Eig_withPrePCA.eps', format='eps')
        fig.savefig(output_filepath+'\PCF3R_Eig_withPrePCA.tif', format='tiff')

# Visualize the abundance maps:
fig, axes =px.plot_utils.plot_map_stack(abun_maps, num_comps=9, heading='SVD Abundance Maps',
                             color_bar_mode='single', cmap='inferno')
if save_figure == True:
    if PCA_pre_reconstruction_clean == False:
        fig.savefig(output_filepath+'\PCAF3R_Loadings_noPrePCA.eps', format='eps')
        fig.savefig(output_filepath+'\PCF3R_Loadings_noPrePCA.tif', format='tiff')
    else:
        fig.savefig(output_filepath+'\PCAF3R_Loadings_withPrePCA.eps', format='eps')
        fig.savefig(output_filepath+'\PCF3R_Loadings_withPrePCA.tif', format='tiff')


#%% Here you can PCA clean data if you like
PCA_post_reconstruction_clean = True

if PCA_post_reconstruction_clean == True:
    clean_components = np.array([0,1, 2, 3]) ##Components you want to keep
    #num_components = len(clean_components)

    #test = px.svd_utils.rebuild_svd(h5_F3rresh, components=num_components)
    test = px.svd_utils.rebuild_svd(h5_F3rresh, components=clean_components)
    PCA_clean_data_postrecon = test[:,:].reshape(num_rows*num_cols,-1)

#%% Test fitting on sample data

# This is number of periods you want to average over,
# for best time resolution =1 (but takes longer to fit)
periods = 4
 
num_periods_per_sample = int(np.floor(num_periods / periods))
pnts_per_sample = int(np.floor(pnts_per_period * periods))


# new approach since it's base-2 samples and can curve-fit to less than full cycle
decimation = 2**int(np.floor(np.log2(pnts_per_sample)))
pnts_per_CPDpix = int(N_points_per_pixel/decimation)
tx = np.linspace(0, pxl_time, pnts_per_CPDpix) # time-axis

deg = 2
m = 2   #random sample pixel
k4 = 3 #random oscillation in that pixel
# note k4 cannot exceed Npoints_per_pixel/periods, obviously

##Raw F3R response

'''Use one of the resp functions below, either cleaned or not (comment as needed)'''
# Use PCA clean or not
if PCA_post_reconstruction_clean == False:
    print('Not post-filtered')
    resp = h5_F3rresh[m][pnts_per_CPDpix*k4:pnts_per_CPDpix*(k4+1)]
else:
    resp = PCA_clean_data_postrecon[m][pnts_per_CPDpix*k4:pnts_per_CPDpix*(k4+1)]

resp=resp-np.mean(resp)
print(resp.shape)
V_per_osc=pixel_ex_wfm[pnts_per_CPDpix*k4:pnts_per_CPDpix*(k4+1)]

p1,s = npPoly.polyfit(V_per_osc,resp,deg,full=True)
y1 = npPoly.polyval(V_per_osc,p1)
print(-0.5*p1[1]/p1[2], ' V for CPD')

plt.figure()
plt.plot(V_per_osc,resp, 'k')
plt.plot(V_per_osc,y1, 'g')

#%% Repeat on the full dataset

# This is number of periods you want to average over,
# for best time resolution =1 (but takes longer to fit)
periods = 4

# Divides each time slice into several oscillation "samples"
num_sample_periods = int(np.floor(num_periods / periods)) # number of time points per CPD pixel
pnts_per_sample = int(np.floor(pnts_per_period * periods))
pnts_per_CPDpix = pnts_per_sample

# new approach since it's base-2 samples and can curve-fit to less than full cycle
decimation = 2**int(np.floor(np.log2(pnts_per_sample)))
pnts_per_CPDpix = int(N_points_per_pixel/decimation)

tx = np.linspace(0, pxl_time, pnts_per_CPDpix) # time-axis

deg = 2 #parabola
wHfit3 = np.zeros((num_rows*num_cols, pnts_per_CPDpix, deg+1))

for n in range((num_rows*num_cols)):

    if n%1000 == 0:
        print('Pixel: ', n)
        
    for k4 in range(pnts_per_CPDpix): #osc_period

        if PCA_post_reconstruction_clean == False:
            resp = h5_F3rresh[n][pnts_per_CPDpix*k4:
                                 pnts_per_CPDpix*(k4+1)]
        else:
            resp = PCA_clean_data_postrecon[n][pnts_per_CPDpix*k4:
                                               pnts_per_CPDpix*(k4+1)]

                
#        resp = h5_F3rresh[n][pnts_per_CPDpix*k4:
#                             pnts_per_CPDpix*(k4+1)]
        resp = resp-np.mean(resp)
        V_per_osc = pixel_ex_wfm[pnts_per_CPDpix*k4:
                                 pnts_per_CPDpix*(k4+1)]
        p1, _ = npPoly.polyfit(V_per_osc, resp, deg, full=True)
        wHfit3[n,k4,:] = p1

# polyfit returns a + bx + cx^2 coefficients
        
# lets us debug further
if PCA_post_reconstruction_clean == True:

    CPD_PCA = -0.5*np.divide(wHfit3[:,:,1],wHfit3[:,:,2]) # vertex of parabola
    CPD = CPD_PCA
    
else:
    
    CPD_raw = -0.5*np.divide(wHfit3[:,:,1],wHfit3[:,:,2])
    CPD = CPD_raw
    
#CPD = np.reshape(CPD, [64, 128, pnts_per_CPDpix])    

grp_name = h5_main.name.split('/')[-1] + '-CPD'
grp_CPD = px.MicroDataGroup(grp_name, h5_main.parent.name + '/')

ds_CPDon = px.MicroDataset('CPD', data=CPD, parent = '/')
grp_CPD.addChildren([ds_CPDon])
hdf.writeData(grp_CPD, print_log=True)

#%% Store to H5

try:
    dset = hdf.file.create_dataset("parafit_main", shape=wHfit3.shape, dtype=np.float32)
    dset[:,:] = wHfit3
except:
    print('Issue with parafit save')
    
hdf.file.flush()

#%% Fit function for CPD

def fitexp(x, A, tau, y0, x0):
    return A * np.exp(-(x - x0) /tau) + y0

#%% Visualize CPD vs time

# Separate CPDs into "light on" and "light off" case

CPD_off = CPD
CPD_on = CPD

time = np.linspace(0.0, pxl_time, CPD.shape[1])

dtCPD = pxl_time/CPD.shape[1] #dt for the CPD since not same length as raw data
p_on = int(light_on_time[0]*1e-3 / dtCPD) 
p_off = int(light_on_time[1]*1e-3 / dtCPD) 

time_on = time[p_on:p_off]
time_off = time[p_off:]   # last point is sometimes NaN for some reason

# Make CPD on and off, reshape into images by takign averages
CPD_on = CPD[:, p_on:p_off]
CPD_off = CPD[:, p_off:]

CPD_on_avg = np.zeros((num_rows, num_cols))
CPD_off_avg = np.zeros((num_rows, num_cols))

CPD_on_time = np.zeros((num_rows, num_cols))
CPD_off_time = np.zeros((num_rows, num_cols))

bds_on = ([-10, (1e-5), -5, time_on[0]-1e-10], 
       [-1e-10, (1e-1), 5, time_on[0]+1e-10])
p0on = [-0.025, 1e-3, 0, time_on[0]]

for r in np.arange(CPD_on_avg.shape[0]):

    for c in np.arange(CPD_on_avg.shape[1]):
        
        CPD_on_avg[r][c] = np.mean(CPD_on[r*num_cols + c,:])
        cut = CPD_on[r*num_cols + c, :] - CPD_on[r*num_cols + c, 0]
        try:
            popt, _ = curve_fit(fitexp, time_on, cut, 
                                bounds=bds_on, p0=p0on)
            CPD_on_time[r][c] = popt[1]
        except:
            CPD_on_time[r][c] = CPD_on_time[r][c-1] # blur bad pixels
            print( 'error_on')
            print(r, ' ', c)
            break

        bds_off = ([1e-10, (1e-5), -5, time_off[0]-1e-10], 
                   [10, (1e-1), 5, time_off[0]+1e-10])
        
        CPD_off_avg[r][c] = np.mean(CPD_off[r*num_cols + c,:-1])
#        cut = CPD_off[r*num_cols + c, :-1] - CPD_off[r*num_cols + c, 0]
#        try:
#            popt, _ = curve_fit(fitexp, time_off, cut, bounds=bds_off)
#            CPD_off_time[r][c] = popt[1]
#        except:
#            CPD_off_time[r][c] = CPD_off_time[r][c-1] #blur bad pixels
#            print( 'error')
#            print(r, ' ', c)
#            break
            
SPV = CPD_on_avg - CPD_off_avg

if PCA_post_reconstruction_clean == True:
    np.savetxt(output_filepath+r'\CPD_on_PCApost.txt', CPD_on_avg, delimiter=' ')
    np.savetxt(output_filepath+r'\CPD_off_PCApost.txt', CPD_off_avg, delimiter=' ')
    np.savetxt(output_filepath+r'\CPD_on_time_PCApost.txt', CPD_on_time, delimiter=' ')
    np.savetxt(output_filepath+r'\CPD_off_time_PCApost.txt', CPD_off_time, delimiter=' ')
    np.savetxt(output_filepath+r'\SPV_PCApost.txt', SPV, delimiter=' ')
else:
    np.savetxt(output_filepath+r'\CPD_on_noPCApost.txt', CPD_on_avg, delimiter=' ')
    np.savetxt(output_filepath+r'\CPD_off_noPCApost.txt', CPD_off_avg, delimiter=' ')
    np.savetxt(output_filepath+r'\CPD_on_time_noPCApost.txt', CPD_on_time, delimiter=' ')
    np.savetxt(output_filepath+r'\CPD_off_time_noPCApost.txt', CPD_off_time, delimiter=' ')
    np.savetxt(output_filepath+r'\SPV_noPCApost.txt', SPV, delimiter=' ')

# Save CPD to the H5 file, currently doesn't check for overwrites
grp_name = h5_main.name.split('/')[-1] + '-CPD'
grp_CPD = px.MicroDataGroup(grp_name, h5_main.parent.name + '/')

try: 
    CPD_exists = h5_main.parent.name + '/' + grp_CPD.name + '/' + 'CPD_on_time'
    CPD_on_exists = hdf.file[CPD_exists]   # does this file exist already?
    
    CPD_on_exists = CPD_on_time
    CPD_exists = h5_main.parent.name + '/' + grp_CPD.name + '/' + 'CPD_off_time'
    
    CPD_off_exists = hdf.file[CPD_exists]   
    CPD_off_exists = CPD_off_time
    
    print('Overwriting CPD Data!')
except:
    print('Creating new Datasets')
    ds_CPDon = px.MicroDataset('CPD_on_time', data=CPD_on_time, parent = '/')
    ds_CPDoff = px.MicroDataset('CPD_off_time', data=CPD_off_time, parent = '/')
    grp_CPD.addChildren([ds_CPDon])
    grp_CPD.addChildren([ds_CPDoff])
    grp_CPD.attrs['pulse_time'] = [light_on_time[0], light_on_time[1]]
    hdf.writeData(grp_CPD, print_log=True)

# Crop noisy data
#lines_to_cut = np.arange(63, 55, -1)
#CPD_on_avg = np.delete(CPD_on_avg, lines_to_cut, axis=0 )
#CPD_off_avg = np.delete(CPD_off_avg, lines_to_cut, axis=0 )
#CPD_on_time = np.delete(CPD_on_time, lines_to_cut, axis=0 )
#CPD_off_time = np.delete(CPD_off_time, lines_to_cut, axis=0 )
#SPV = CPD_on_avg - CPD_off_avg

#%%
# Plotting

#1e3 to put in mV
mx = np.max([np.max(CPD_on_avg), np.max(CPD_off_avg)])*1e3
mn = np.min([np.min(CPD_on_avg), np.min(CPD_off_avg)])*1e3

fig = plt.figure(figsize=(8,8))
a = fig.add_subplot(211)
a.set_axis_off()
a.set_title('CPD Off Average', fontsize=12)
a.imshow(CPD_off_avg*1e3, cmap='inferno', vmin=mn, vmax=mx)

a = fig.add_subplot(212)
a.set_axis_off()
a.set_title('CPD On Average', fontsize=12)
im = a.imshow(CPD_on_avg*1e3, cmap='inferno', vmin=mn, vmax=mx)

cx = fig.add_axes([0.88, 0.11, 0.02, 0.77])
cbar = fig.colorbar(im, cax=cx)
cbar.set_label('CPD (mV)', rotation=270, labelpad=16)
#plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

if save_figure == True:
    if PCA_post_reconstruction_clean == True:
        fig.savefig(output_filepath+'\CPDon_vs_off_PCApost.eps', format='eps')
        fig.savefig(output_filepath+'\CPDon_vs_off_PCApost.tif', format='tiff')
    else:
        fig.savefig(output_filepath+'\CPDon_vs_off_noPCApost.eps', format='eps')
        fig.savefig(output_filepath+'\CPDon_vs_off_noPCApost.tif', format='tiff')
    
# Rate images
mx = np.max([np.max(CPD_on_time), np.max(CPD_off_time)])*1e3
mn = np.min([np.min(CPD_on_time), np.min(CPD_off_time)])*1e3

# some clean-up for plotting to remove curve-fit errors
from scipy import signal

testC = signal.medfilt(CPD_on_time, kernel_size=[3,3])
testD = signal.medfilt(CPD_off_time, kernel_size=[3,3])

mnC = np.mean(testC) - 2*np.std(testC)
mxC = np.mean(testC) + 2*np.std(testC)
mnD = np.mean(testD) - 2*np.std(testD)
mxD = np.mean(testD) + 2*np.std(testD)

mn = np.min([mnC, mnD])
mx = np.max([mxC, mxD])

fig = plt.figure(figsize=(8,6))
a = fig.add_subplot(211)
a.set_axis_off()
a.set_title('CPD Off Time', fontsize=12)
a.imshow(CPD_off_time, cmap='inferno', vmin=mn, vmax=mx)

a = fig.add_subplot(212)
a.set_axis_off()
a.set_title('CPD On Time', fontsize=12)
im = a.imshow(CPD_on_time, cmap='inferno', vmin=mn, vmax=mx)

cx = fig.add_axes([0.82, 0.11, 0.02, 0.77])
cbar = fig.colorbar(im, cax=cx)
cbar.set_label('Time Constant (ms)', rotation=270, labelpad=16)

if save_figure == True:
    fig.savefig(output_filepath+'\CPD_times.eps', format='eps')
    fig.savefig(output_filepath+'\CPD_times.tif', format='tiff')

#%%
# SPV plotting

# 1e3 to put in mV
mn = (np.mean(SPV)-2*np.std(SPV))*1e3
mx = (np.mean(SPV)+2*np.std(SPV))*1e3
fig = plt.figure()
a = fig.add_subplot(111)
a.set_axis_off()
im = a.imshow(SPV*1e3, cmap='inferno', vmin=mn, vmax=mx)
cb = fig.colorbar(im)
cb.set_label('SPV (mV)')

if save_figure == True:
    if PCA_post_reconstruction_clean == True:
        fig.savefig(output_filepath+'\SPV_PCApost.eps', format='eps')
        fig.savefig(output_filepath+'\SPV_PCApost.tif', format='tiff')
    else:
        fig.savefig(output_filepath+'\SPV_noPCApost.eps', format='eps')
        fig.savefig(output_filepath+'\SPV_noPCApost.tif', format='tiff')

#%% Slice of one CPD set
test = CPD[100,:]

plt.figure()
plt.plot(time,test)
plt.xlabel('Time (ms)', fontsize=16)
plt.ylabel('CPD (V)', fontsize=16)
plt.savefig(output_filepath+'\CPD_sample.tif', format='tiff')

# random pixel
r = 16
c = 14
bds = ([-10, (1e-5), -5, time_on[0]-1e-10], 
       [-1e-10, (1e-1), 5, time_on[0]+1e-10])

p0s = [-0.025, 1e-3, 0, time_on[0]]

cut = CPD_on[r*num_cols + c, :]# - CPD_on[r*num_cols + c, 0]
popt1, _ = curve_fit(fitexp, time_on, cut, bounds=bds, p0=p0s)
print(popt1[1]*1e3, ' ms CPD on tau')
plt.figure()
plt.plot(time_on, cut)
plt.plot(time_on, fitexp(time_on, *popt1), 'g--')
plt.savefig(output_filepath+'\CPD_on_fitting_example.tif', format='tiff')

bds = ([1e-10, (1e-5), -5, time_off[0]-1e-10], 
       [10, (1e-1), 5, time_off[0]+1e-10])

cut = CPD_off[r*num_cols + c, :] - CPD_off[r*num_cols + c, 0]
popt2, _ = curve_fit(fitexp, time_off, cut, bounds=bds )
print(popt2[1]*1e3, ' ms CPD off tau')
plt.figure()
plt.plot(time_off, cut)
plt.plot(time_off, fitexp(time_off, *popt2), 'r--')
plt.savefig(output_filepath+'\CPD_off_fitting_example.tif', format='tiff')

#%% CPD vs time

indices = {106:47,
           76:49,
           75:48,
           50:29,
           13:48
          }

plt.figure()
for col in indices:
    row = indices[col]
    plt.plot(CPD[row*num_cols + col,:-1])
    print(CPD_off_time[row, col], ' s CPD off time at ', row, ' ', col)
    print(CPD_on_time[row, col], ' s CPD on time at ', row, ' ', col)
    
#%% Data Visualization of separate CPDs
    
from sklearn.utils.extmath import randomized_svd

####### CPD_ON CASE
U, S, V = randomized_svd(CPD_on, 256, n_iter=3)

# Since the two spatial dimensions (x, y) have been collapsed to one, we need to reshape the abundance maps:
abun_maps = np.reshape(U[:,:25], (num_rows, num_cols,-1))

# Visualize the variance / statistical importance of each component:
fig, axes = px.plot_utils.plot_scree(S, title='Skree plot')

if save_figure == True:
    fig.savefig(output_filepath+'\CPDon_Skree.eps', format='eps')
    fig.savefig(output_filepath+'\CPDon_Skree.tif', format='tiff')


# Visualize the eigenvectors:
first_evecs = V[:9, :]

fig, axes =px.plot_utils.plot_loops(time_on*1E+3, first_evecs, x_label='Time (ms)', 
                                    y_label='CPD Eig (a.u.)', plots_on_side=3,
                                    subtitle_prefix='Component', title='SVD Eigenvectors (F3R)',
                                    evenly_spaced=False)

if save_figure == True:
    fig.savefig(output_filepath+'\CPDonEig.eps', format='eps')
    fig.savefig(output_filepath+'\CPDon_Eig.tif', format='tiff')

# Visualize the abundance maps:
fig, axes =px.plot_utils.plot_map_stack(abun_maps, num_comps=9, heading='SVD Abundance Maps',
                                        color_bar_mode='single', cmap='inferno')

if save_figure == True:
    fig.savefig(output_filepath+'\CPDon_Loadings.eps', format='eps')
    fig.savefig(output_filepath+'\CPDon_Loadings.tif', format='tiff')

####### CPD_OFF CASE
U, S, V = randomized_svd(CPD_off, 256, n_iter=3)

# Since the two spatial dimensions (x, y) have been collapsed to one, we need to reshape the abundance maps:
abun_maps = np.reshape(U[:,:25], (num_rows, num_cols,-1))

# Visualize the variance / statistical importance of each component:
fig, axes = px.plot_utils.plot_scree(S, title='Skree plot')

if save_figure == True:
    fig.savefig(output_filepath+'\CPDoff_Skree.eps', format='eps')
    fig.savefig(output_filepath+'\CPDoff_Skree.tif', format='tiff')


# Visualize the eigenvectors:
first_evecs = V[:9, :]

fig, axes =px.plot_utils.plot_loops(time_off*1E+3, first_evecs, x_label='Time (ms)', 
                                    y_label='CPD Eig (a.u.)', plots_on_side=3,
                                    subtitle_prefix='Component', title='SVD Eigenvectors (F3R)',
                                    evenly_spaced=False)
if save_figure == True:
    fig.savefig(output_filepath+'\CPDoff_Eig.eps', format='eps')
    fig.savefig(output_filepath+'\CPDoff_Eig.tif', format='tiff')

# Visualize the abundance maps:
fig, axes =px.plot_utils.plot_map_stack(abun_maps, num_comps=9, heading='SVD Abundance Maps',
                             color_bar_mode='single', cmap='inferno')

if save_figure == True:
    fig.savefig(output_filepath+'\CPDoff_Loadings.eps', format='eps')
    fig.savefig(output_filepath+'\CPDoff_Loadings.tif', format='tiff')
    
#%% Data Visualization

'''PCA of the total CPD data'''

# do_svd = px.processing.svd_utils.SVD(h5_F3R, num_components=256)
# h5_svd_group = do_svd.compute()
from sklearn.utils.extmath import randomized_svd

U, S, V = randomized_svd(CPD[:,:-1], 256, n_iter=3)
#######U, S, V = randomized_svd(h5_filt[:].reshape([-1, pixel_ex_wfm.size]), 256, n_iter=3)
######## h5_u = h5_svd_group['U']
####### h5_v = h5_svd_group['V']
####### h5_s = h5_svd_group['S']

# Since the two spatial dimensions (x, y) have been collapsed to one, we need to reshape the abundance maps:
abun_maps = np.reshape(U[:,:25], (num_rows, num_cols,-1))

# Visualize the variance / statistical importance of each component:
fig, axes = px.plot_utils.plot_scree(S, title='Skree plot')

if save_figure == True:
    fig.savefig(output_filepath+'\CPDtotal_Skree.eps', format='eps')
    fig.savefig(output_filepath+'\CPDtotal_Skree.tif', format='tiff')


# Visualize the eigenvectors:
first_evecs = V[:6, :]

fig, axes =px.plot_utils.plot_loops(time[:-1]*1E+3, first_evecs, x_label='Time (ms)', y_label='CPD Eig (a.u.)', plots_on_side=3,
                         subtitle_prefix='Component', title='SVD Eigenvectors (F3R)', evenly_spaced=False)

if save_figure == True:
    fig.savefig(output_filepath+'\CPDtotal_Eig.eps', format='eps')
    fig.savefig(output_filepath+'\CPDtotal_Eig.tif', format='tiff')

# Visualize the abundance maps:
fig, axes =px.plot_utils.plot_map_stack(abun_maps, num_comps=9, heading='SVD Abundance Maps',
                             color_bar_mode='single', cmap='inferno')

if save_figure == True:
    fig.savefig(output_filepath+'\CPDtotal_Loadings.eps', format='eps')
    fig.savefig(output_filepath+'\CPDtotal_Loadings.tif', format='tiff')


#%% CPD Time Slices
    
timeslice = np.floor(np.arange(0.5, 8, .5) *1e-3/dtCPD)

mn = .09
mx = 0.14
for k in timeslice:
    fig = plt.figure(figsize=(10,8))
    a = fig.add_subplot(111)
    CPD_rs = np.reshape(CPD[:, int(k)], [64, 128])
    im = a.imshow(CPD_rs, cmap='inferno', vmin=mn, vmax=mx)
    a.set_axis_off()
    tl = '{0:.2f}'.format(k*dtCPD/1e-3)
    plt.title('At '+ tl + ' ms', fontsize=12)
    cx = fig.add_axes([0.92, 0.31, 0.02, 0.37])
    cbar = fig.colorbar(im, cax=cx)
    cbar.set_label('CPD (mV)', rotation=270, labelpad=16)
    #fig.savefig(output_filepath+'\CPDslice_' + tl + '_ms.eps', format='eps')
    fig.savefig(output_filepath+'\CPDslice_' + tl + '_ms.tif', format='tiff')
    
##%% Parabolic Fitting of Cleaned Data
#def single_poly(h5_resh,pixel_ex_wfm, m, pnts_per_per, k4):
#
#    resp=h5_resh[m][pnts_per_per*k4:pnts_per_per*(k4+1)]
#
#    resp=resp-np.mean(resp)
#
#    V_per_osc=pixel_ex_wfm[pnts_per_per*k4:pnts_per_per*(k4+1)]
#    p1,s=npPoly.polyfit(V_per_osc,resp,deg,full=True)
#    y1=npPoly.polyval(V_per_osc,p1)
#
#    print('pixel: ' + str(m) + 'period: ' + str(k4))
#
#    return y1
#%% 
hdf.close()