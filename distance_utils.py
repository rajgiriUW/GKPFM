# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:19:26 2018

@author: Raj
"""

import numpy as np
from sklearn import metrics, cluster
import pycroscopy as px
from matplotlib import pyplot as plt
from matplotlib import cm

import mask_utils

""" 
Creates a Class with various CPD data grouped based on distance to a grain boundary

The Add_position_sets function is not in the Class. It's for setting up data

"""

def add_position_sets(h5_file, group, fast_x=32e-6, slow_y=8e-6):
    """
    Adds Position_Indices and Position_Value datasets to a folder within the h5_file
    
    Uses the values of fast_x and fast_y to determine the values
    """
    
    hdf = px.ioHDF5(h5_file)
    parms_dict = hdf.file['/Measurement_000'].attrs
    
    if 'FastScanSize' in parms_dict:
        fast_x = parms_dict['FastScanSize']
    
    if 'SlowScanSize' in parms_dict:
        slow_y = parms_dict['SlowScanSize']
        
    num_rows = parms_dict['grid_num_rows']
    num_cols = parms_dict['grid_num_cols']
    pnts_per_CPDpix = hdf.file[group  +'/CPD'].shape[1]
    dt = hdf.file[group  +'/CPD'].shape[0]*1e-6/pnts_per_CPDpix
    
    grp = px.MicroDataGroup(group)
    ds_pos_ind, ds_pos_val = px.io.translators.utils.build_ind_val_dsets([num_cols, num_rows], is_spectral=False,
                                              steps=[1.0 * fast_x / num_cols,
                                                     1.0 * slow_y / num_rows],
                                              labels=['X', 'Y'], units=['m', 'm'], verbose=True)
    
    ds_spec_inds, ds_spec_vals = px.io.translators.utils.build_ind_val_dsets([pnts_per_CPDpix], is_spectral=True,
                                                                             labels=['Time'], units=['s'], steps=[dt])
    
    aux_ds_names = ['Position_Indices', 'Position_Values', 
                    'Spectroscopic_Indices', 'Spectroscopic_Values']
    
    grp.addChildren([ds_pos_ind, ds_pos_val, ds_spec_inds, ds_spec_vals])
    
    h5_refs = hdf.writeData(grp, print_log=False)
    px.hdf_utils.linkRefs(hdf.file[group], px.hdf_utils.getH5DsetRefs(aux_ds_names, h5_refs))
    
    h5_main =  hdf.file[group]
    
    return h5_main


class CPD_cluster(object):
    
    def __init__(self, h5_file, mask, imgsize, 
                 CPD_group='/Measurement_000/Channel_000/Raw_Data-CPD',
                 light_on=[1,6]):
    
        hdf = px.ioHDF5(h5_file)
        self.h5_main = hdf.file[CPD_group]
        
        self.FastScanSize = imgsize[0]
        self.SlowScanSize = imgsize[1]
        self.light_on_time = light_on

        # Set up CPD data
        self.CPD = self.h5_main['CPD'].value
        self.CPD_orig = self.h5_main['CPD']
        self.pnts_per_CPDpix = self.CPD.shape[1]
        
        self.CPD_params()
        
        # Create mask for grain boundaries
        self.mask = mask
        self.mask_nan, self.mask_on_1D, self.mask_off_1D = mask_utils.load_masks(self.mask)
        self.CPD_1D_idx = np.copy(self.mask_off_1D)
        
        return
        
    def analyze_CPD(self, CPD_avg):
        """ 
        Creates 1D arrays of data and masks 
        Then, calculates the distances and saves those.
        
        This also creates CPD_scatter within the distances function
        
        """
        # Create 1D arrays 
        self.CPD_values(CPD_avg,self.mask)
        self.make_distance_arrays()
        
        self.CPD_dist, _ = self.CPD_distances(self.CPD_1D_pos, self.mask_on_1D_pos)
        
        return

    def CPD_params(self):
        """ creates CPD averaged data and extracts parameters """
        CPD_on_time = self.h5_main['CPD_on_time']
        
        self.CPD_off_avg = np.zeros(CPD_on_time.shape)
        self.CPD_on_avg = np.zeros(CPD_on_time.shape)
        parms_dict = self.h5_main.parent.parent.attrs
        self.num_rows = parms_dict['grid_num_rows']
        self.num_cols = parms_dict['grid_num_cols']
        
        N_points_per_pixel = parms_dict['num_bins']
        IO_rate = parms_dict['IO_rate_[Hz]']     #sampling_rate
        self.pxl_time = N_points_per_pixel/IO_rate    #seconds per pixel
        
        self.dtCPD = self.pxl_time/self.CPD.shape[1] 
        p_on = int(self.light_on_time[0]*1e-3 / self.dtCPD) 
        p_off = int(self.light_on_time[1]*1e-3 / self.dtCPD) 
        
        for r in np.arange(CPD_on_time.shape[0]):
            for c in np.arange(CPD_on_time.shape[1]):
                
                self.CPD_off_avg[r][c] = np.mean(self.CPD[r*self.num_cols + c,p_off:])
                self.CPD_on_avg[r][c] = np.mean(self.CPD[r*self.num_cols + c,p_on:p_off])
        
        return

    def CPD_values(self, CPD_avg, mask):
        """
        Uses 1D Mask file (with NaN and 0) and generates CPD of non-grain boundary regions
        
        h5_file : H5Py File
            commonly as hdf.file
            
        CPD : ndarray
            RowsXColumns matrix of CPD average values such as CPD_on_avg
            
        mask : ndarray, 2D
            Unmasked locations (indices) as 1D location
        
        CPD_loc : str, optional
            The path to the dataset within the h5_file
        
        """
            
        ones = np.where(mask == 1)
        self.CPD_avg_1D_vals = np.zeros(ones[0].shape[0])
        self.CPD_1D_vals = np.zeros([ones[0].shape[0], self.CPD.shape[1]])
    
        for r,c,x in zip(ones[0], ones[1], np.arange(self.CPD_avg_1D_vals.shape[0])):
            self.CPD_avg_1D_vals[x] = CPD_avg[r][c]
            self.CPD_1D_vals[x,:] = self.CPD[self.num_cols*r + c,:]
    
        return 

    def make_distance_arrays(self):
        """
        Generates 1D arrays where the coordinates are scaled to image dimenions
        
        pos_val : ndarray of H5Py Dataset
            This is the Position_Values generated by pycroscopy. The last element
            contains the size of the image. Use add_position_sets to generate
            this in the folder
            
            can also be [size_x, size-y]
        
        Returns
        -------
        mask_on_1D_pos : ndarray Nx2
            Where mask is applied (e.g. grain boundaries)
            
        mask_off_1D_pos : ndarray Nx2
            Where mask isn't applied (e.g. grains)
        
        CPD_1D_pos : ndarray Nx2
            Identical to mask_off_1D_scaled, this exists just for easier bookkeeping
            without fear of overwriting one or the other
        
        """
        
        csz = self.FastScanSize / self.num_cols
        rsz = self.SlowScanSize / self.num_rows
              
        mask_on_1D_pos = np.zeros([self.mask_on_1D.shape[0],2])
        mask_off_1D_pos = np.zeros([self.mask_off_1D.shape[0],2])
        
        for x,y in zip(self.mask_on_1D, np.arange(mask_on_1D_pos.shape[0])):
            mask_on_1D_pos[y,0] = x[0] * rsz
            mask_on_1D_pos[y,1] = x[1] * csz
            
        for x,y in zip(self.mask_off_1D, np.arange(mask_off_1D_pos.shape[0])):
            mask_off_1D_pos[y,0] = x[0] * rsz
            mask_off_1D_pos[y,1] = x[1] * csz
        
        CPD_1D_pos = np.copy(mask_off_1D_pos) # to keep straight, but these are the same
        
        self.mask_on_1D_pos = mask_on_1D_pos
        self.mask_off_1D_pos = mask_off_1D_pos
        self.CPD_1D_pos = CPD_1D_pos
        
        return


    def CPD_distances(self,CPD_1D_pos, mask_on_1D_pos):
        """
        Calculates pairwise distance between CPD array and the mask on array.
        For each pixel, this generates a minimum distance that defines the "closeness" to 
        a grain boundary in the mask
        
        """
        CPD_dist = np.zeros(CPD_1D_pos.shape[0])
        CPD_avg_dist = np.zeros(CPD_1D_pos.shape[0])
        
        for i, x in zip(CPD_1D_pos, np.arange(CPD_dist.shape[0])):
            
            d = metrics.pairwise_distances([i], mask_on_1D_pos)
            CPD_dist[x] = np.min(d)
            CPD_avg_dist[x] = np.mean(d)
        
        # create single [x,y] dataset
        self.CPD_scatter = np.zeros([CPD_dist.shape[0],2])
        for x,y,z in zip(CPD_dist, self.CPD_avg_1D_vals, np.arange(CPD_dist.shape[0])):
            self.CPD_scatter[z] = [x, y]
        
        return CPD_dist, CPD_avg_dist

    def kmeans(self, data, clusters=3, show_results=False):
        
        """"
        
        Simple k-means
        
        Data typically is self.CPD_scatter
        
        Returns
        -------
        self.results : KMeans type
        
        self.segments : dict, Nclusters
            Contains the segmented arrays for displaying
            
        """
        
        # create single [x,y] dataset
        estimators = cluster.KMeans(clusters)
        self.results = estimators.fit(data)
        
        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)
        
        self.segments = {}
        
        if show_results:
            plt.figure()
            plt.xlabel('Distance to Nearest Boundary (um)')
            plt.ylabel('CPD (V)')
            for i in range(clusters):
                
                plt.plot(self.CPD_scatter[labels==labels_unique[i],0]*1e6,
                         self.CPD_scatter[labels==labels_unique[i],1],
                         'C'+str(i)+'.')
        
                plt.plot(cluster_centers[i][0]*1e6, cluster_centers[i][1],
                         marker='o',markerfacecolor ='C'+str(i), markersize=8, 
                         markeredgecolor='k')
                
        return self.results
    
    def segment_maps(self):
        
        """
        segments is in actual length
        segments_idx is in index coordinates
        segments_CPD is the the full CPD trace
        
        To display, make sure to do [:,1], [:,0] given row, column ordering
        Also, segments_idx is to display since pyplot uses the index on the axis
        
        """
        
        labels = self.results.labels_
        cluster_centers = self.results.cluster_centers_
        labels_unique = np.unique(labels)
        
        self.segments = {}
        self.segments_idx = {}
        self.segments_CPD = {}
        self.segments_CPD_avg = {}
        
        for i in range(len(labels_unique)):
            self.segments[i] = self.CPD_1D_pos[labels==labels_unique[i],:]
            self.segments_idx[i] = self.CPD_1D_idx[labels==labels_unique[i],:]
            self.segments_CPD[i] = self.CPD_1D_vals[labels==labels_unique[i],:]
            self.segments_CPD_avg[i] = self.CPD_avg_1D_vals[labels==labels_unique[i]]
        
        # the average CPD in that segment
        self.CPD_time_avg = {}
        for i in range(len(labels_unique)):
            
            self.CPD_time_avg[i] = np.mean(self.segments_CPD[i], axis=0)
        
        return


###########################################

def kmeans(CPD_dist, CPD_1D_vals, clusters=3, show_results=False):
    
    # create single [x,y] dataset
    CPD_scatter = np.zeros([CPD_dist.shape[0],2])
    for x,y,z in zip(CPD_dist, CPD_1D_vals, np.arange(CPD_dist.shape[0])):
        CPD_scatter[z] = [x, y]
        
    estimators = cluster.KMeans(clusters)
    results = estimators.fit(CPD_scatter)
    
    labels = results.labels_
    cluster_centers = results.cluster_centers_
    labels_unique = np.unique(labels)
    
    if show_results:
        plt.figure()
        plt.xlabel('Distance to Nearest Boundary (um)')
        plt.ylabel('CPD (V)')
        for i in range(clusters):
            
            plt.plot(CPD_scatter[labels==labels_unique[i],0]*1e6,
                     CPD_scatter[labels==labels_unique[i],1],
                     'C'+str(i)+'.')
    
            plt.plot(cluster_centers[i][0]*1e6, cluster_centers[i][1],
                     marker='o',markerfacecolor ='C'+str(i), markersize=8, 
                     markeredgecolor='k')
            
    return results

def seg_distances(CPD_scatter, labels):
    return

def CPD_positions(h5_file, CPD_avg, mask, CPD_loc = '/Measurement_000/Channel_000/Raw_Data-CPD'):
    """
    Uses 1D Mask file (with NaN and 0) and generates CPD of non-grain boundary regions
    
    h5_file : H5Py File
        commonly as hdf.file
        
    CPD : ndarray
        RowsXColumns matrix of CPD average values such as CPD_on_avg
        
    mask : ndarray, 2D
        Unmasked locations (indices) as 1D location
    
    CPD_loc : str, optional
        The path to the dataset within the h5_file
    
    """
        
    hdf = px.ioHDF5(h5_file)

    # We know what folder we are operating in
    pos_ind = hdf.file[CPD_loc+'/Position_Indices']
    pos_val = hdf.file[CPD_loc+'/Position_Values']

    ones = np.where(mask == 1)
    CPD_1D_vals = np.zeros(ones[0].shape[0])

    for r,c,x in zip(ones[0], ones[1], np.arange(CPD_1D_vals.shape[0])):
        CPD_1D_vals[x] = CPD_avg[r][c]

    return CPD_1D_vals

def make_distance_arrays(mask_on_1D, mask_off_1D, pos_val, 
                         num_rows=64, num_cols=128):
    """
    Generates 1D arrays where the coordinates are scaled to image dimenions
    
    pos_val : ndarray of H5Py Dataset
        This is the Position_Values generated by pycroscopy. The last element
        contains the size of the image. Use add_position_sets to generate
        this in the folder
        
        can also be [size_x, size-y]
    
    Returns
    -------
    mask_on_1D_pos : ndarray Nx2
        Where mask is applied (e.g. grain boundaries)
        
    mask_off_1D_pos : ndarray Nx2
        Where mask isn't applied (e.g. grains)
    
    CPD_1D_pos : ndarray Nx2
        Identical to mask_off_1D_scaled, this exists just for easier bookkeeping
        without fear of overwriting one or the other
    
    """
    
    # If from H5 file
    if 'Dataset' in str(type(pos_val)):
        pos_val = pos_val.value
    
    # the length and width of the image
    try:
        csz = pos_val[-1][0] / num_cols # per pixel
        rsz = pos_val[-1][1] / num_rows # per pixel
    except:
        pos_val = [pos_val]
        csz = pos_val[-1][0] / num_cols # per pixel
        rsz = pos_val[-1][1] / num_rows # per pixel
    
    mask_on_1D_pos = np.zeros([mask_on_1D.shape[0],2])
    mask_off_1D_pos = np.zeros([mask_off_1D.shape[0],2])
    
    for x,y in zip(mask_on_1D, np.arange(mask_on_1D_pos.shape[0])):
        mask_on_1D_pos[y,0] = x[0] * rsz
        mask_on_1D_pos[y,1] = x[1] * csz
        
    for x,y in zip(mask_off_1D, np.arange(mask_off_1D_pos.shape[0])):
        mask_off_1D_pos[y,0] = x[0] * rsz
        mask_off_1D_pos[y,1] = x[1] * csz
    
    CPD_1D_pos = np.copy(mask_off_1D_pos) # to keep straight, but these are the same
    
    return mask_on_1D_pos, mask_off_1D_pos, CPD_1D_pos

def CPD_distances(CPD_1D_pos, mask_on_1D_pos):
    """
    Calculates pairwise distance between CPD array and the mask on array.
    For each pixel, this generates a minimum distance that defines the "closeness" to 
    a grain boundary in the mask
    
    """
    CPD_dist = np.zeros(CPD_1D_pos.shape[0])
    CPD_avg_dist = np.zeros(CPD_1D_pos.shape[0])
    for i, x in zip(CPD_1D_pos, np.arange(CPD_dist.shape[0])):
        d = metrics.pairwise_distances([i], mask_on_1D_pos)
        CPD_dist[x] = np.min(d)
        CPD_avg_dist[x] = np.mean(d)
    
    return CPD_dist, CPD_avg_dist


