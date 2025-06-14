'''
These functions test the robustness of the pm quantification.
'''


import cv2
from glob import glob
from math import pi
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skimage as ski
from statistics import stdev
import warnings

import ukb_functions as pmf


def auto_crop_ventricle_4D(images, template, padding=5):
    '''
    automatically crops the left ventricle in the input CMR image

    arguments:
        images:         list of CMR image arrays
        template:       template array for template matching
        padding:        number of pixels added as padding in x and y directions
    
    returns: 
        crops:          list of arrays 
        fails:          list of indices of arrays where cropping failed
    '''

    template = template.astype(np.float32)
    images = [image.astype(np.float32) for image in images]
    
    crops = []  # gets filled with cropped image arrays
    fails = []  # gets filled with indices of arrays where croppin failed

    index = 0
    for img in images:
        x, y, z, t = img.shape

        # rough cropping
        # b = 0.19
        # xmin = int(x*b)
        # xmax = int(x*(1-b))
        # ymin = int(y*b)
        # ymax = int(y*(1-b*1.5))
        # img = img[xmin:xmax, ymin:ymax, :, :]
        
        # match template to img
        results = cv2.matchTemplate(img[:, :, 4, 0], template[:, :, 4, 0], cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(results)         # get best match

        # crop image
        ymin = max_loc[0]-int((padding*0.5))
        ymax = max_loc[0]+template.shape[1]+int((padding*0.5)+0.5)
        xmin = max_loc[1]-int((padding*0.5))
        xmax = max_loc[1]+template.shape[0]+int((padding*0.5)+0.5)
        # print(ymin, ymax, xmin, xmax)
        crop = img[ xmin:xmax, ymin:ymax, :, :].copy()

        padded_shape = (template.shape[0]+padding, template.shape[1]+padding, z, t)

        if crop.shape == padded_shape:
            crops.append(crop)
        else: 
            fails.append(index)
            print('Cropping unsuccessful. Shape is', crop.shape, 'but expected to be', padded_shape)
        
        index += 1

    return crops, fails


def segment_pm_2D_v2(images, template=[], namelist=[], crop=True, show_bloodpool=False, sort=True, show_images=True):
    ''' 
    segment papillary muscles in CMR images in 2D
    version for use in UK Biobank RAP
    
    arguments:
        images:         list of images as arrays
        template:       template as array for template matching
        namelist:       list of names to be used in the diagnostic figure
        crop:           should be set to False, if images in images list are already cropped. if True, left ventricle is cropped automatically with crop_ventricle()
        show_bloodpool: if True, bloodpool segmentation is shown as orange contour in the diagnostic plot
        sort:           if True, pm segmentation is sorted and only contains papillary muscles (sorted by size)
        show_images:    if True, show diagnostic images

    returns:
        imgs:           list of cropped image arrays
        pms:            list of image arrays containing papillary muscle segmentation
        pools:          list of image arrays containing bloodpool segmentation
    '''

    # print(len(images), 'images found')

    # crop left ventricles from images
    if crop:
        template = template.astype(np.float32)
        imgs = [image.astype(np.float32) for image in images]
        imgs = pmf.auto_crop_ventricle_2D(imgs, template)
    
    imgs = [image.astype(np.uint16) for image in images]

    # segment papillary muscles
    pms = []    # gets filled with arrays containing segmented papillary muscles
    bps = []    # gets filled with blood pool contours
    pools = []  # gets filled with arrays containing segmented bloodpools

    for i, img in zip(range(0, len(imgs)), imgs):

        # segment blood pool
        threshold = ski.filters.threshold_li(img)
        binary = img > threshold
        binary = ski.segmentation.clear_border(binary)

        labels = ski.measure.label(binary)
        try:
            largest_blob = labels == np.argmax(np.bincount(labels[binary]))
        except ValueError:  # if labels does not contain any labels
            largest_blob = labels

        largest_blob = ski.morphology.binary_closing(largest_blob, footprint=np.ones(shape=(11, 11)), mode='min')
        bloodpool = ski.morphology.remove_small_holes(largest_blob, area_threshold=100)

        bloodpool = ski.morphology.binary_erosion(bloodpool)

        # segment papillary muscles in blood pool mask
        lv_contents = img*bloodpool

        preproc = ski.exposure.equalize_adapthist(lv_contents)

        threshold = ski.filters.threshold_otsu(preproc)
        if threshold <= 0.3:
            # print(namelist[i], ': Theshold of ', threshold, ' is too low, segmentation omitted for this image.', sep='')
            threshold = 0.0
        
        binary = preproc < threshold  # threshold is the upper limit here!

        pm = bloodpool*binary

        pm = ski.morphology.remove_small_objects(pm, min_size=3)
        pm = ski.morphology.binary_dilation(pm, footprint=np.ones(shape=(2, 2)))
        
        if sort:
            try:
                pm = pmf.sort_pms(binary=pm)
            except IndexError:
                print('Less than 2 segmented objects were found. Sorting is skipped for this image.')

        pms.append(pm)

        pools.append(bloodpool)
        bps.append(ski.measure.find_contours(bloodpool))

    # quantify papillary muscles
    ## removed

    # create diagnostic figure
    if show_images:
        if namelist == []:
            namelist = [str(i) for i in range(1, len(images)+1)]
        pmf.plot_segmentation(imgs, namelist, pms, bps, show_bloodpool=show_bloodpool)

    # create plot of pm areas
    ## removed

    return imgs, pms, pools

    
def test_pm_robustness(ids, template_array, selected_slices, show_images=False):
    '''
    Quantify PMH over all suitable Z and T slices and show results as a heatmap.
    
    arguments:
        ids:             list of names of zip archives to be quantified
        template_array:  image array of template for LV cropping
        selected_slices: list of selected Z and T slices from automated quantification
        show_images:     if True, show diagnostic images instead of the heatmap
    '''
    
    ids = [str(i) for i in ids]
    
    if not show_images:
        # fig, axes = plt.subplots(nrows=len(ids), figsize=(7, 1.5*len(ids)), layout='compressed')
        fig = plt.figure(figsize=(7, 1.5*len(ids)))
        gs = GridSpec(len(ids), 1, hspace=0.5)
    
    heights = []  ## gets filled with heights of subplots (number of z slices)
    
    for i, this_id in zip(range(0, len(ids)), ids):
        print('Quantifying archive ', this_id, ' (',(i+1), '/', len(ids), ')...', sep='', end='\r')
        
        # read image
        filepath = glob('/mnt/project/Bulk/Heart MRI/Short axis/'+this_id[:2]+'/'+this_id+'*.zip')[0]
        img, px = pmf.read_dicom(filepath, print_output=False, return_pixel_size=True)
        
        # crop image
        crop, fails = auto_crop_ventricle_4D([img], template_array)
        if fails != []:
            print('Cropping of archive ', this_id, ' (',(i+1), '/', len(ids), ') failed. This image is skipped.', sep='')
            if not show_images:
                # plt.subplot(len(ids), 1, i+1)
                fig.add_subplot(gs[i])
                plt.imshow(np.full([10, 50], np.nan))
                plt.xticks([])
                plt.yticks([])
            continue
        crop = crop[0]
        
        # quantify pm
        
        _, _, nz, nt = crop.shape

        these_pm_areas = np.full([nz, nt], np.nan)
        
        for t in range(0, nt):
            slices = [crop[:, :, z, t] for z in range(0, nz)]
            names = ['z='+str(z)+', t='+str(t) for z in range(0, nz)]

            imgs, pms, pools = segment_pm_2D_v2(slices, namelist=names, crop=False, sort=False, show_images=show_images, show_bloodpool=True)
            
            # filter out unsuitable z slices using circularity/isometric quotient
            drop = []
            # for j, bloodpool in zip(range(0, len(pools)), pools):
            #     bloodpool = ski.measure.label(bloodpool)
            #     warnings.filterwarnings('error')  # to catch RuntimeWarning in except
            #     try:
            #         props = ski.measure.regionprops(bloodpool)
            #         perimeter = props[0].perimeter
            #         area = props[0].area
            #         c = (4*pi*area)/(perimeter**2)  # isometric quotient        ## for a while, I used this to sort out the slices
            #     except (RuntimeWarning, IndexError):
            #         c = 0
            #     if c < 0.6 or c > 1 or np.sum(bloodpool) < 100 or np.sum(bloodpool) > 900:
            #         drop.append(j)
                # print('iq:', c)
                # print('bp area:', np.sum(bloodpool))
            
            for z in range(0, nz):
                if z not in drop:
                    these_pm_areas[z, t] = np.sum(pms[z])*px*px
        if not show_images:
            # plot pm areas
            # plt.subplot(len(ids), 1, i+1)
            fig.add_subplot(gs[i])
            if i == 0:
                plt.ylabel('Z Slice', loc='top')
            plt.gca().set_title(str(this_id), fontsize='medium', loc='left')
            heatmap = plt.imshow(these_pm_areas)
            plt.colorbar(heatmap, ax=plt.gca(), pad=0.01, label='[mm²]')
            heatmap.set_zorder(2)
            rectangle = plt.Rectangle((selected_slices[i][1]-0.5, selected_slices[i][0]-0.5), 1, 1, fill=False, color='black', zorder=3)
            plt.gca().add_patch(rectangle)
            if i+1 == len(ids):
                plt.xlabel('T Slice', loc='right')
            else:
                plt.xticks([])
                
        # calculate standard deviation of selected slice and neighboring slices
        z_neighbors = 1  # select number of neighboring Z slices to be considered in each direction
        t_neighbors = 3  # select number of neighboring T slices to be considered in each direction
        
        rows, cols = these_pm_areas.shape
        
        z_start, z_end = selected_slices[i][0]-z_neighbors % rows, selected_slices[i][0]+z_neighbors+1 % rows
        t_start, t_end = selected_slices[i][1]-t_neighbors % cols, selected_slices[i][1]+t_neighbors+1 % cols
        
        z_start, t_start = [start if start >= 0 else 50+start for start in [z_start, t_start]]
        z_end, t_end = [end if end < 50 else end-50 for end in [z_end, t_end]]
        
        if z_start > z_end:
            z_indices = np.r_[z_start:rows, 0:z_end]
        else:
            z_indices = np.arange(z_start, z_end)
            
        if t_start > t_end:
            t_indices = np.r_[t_start:cols, 0:t_end]
        else:
            t_indices = np.arange(t_start, t_end)
            
        neighborhood = these_pm_areas[np.ix_(z_indices, t_indices)].flatten().tolist()
        neighborhood = [n for n in neighborhood if not np.isnan(n) and not n == 0]
        
        if not show_images:
            plt.gca().set_title('SD = '+str(np.round(stdev(neighborhood), 2))+' mm²', fontsize='medium', loc='right')
            if t_start > t_end:
                rectangle = plt.Rectangle((t_start-0.52, z_start-0.52), t_neighbors*2+1, z_neighbors*2+1, fill=False, color='white', zorder=2)
                plt.gca().add_patch(rectangle)
                rectangle = plt.Rectangle((0-0.52, z_start-0.52), t_end, z_neighbors*2+1, fill=False, color='white', zorder=2)
                plt.gca().add_patch(rectangle)
            else:
                rectangle = plt.Rectangle((selected_slices[i][1]-t_neighbors-0.52, selected_slices[i][0]-z_neighbors-0.52), t_neighbors*2+1, z_neighbors*2+1, fill=False, color='white', zorder=2)
                plt.gca().add_patch(rectangle)

            # update height ratios to make all subplots the same width
            heights.append(these_pm_areas.shape[0])

    gs.set_height_ratios(heights)
    gs.update()
        
    
    print('                                                   ')