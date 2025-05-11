'''
These functions are for the quantification of papillary muscle hypertrophy.
Version for local use on Windows OS and for FaZiT file structure.
'''


import cv2
from glob import glob
import io
import itertools
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numbers
import numpy as np
from random import sample
import scipy
import skimage as ski
from statistics import mean, stdev
import os
import pandas as pd
import zipfile


def mean_difference(X, Y):
    return abs((mean(X) - mean(Y)))


def cohens_d(X, Y):
    s = sqrt( (stdev(X)**2) + (stdev(Y)**2) / 2)
    d = abs(mean_difference(X, Y) / s)
    return d
    

def get_colors(colors, n):
    '''
    get n colors from a list of colors
    '''
    
    if n <= len(colors):
        return colors[:n]
    else:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', colors)
        colors = cmap(np.linspace(0, 1, n))
        return colors


def ibm_colors():
    ## IBM Color Blind Safe Palette
    ## -> https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
    return ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']

    
def thesis_colors(s='md'):
    '''
    arguments:
        s:      string specifying the colors that are returned:
                    'l': only light variations,
                    'm': only medium variations,
                    'd': only dark variations,
                    or any combination
    '''
    
    l = ["#6fb9ca", "#d2899f", "#e28e61", "#8f9ec9", "#6fab7a"]   ## teal, pink, orange, purple, green
    m = ["#388c9e", "#a44c68", "#c06141", "#576ba3", "#42754b"]
    d = ["#035f71", "#761333", "#88341a", "#324375", "#1c4c28"]
    
    options = {'l': l, 'm': m, 'd': d}
    selected = [options[var] for var in s if var in options]
    
    if len(selected) == 1:
        return selected[0]
    else:
        return [color for group in zip(*selected) for color in group]


def cctb_colors():
    ## first two colors are used in the CCTB logo,
    ## additional colors are picked to match
    return ['#265596', '#8DCA80', '#FFF165', '#EA725C']
        

def binary_cmap(c):
    return mpl.colors.ListedColormap([(0, 0, 0, 0), c])


def auto_crop_ventricle_2D(images, template):
    '''
    automatically crops the left ventricle in the input CMR image

    arguments:
        images:         list of CMR image arrays
        template:       template array for template matching
    
    returns: 
        crops:          list of arrays 
    '''

    crops = []  # gets filled with cropped image arrays
    for img in images:
        #match template to img
        # results = cv2.matchTemplate(img[0, 4, :, :], template, cv2.TM_CCOEFF_NORMED)
        results = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)  # dtypes must be float32 or uint8!  ## TM_CCORR_NORMED performs about the same
        _, _, _, max_loc = cv2.minMaxLoc(results)         # get best match

        # crop image
        ymin = max_loc[1]-10
        # ymax = max_loc[1]+template.shape[0]+10    # padding removed to ensure right ventricle pool touches border
        ymax = max_loc[1]+template.shape[0]
        xmin = max_loc[0]-10
        # xmax = max_loc[0]+template.shape[1]+10
        xmax = max_loc[0]+template.shape[1]
        # print(ymin, ymax, xmin, xmax)
        # crop = img[:, :, ymin:ymax, xmin:xmax].copy()
        crop = img[ymin:ymax, xmin:xmax].copy()
        crops.append(crop)

    return crops


def auto_crop_ventricle_3D(images, template, padding=5):
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

    crops = []  # gets filled with cropped image arrays
    fails = []  # gets filled with indices of arrays where croppin failed

    index = 0
    for img in images:
        # rough cropping
        x, y, t = img.shape
        # b = 0.19
        # xmin = int(x*b)
        # xmax = int(x*(1-b))
        # ymin = int(y*b)
        # ymax = int(y*(1-b*1.5))
        # img = img[xmin:xmax, ymin:ymax, :]
        
        # match template to img
        results = cv2.matchTemplate(img[:, :, 0], template[:, :, 0], cv2.TM_CCOEFF_NORMED)
        # results = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(results)         # get best match

        # crop image
        # ymin = max_loc[0]-3
        # ymax = max_loc[0]+template.shape[1]+2
        # xmin = max_loc[1]
        # xmax = max_loc[1]+template.shape[0]+5
        ymin = max_loc[0]-int((padding*0.5))
        ymax = max_loc[0]+template.shape[1]+int((padding*0.5)+0.5)
        xmin = max_loc[1]-int((padding*0.5))
        xmax = max_loc[1]+template.shape[0]+int((padding*0.5)+0.5)
        # print(ymin, ymax, xmin, xmax)
        crop = img[xmin:xmax, ymin:ymax, :].copy()

        padded_shape = (template.shape[0]+padding, template.shape[1]+padding, t)

        if crop.shape == padded_shape:
            crops.append(crop)
        else: 
            fails.append(index)
            print('Cropping unsuccessful. Shape is', crop.shape, 'but expected to be', padded_shape)
        
        index += 1

    return crops, fails


def auto_crop_ventricle_4D(images, template):
    '''
    automatically crops the left ventricle in the input CMR image

    arguments:
        images:         list of CMR image arrays
        template:       template array for template matching
    
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
        # rough cropping
        x, y, z, t = img.shape
        b = 0.19
        xmin = int(x*b)
        xmax = int(x*(1-b))
        ymin = int(y*b)
        ymax = int(y*(1-b*1.5))
        img = img[xmin:xmax, ymin:ymax, :, :]
        
        # match template to img
        results = cv2.matchTemplate(img[:, :, 4, 0], template[:, :, 4, 0], cv2.TM_CCOEFF_NORMED)
        # results = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(results)         # get best match

        # crop image
        # ymin = max_loc[0]-10                      # padding removed to ensure right ventricle pool touches border
        # ymax = max_loc[0]+template.shape[1]+10
        # xmin = max_loc[1]-10
        # xmax = max_loc[1]+template.shape[0]+10
        ymin = max_loc[0]-3
        ymax = max_loc[0]+template.shape[1]+2
        xmin = max_loc[1]
        xmax = max_loc[1]+template.shape[0]+5
        # print(ymin, ymax, xmin, xmax)
        crop = img[ xmin:xmax, ymin:ymax, :, :].copy()

        # if crop.shape == tuple(sum(x) for x in zip(template.shape, (5, 5, 0, 0))):
        if crop.shape == (template.shape[0]+5, template.shape[1]+5, z, t):
            crops.append(crop)
        else: 
            fails.append(index)
            print('Cropping unsuccessful. Shape is', crop.shape, 'but expected to be', template.shape)
        
        index += 1

    return crops, fails


def sort_pms(binary, sortby='size'):
    '''
    sort papillary muscles from binary image

    arguments:
        binary:         2D segmented image
        sortby:         sorting method:
                            'size': sorts two largest objects
    
    returns:
        sorted_binary:  2D segmented image containing only two papillary muscles
    '''

    labeled = ski.measure.label(binary, connectivity=1)
    regions = ski.measure.regionprops(labeled)
    regions = sorted(regions, key=lambda r: r.area, reverse=True)

    mask = np.isin(labeled, [regions[0].label, regions[1].label])
    sorted_binary = np.zeros_like(binary, dtype=binary.dtype)
    sorted_binary[mask] = 1

    return sorted_binary


def plot_segmentation(imgs, names, pms, bps, show_bloodpool):
    '''
    plot segmented papillary uscles and optionally blood pool. creates correct number of subplots for any number of images

    arguments: 
        imgs:           list of arrays of cropped CMR images
        names:          list of names for the created subplots
        pms:            list of arrays of papillary muscle segmentations
        bps:            list of contours of blood pools
        show_bloodpool: if True, bloodpool segmentation is shown as orange contour in the diagnostic plot
    '''
    
    # automatically calculate nrows and ncols
    nimgs = len(imgs)
    if nimgs <= 100:     # only plot first n images
        nplots = nimgs
    else:
        nplots = 100
        print('Only the first', nplots, 'are plotted.')

    # ncols = 10
    ncols = 6
    if nplots <= ncols:
        ncols = nplots
        nrows = 1

    nrows = nplots // ncols
    if nplots % ncols != 0:
        nrows +=1 
    if nrows == 0:
        nrows = 1

    # plot results
    plt.subplots(nrows, ncols, figsize=(ncols*1.9, nrows*2.2))
    for n in range(0, nplots):
        plt.subplot(nrows, ncols, n+1)

        plt.imshow(imgs[n], cmap='gray')
        if show_bloodpool:
            for contour in bps[n]:
                try:
                    plt.plot(contour[:, 1], contour[:, 0], color='orange', lw=1)
                except:
                    pass
        plt.imshow(pms[n], cmap=binary_cmap('red'), alpha=0.4)

        plt.title(names[n], fontsize='medium')
        plt.yticks([])
        plt.xticks([])
    
    if nplots < nrows*ncols:
        for n in range(0, nrows*ncols-nplots+1):
            plt.subplot(nrows, ncols, nplots+n)
            plt.gca().set_axis_off()

    plt.tight_layout()
    plt.show()


## def read_dicom(archive_path, print_output=True, return_pixel_size=False):
## removed, only useful for UKB RAP

def read_nifti(folder_path, print_output=True, return_pixel_size=False, save=False, anonymize=False):
    '''
    read nifti files from storage folder into 3D-t array

    arguments:
        folder_path:        file path to folder containing nifti files of all patients
        print_output:       if True, prints information about current archive, including name and shape
        return_pixel_size:  if True, return pixel size
        save:               if True, saves images as .nii
        anonymize:          if False, original patient ids are used. if string, use this string in the file name instead

    returns:
        array:              array read from niftis
    '''
        
    if glob(folder_path+"/origs/*.nii.gz") == []:
        print(folder_path, "doesn't seem to contain the correct data. Check if the correct file structure exists and try again.")
        return

    for i, file in enumerate(glob(folder_path+"/origs/*.nii.gz"), start=0):     # iterate over scans

        img = nib.load(file)
        
        if i == 0 and return_pixel_size:
            dx, dy, _, _ = img.header.get_zooms()
            if dx == dy:
                pixel_size = dx
            else:
                print("\\".join(folder_path.split("\\")[-2:]), 'has different pixel sizes in x and y dimensions! Only x size is used.')
                pixel_size = dx
        if i == 0:
            affine = img.affine
            header = img.header.copy()

        img = img.get_fdata()
        if i == 0:
            x, y, _, t = img.shape
            image = np.zeros(shape=(x, y, len(glob(folder_path+"/origs/*.nii.gz")), t))

        img = img[:, :, 0, :]
        if img.shape[0] == image.shape[0] and img.shape[1] == img.shape[1]:
            image[:, :, i, :] = img
        elif img.shape[0] < image.shape[0] and img.shape[1] < img.shape[1]:
            x, y, _, _ = img.shape
            image[:x, :y, i, :] = img
        else:
            print("img is larger than image, please fix this!!")
            return

    if print_output:
        print(image.shape[2], "Z slices combined to shape", image.shape)

    if save:
        header.set_data_shape(image.shape)
        nii = nib.Nifti1Image(image, affine, header)
        if anonymize:
            name = anonymize+'-'+folder_path.split('\\')[-1]
        else:
            name = '-'.join(folder_path.split('\\')[-2:])
        path = '../data/fazit/'+name+".nii.gz"
        nib.save(nii, path)  
        if print_output:
            print("Saved to", path)
        if i == 0:
            print("Saved NIfTIs may not contain all original metadata, so be mindful of this!")
    
    if return_pixel_size:
        return image, pixel_size
    else:
        return image


def segment_pm_2D(images, template=[], namelist=[], crop=True, show_bloodpool=False, sort=True):
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

    returns:
        imgs:           list of cropped image arrays
        pms:            list of image arrays containing papillary muscle segmentation
        pools:          list of image arrays containing bloodpool segmentation
    '''

    print(len(images), 'images found')

    # crop left ventricles from images
    if crop:
        template = template.astype(np.float32)
        imgs = [image.astype(np.float32) for image in images]
        imgs = auto_crop_ventricle_2D(imgs, template)
    
    imgs = [image.astype(np.uint16) for image in images]

    # segment papillary muscles
    pms = []    # gets filled with arrays containing segmented papillary muscles
    bps = []    # gets filled with blood pool contours
    pools = []  # gets filled with arrays containing segmented bloodpools

    for img in imgs:
        # segment blood pool
        #preproc = ski.exposure.equalize_hist(img)

        threshold = ski.filters.threshold_mean(img)
        binary = img > threshold
        binary = ski.segmentation.clear_border(binary)

        labels = ski.measure.label(binary)
        largest_blob = labels == np.argmax(np.bincount(labels[binary]))

        # largest_blob = ski.morphology.binary_closing(largest_blob, footprint=np.ones(shape=(3, 3)))
        largest_blob = ski.morphology.binary_closing(largest_blob)
        bloodpool = ski.morphology.remove_small_holes(largest_blob, area_threshold=100)

        bloodpool = ski.morphology.binary_erosion(bloodpool, footprint=np.ones(shape=(3, 3)))

        # segment papillary muscles in blood pool mask
        lv_contents = img*bloodpool

        preproc = ski.exposure.equalize_adapthist(lv_contents)

        threshold = ski.filters.threshold_otsu(preproc)
        binary = preproc < threshold  # threshold is the upper limit here!

        pm = bloodpool*binary

        pm = ski.morphology.remove_small_objects(pm, min_size=3)
        #pm = ski.morphology.binary_opening(pm)  ## maybe remove
        pm = ski.morphology.binary_dilation(pm, footprint=np.ones(shape=(2, 2)))
        
        if sort:
            try:
                pm = sort_pms(binary=pm)
            except IndexError:
                print('Less than 2 segmented objects were found. Sorting is skipped for this image.')

        pms.append(pm)

        pools.append(bloodpool)
        bps.append(ski.measure.find_contours(bloodpool))

    # quantify papillary muscles
    ## removed

    # create diagnostic figure
    if namelist == []:
        namelist = [str(i) for i in range(1, len(images)+1)]
    plot_segmentation(imgs, namelist, pms, bps, show_bloodpool=show_bloodpool)

    # create plot of pm areas
    ## removed

    return imgs, pms, pools


def segment_pm_2DT(images, tdim=2, template=[], namelist=[], crop=True, show_bloodpool=False, sort=True, returns='diameters'):
    ''' 
    segment papillary muscles in CMR images in 2D-t
    version for use in UK Biobank RAP
    
    arguments:
        images:         list of images as arrays
        tdim:           position of temporal dimension of the image arrays
        template:       template as array for template matching
        namelist:       list of names to be used in the diagnostic figure
        crop:           should be set to False, if images in images list are already cropped. if True, left ventricle is cropped automatically with crop_ventricle()
        show_bloodpool: if True, bloodpool segmentation is shown as orange contour in the diagnostic plot
        sort:           if True, pm segmentation is sorted and only contains papillary muscles (sorted by size)
        returns:        select, what should be returned:
                            'diameters':        horizontal and vertical diameters are calculated and returned in a dictionary
                            'segmentations':    pm and blood pool segmentations are returned in a dictionary
                            any other value:    nothing is returned

    returns:
        imgs:           list of cropped image arrays
        pms:            list of image arrays containing papillary muscle segmentation
        pools:          list of image arrays containing bloodpool segmentation
    '''

    print(len(images), 'image(s) found')

    # crop left ventricles from images
    if crop:
        template = template.astype(np.float32)
        imgs = [image.astype(np.float32) for image in images]
        imgs, fails = auto_crop_ventricle_3D(imgs, template)
        for f in fails[::-1]:        # remove images where cropping was unsuccessful
            print(namelist[f], 'could not be cropped successfully and is omitted from further processing.')
            imgs = imgs.pop(f)
            namelist = namelist.pop(f)
        imgs = [img.astype(np.uint16) for img in imgs]
    else:
        imgs = [img.astype(np.uint16) for img in images]

    # segment papillary muscles
    segs = {'pm': [], 'bp': []} # gets filled with arrays containing segmented papillary muscles and bloodpools
    bps = []    # gets filled with blood pool contours for diagnostic plot
    tslc = []   # gets filled with selected t slices for diagnostic plot
    index = 0   # index of img in imgs, imgs.index() raises a lot of errors

    if returns == 'diameters':
        diams = {'h': [], 'v': []}  # gets filled with horizontal and vertical diameters
        fails = 0   # count failed diameter measurements

    for img in imgs:
        # segment blood pools for each t slice
        slicepools = [] # gets filled with blood pool segmentations
        areas = []      # gets filled with blood pool areas for each t slice
        for t in range(0, img.shape[tdim]):
            if tdim == 0:
                sliceimg = img[t, :, :]
            elif tdim == 1:
                sliceimg = img[:, t, :]
            elif tdim == 2:
                sliceimg = img[:, :, t]
            else: 
                print('Invalid value of tdim argument!')
            #preproc = ski.exposure.equalize_hist(img)

            # threshold = ski.filters.threshold_mean(sliceimg)
            threshold = ski.filters.threshold_triangle(sliceimg)
            binary = sliceimg > threshold
            binary = ski.segmentation.clear_border(binary)

            labels, nlabels = ski.measure.label(binary, return_num=True)
            if nlabels != 0:
                largest_blob = labels == np.argmax(np.bincount(labels[binary]))

                largest_blob = ski.morphology.binary_closing(largest_blob, footprint=np.ones(shape=(4, 4)))
                # largest_blob = ski.morphology.binary_closing(largest_blob)
                bloodpool = ski.morphology.remove_small_holes(largest_blob, area_threshold=100)

                bloodpool = ski.morphology.binary_erosion(bloodpool, footprint=np.ones(shape=(3, 3)))

                slicepools.append(bloodpool)
                areas.append(np.sum(bloodpool))
            else:
                slicepools.append(labels)   # fill slicepools and areas with anything to preserve indexing
                areas.append(0)
        
        # select end-diastole t slice for pm segmentation
        selected = areas.index(np.max(areas)) # selected t slice index
        tslc.append(str(selected))
        # print('T slice:', selected)

        if tdim == 0:
            imgs[index] = img[selected, :, :]     # replace img by selected t slice in img and imgs
            img = img[selected, :, :]
        elif tdim == 1:
            imgs[index] = img[:, selected, :]
            img = img[:, selected, :]
        elif tdim == 2:
            imgs[index] = img[:, :, selected]
            img = img[:, :, selected]
        bloodpool = slicepools[selected]

        # segment papillary muscles in blood pool mask
        lv_contents = img*bloodpool

        preproc = ski.exposure.equalize_adapthist(lv_contents)

        threshold = ski.filters.threshold_otsu(preproc)
        binary = preproc < threshold  # threshold is the upper limit here!

        pm = bloodpool*binary

        pm = ski.morphology.remove_small_objects(pm, min_size=3)
        #pm = ski.morphology.binary_opening(pm)  ## maybe remove
        pm = ski.morphology.binary_dilation(pm, footprint=np.ones(shape=(2, 2)))
        
        if sort:
            try:
                pm = sort_pms(binary=pm)
            except IndexError:
                print('Less than 2 segmented objects were found. Sorting is skipped for this image.')

        segs['pm'].append(pm)
        bps.append(ski.measure.find_contours(bloodpool))
        
        if returns == 'segmentations':
            segs['bp'].append(bloodpool)

        if returns == 'diameters':
            try:
                h, v = get_diameters(pm)
                diams['h'].extend(h)
                diams['v'].extend(v)
            except:
                fails += 1

        index += 1

    # create diagnostic figure
    if namelist == []:
        namelist = [str(i) for i in range(1, len(images)+1)]
    namelist = [namelist[i]+' ('+tslc[i]+'/'+str(images[0].shape[tdim])+')' for i in range(0, len(tslc))]
    plot_segmentation(imgs, namelist, segs['pm'], bps, show_bloodpool=show_bloodpool)

    if returns == 'segmentations':
        return imgs, segs
    elif returns == 'diameters':
        if fails != 0:
            print('Measurement failed for', fails, 'out of', len(imgs), 'images.')
        return imgs, diams
    else:
        return imgs
    

def read_and_segment_dicoms(archive_list, tdim=2, template=[], crop=True, show_bloodpool=False, sort=True, returns='diameters', print_output=True, show_images=True, return_slices=False, crop_padding=5):
    ''' 
    segment papillary muscles in CMR images in 2D-t
    version for use in UK Biobank RAP
    
    arguments:
        archive_list:   list of zip file paths or glob
        tdim:           position of temporal dimension of the image arrays
        template:       template as array for template matching
        crop:           should be set to False, if images in images list are already cropped. if True, left ventricle is cropped automatically with crop_ventricle()
        show_bloodpool: if True, bloodpool segmentation is shown as orange contour in the diagnostic plot
        sort:           if True, pm segmentation is sorted and only contains papillary muscles (sorted by size)
        returns:        select, what should be returned:
                            'diameters':        horizontal and vertical diameters are calculated and returned in a dictionary
                            'areas':            pm areas in mm²
                            any other value:    nothing is returned
        print_output:   if True, print output. if False, only print important output
        show_images:    if True, show diagnostic figures
        return_slices:  if True, returns selected Z and T slices as a list, suppresses all other outputs and returns
        crop_padding:   padding parameter handed to auto_crop_ventricle_3D function

    returns:
        imgs:           list of cropped image arrays
        segmentations:  dictionary containing pm and blood pool segmentations as arrays
        if returns is 'diameters':
            diameters:      dictionary containing measurements of horizontal and vertical diameters
        if returns is 'areas':
            pmareas:        list containing pm areas in mm²
    '''

    if print_output:
        print(len(archive_list), 'archive(s) found.')

    namelist = []   # gets filled with file names for diagnostic plot
    tslc = []       # gets filled with selected T slices
    imgs = []       # gets filled with cropped images for diagnostic plot
    pms = []        # gets filled with pm segmentations for diagnostic plot
    bps = []        # gets filled with pool contours for diagnostic plot
    failcount = 0   # count failed diameter measurements

    if returns == 'diameters':
        diams = {'h': [], 'v': []}  # gets filled with horizontal and vertial diameters
    elif returns == 'areas':
        pmareas = [] # gets filled with papillary muscle areas
        # pmareas = dict()
    pools = []  # gets filled with blood pool segmentations

    if return_slices:
        selected_slices = []  # gets filled with selected Z and T slices
    
    # z = 6  # Z slice to be selected
    
    for i, archive_path in enumerate(archive_list):

        if print_output:
            print('Processing archive ', archive_list.index(archive_path)+1, '/', len(archive_list), '...', sep='', end='\r')
        
        # read nifti image
        name = os.path.basename(archive_path).removesuffix('.gz')
        name = name.removesuffix('.nii')
        nii = nib.load(archive_path)
        px = nii.header.get_zooms()[0]
        if i == 0:
            px_rule = px
            px_lower_rule, px_upper_rule = px*0.7, px*1.3
        else:
            if px < px_lower_rule or px > px_upper_rule:
                print('Image ', archive_list.index(archive_path)+1, '/', len(archive_list), ' has pixel size ', px, ', but should be close to ', px_rule, '. Images with different pixel sizes probably aren\'t segmented correctly.', sep='')
        image = nii.get_fdata()
        z = int(image.shape[2]/2)  # get middle slice
        image = image[:, :, z, :]  # select z slice

        # crop images
        if crop:
            template = template.astype(np.float32)
            image = image.astype(np.float32)
            img, fail = auto_crop_ventricle_3D([image], template, padding=crop_padding)
            if fail != []:
                if print_output:
                    print(name, 'could not be cropped successfully and is omitted from further processing.')
                failcount += 1
                continue  # skip this image
            img = img[0].astype(np.uint16)
        
        # segment blood pools for each t slice
        slicepools = [] # gets filled with blood pool segmentations
        areas = []      # gets filled with blood pool areas for each t slice
        for t in range(0, img.shape[tdim]):
            if tdim == 0:
                sliceimg = img[t, :, :]
            elif tdim == 1:
                sliceimg = img[:, t, :]
            elif tdim == 2:
                sliceimg = img[:, :, t]
            else: 
                print('Invalid value of tdim argument!')

            # sliceimg = ski.filters.gaussian(sliceimg, sigma=3.5)
            # sliceimg = ski.exposure.rescale_intensity(sliceimg)

            threshold = ski.filters.threshold_li(sliceimg)
            # threshold = ski.filters.threshold_yen(sliceimg)
            if threshold > 170:
                threshold = 170
            binary = sliceimg > threshold
            binary = ski.segmentation.clear_border(binary)

            labels, nlabels = ski.measure.label(binary, return_num=True)
            if nlabels != 0:
                largest_blob = labels == np.argmax(np.bincount(labels[binary]))

                # largest_blob = ski.morphology.binary_closing(largest_blob, footprint=np.ones(shape=(11, 11)), mode='min')
                largest_blob = ski.morphology.binary_dilation(largest_blob, footprint=np.ones(shape=(15, 15)), mode='min')
                # bloodpool = ski.morphology.remove_small_holes(largest_blob, area_threshold=100)
                bloodpool = ski.morphology.remove_small_holes(largest_blob, area_threshold=200)
                bloodpool = ski.morphology.binary_erosion(bloodpool, footprint=np.ones(shape=(15, 15)), mode='min')

                # bloodpool = ski.morphology.binary_erosion(bloodpool, footprint=np.ones(shape=(3, 3)))
                # bloodpool = ski.morphology.binary_erosion(bloodpool)

                slicepools.append(bloodpool)
                areas.append(np.sum(bloodpool))
            else:
                slicepools.append(np.nan)   # fill slicepools and areas with anything to preserve indices
                areas.append(0)
            
        # select end-diastole t slice for pm segmentation
        selected = areas.index(np.max(areas)) # selected t slice index
        tslc.append(str(selected))
        # print('T slice:', selected)
        if return_slices:
            selected_slices.append([z, selected])

        if tdim == 0:
            imgs.append(img[selected, :, :])        # replace img by selected t slice in img and imgs
            img = img[selected, :, :]
        elif tdim == 1:
            imgs.append(img[:, selected, :])
            img = img[:, selected, :]
        elif tdim == 2:
            imgs.append(img[:, :, selected])
            img = img[:, :, selected]
        bloodpool = slicepools[selected]
        
        if np.sum(bloodpool) <=50:
            if print_output:
                print('Bloodpool is too small. Skipping image ', archive_list.index(archive_path)+1, '/', len(archive_list), '...', sep='')
            continue

        # segment papillary muscles in blood pool mask
        lv_contents = img*bloodpool

        preproc = ski.exposure.equalize_adapthist(lv_contents)

        threshold = ski.filters.threshold_otsu(preproc)
        if threshold <= 0.3:
            if print_output:
                print(name, ': Theshold of ', threshold, ' is too low, segmentation omitted for this image.', sep='')
            threshold = 0.0
        binary = preproc < threshold  # threshold is the upper limit here!

        pm = bloodpool*binary
        
        try:
            pm = ski.morphology.remove_small_objects(pm, min_size=3)
        except:
            pass
        #pm = ski.morphology.binary_opening(pm)  ## maybe remove
        pm = ski.morphology.binary_dilation(pm, footprint=np.ones(shape=(2, 2)))
        
        if sort:
            try:
                pm = sort_pms(binary=pm)
            except IndexError:
                if print_output:
                    print(name, 'contains less than 2 segmented objects. Sorting is skipped for this image.')

        # if len(pms) <= 100:      ## if nplots is changed, also change this number!!
        #     namelist.append(name)
        #     pms.append(pm)
        #     try:
        #         bps.append(ski.measure.find_contours(bloodpool))
        #     except:
        #         print('!! bloodpool type:', type(bloodpool))
        #         bps.append(ski.measure.find_contours(pm))
        namelist.append(name)
        pms.append(pm)
        try:
            bps.append(ski.measure.find_contours(bloodpool))
        except:
            bps.append([])
        pools.append(bloodpool)

        if returns == 'diameters':
            try:
                h, v = get_diameters(pm)
                h = [value*px for value in h]
                v = [value*px for value in v]
                diams['h'].extend(h)
                diams['v'].extend(v)
            except:
                failcount += 1
        
        if returns == 'areas':
            area = np.sum(pm)*px*px
            pmareas.append(area)
    
    if print_output:
        print('Processing complete.           ')
    
    # create diagnostic figure
    if show_images:
        if namelist == []:
            namelist = [str(i) for i in range(1, 50+1)]      ## if nplots is changed, also change this number!!
        # namelist = [namelist[i]+' (t='+tslc[i]+')' for i in range(0, min([len(namelist), len(tslc)]))]
        plot_segmentation(imgs, namelist, pms, bps, show_bloodpool=show_bloodpool)
        
    # set up returns
    segs = {'pm': pms, 'bp': pools, 'bpconts': bps}
    
    return_list = [imgs, segs]
    if returns == 'diameters':
        return_list.append(diams)
        if failcount != 0 and print_output:
            print('Measurement failed for', failcount, 'out of', len(imgs), 'images.')
    if returns == 'areas':
        return_list.append(pmareas)
    if return_slices:
        return_list.append(selected_slices)
    
    return return_list


def get_all_measurements(img_path, template, tdim=2):
    '''
    Measure all available measurements for image in path

    arguments:
        img_path:       file path to image
        template:       template array for cropping
        tdim:           index of temporal dimension of the image arrays (not counting Z)
    
    returns:
        measurements:   dictionary containing measurements
        segmentations:  dictionary containing img, segmentations and name for segmentation plot
    '''

    # initiate measurements and segmentations dicts
    measurements = {'hdiam1':       [], 
                    'hdiam2':       [], 
                    'vdiam1':       [], 
                    'vdiam2':       [], 
                    'area':         [], 
                    'num':          [], 
                    'pm/lv ratio':  []}
    segmentations = {'imgs': [], 'bps': [], 'bpconts': [], 'pms': [], 'names': []}

    name = os.path.basename(img_path).removesuffix('.nii.gz')

    # measure diameters
    img, seg, d = read_and_segment_dicoms([img_path], tdim=tdim, template=template, returns='diameters', print_output=False, show_images=False)

    if d['h'] == []:
        # print('Measurement of image', str(name), 'failed. This image is omitted from further analysis.')
        return measurements, segmentations

    flat = [i for sublist in d.values() for i in sublist]   # append diameters to measurements
    for i, l in zip(flat, list(measurements.keys())[:4]):
        measurements[l] = i

    # measure pm area and num
    img, seg, a, s = read_and_segment_dicoms([img_path], tdim=2, template=template, sort=False, print_output=False, show_images=False, returns='areas', return_slices=True)

    measurements['area'] = a[0]

    _, n = ski.measure.label(seg['pm'][0], return_num=True, connectivity=1)
    measurements['num'] = n

    # measure pm/lv ratio
    bp_sum = np.sum(seg['bp'][0])
    pm_sum = np.sum(seg['pm'][0])
    measurements['pm/lv ratio'] = pm_sum/bp_sum
    
    # save segmentations
    segmentations['imgs'] = img[0]
    segmentations['bps'] = seg['bp'][0]
    segmentations['bpconts'] = seg['bpconts'][0]
    segmentations['pms'] = seg['pm'][0]
    segmentations['names'] = name+'\n(z='+str(s[0][0])+', t='+str(s[0][1])+')'

    return measurements, segmentations


def get_diameters(binary):
    '''
    measure vertical and horizontal diameters of papillary muscles

    parameters: 
        binary:     2D segmented image containing only the two papillary muscle objects
    
    returns:
        hd:         horizontal diameters
        vd:         vertical diameters
    '''

    # check if image is suitable

    if not len(binary.shape) == 2:
        print('Wrong dimensionality. The binary image must be two-dimensional')
        return 
    
    labeled = ski.measure.label(binary)
    nlabels = np.max(np.unique(labeled))

    if nlabels > 2:
        # print('Too many objects in the binary image. The binary image must include only the two papillary muscle objects.')
        return
    elif nlabels < 2:
        # print('Too few objects in the binary image. The binary image must include only the two papillary muscle objects.')
        return

    # get reference angle

    centroids = []  # gets filled with coordinates of centroids
    for label in range(1, nlabels+1):
        mask = (labeled == label)
        thislabel = np.zeros_like(labeled, dtype=labeled.dtype)
        thislabel[mask] = 1

        centroids.append(ski.measure.centroid(thislabel))
    
    a = np.sqrt((centroids[1][1]-centroids[0][1])**2)   # distances between points of a triangle
    b = np.sqrt((centroids[0][0]-centroids[1][0])**2)
    c = np.sqrt((centroids[0][1]-centroids[1][1])**2+(centroids[0][0]-centroids[1][0])**2)

    refangle = np.arccos((a**2-b**2-c**2)/(-2*b*c))
    refangle = np.degrees(refangle)

    if centroids[0][1] < centroids[1][1]:
        refangle = -refangle

    # print('refangle:', refangle, '°')

    # get diameters

    hd = []         # gets filled with horizontal diameters
    vd = []         # gets filled with vertical diameters
    for label in range(1, nlabels+1):
        mask = (labeled == label)
        thislabel = np.zeros_like(labeled, dtype=labeled.dtype)
        thislabel[mask] = 1

        rotated = ski.transform.rotate(thislabel, angle=refangle+90, resize=True, preserve_range=True)
        projection = np.max(rotated, axis=0)
        # hd.append(np.sum(projection>0))     ## each pixel the object occupies is counted regardless of how much object is in the pixel
        hd.append(np.sum(projection))      ## pixel values are proportional to how much object is in the pixel and thus weighting the pixel's contribution to the diameter calculation  ## however, this is not always more accurate than the unweighted diameter!


        ##!! test if vd measurement still works!!!

        # rotated = ski.transform.rotate(thislabel, angle=refangle, resize=True, preserve_range=True)
        projection = np.max(rotated, axis=1)
        # vd.append(np.sum(projection>0))
        vd.append(np.sum(projection))

    return hd, vd


def max_area_slice(image, dimension=0, plot_areas=False):
    '''
    select T slice with largest blood pool area

    arguments: 
        image:      cropped 2D-t array of image
        dimension:  position of T dimension in the 2D-t array
        plot_areas: if True, blood pool areas of T slices are plotted in a bar plot
    
    returns:
        index:  index of selected T slice
    '''

    if dimension == 0:
        image_list = [image[t, :, :] for t in range(0, image.shape[0])]
    elif dimension == 1:
        image_list = [image[:, t, :] for t in range(0, image.shape[1])]
    elif dimension == 2:
        image_list = [image[:, :, t] for t in range(0, image.shape[2])]
    else:
        print('Invalid value for dimension argument.')
        return
    
    # segment blood pools
    _, _, pools = segment_pm_2D(image_list, crop=False, sort=False, show_bloodpool=True)

    # calculate blood pool areas and select largest
    areas = [np.sum(pool) for pool in pools]
    max_area = np.max(areas)
    index = areas.index(max_area)

    print('T slice', index, 'has the largest blood pool area:', max_area, 'px².')

    # plot areas as bar plot
    if plot_areas:
        colors = ['tab:blue']*len(areas)
        colors[index] = 'tab:orange'        # largest area is highlighted in orange
        plt.bar([n for n in range(0, len(areas))], areas, color=colors)
        plt.xlabel('T slice')
        plt.ylabel('blood pool area [px²]')
        plt.tight_layout

    return index


def plot_measurements(measurements_list, labels, colors=[], save_stats=False):
    '''
    Plot measurements as violin plots
    
    arguments:
        measurements_list:  list of dicts containing measurements
        labels:             list of group names
        colors:             specify list of colors to be used for lvm, bp and pm segmentations, or leave empty for default colors
        save_stats:         if True, statistics are saved to /opt/notebooks/results/stats.csv
    '''
    
    if colors == []:
        # colors = get_colors(ibm_colors(), len(measurements_list))
        colors = get_colors(thesis_colors(), len(measurements_list))
    elif len(colors) != len(measurements_list):
        colors = get_colors(colors, len(measurements_list))
    
    colors = [mpl.colors.to_rgb(c) for c in colors]
    alpha = 0.8
    
    flierprops = dict(marker='.', markersize=6)

    # inititate statistics dataframe
    if len(measurements_list) > 1:
        stats = ['p', 'd', 'MD']
        meas = ['hor.', 'vert.', 'area', 'num', 'pm/lv', 'ma', 'mwt']
        statnames = [s+'('+m+')' for s in stats for m in meas]
        combs = itertools.combinations(labels, 2)
        combnames = [a+' vs '+b for a, b in combs]
        statistics = pd.DataFrame(columns=combnames, index=statnames)

        positions = range(0, len(measurements_list))
    else:
        positions = [0]


    nsubs = sum([m in measurements_list[0] for m in ['hdiam1', 'area', 'num', 'pm/lv ratio', 'ma', 'mwt']])
    ncols = int((nsubs+1)/2)
    fig, _ = plt.subplots(2, ncols, figsize=(ncols*3, 5))

    sub = 1

    # plot diameters
    if 'hdiam1' in list(measurements_list[0].keys()):
        plt.subplot(2, ncols, sub)

        m = []
        for group in measurements_list:     # structure measurements for plot
            m.extend([group['hdiam1']+group['hdiam2']])
        for group in measurements_list:
            m.extend([group['vdiam1']+group['vdiam2']])

        # viols = plt.violinplot(m, positions=range(0, len(measurements_list)*2), showmeans=True, showextrema=True)
        # for item in viols:
        #     if item == 'bodies':
        #         for body, color in zip(viols[item], [c for c in colors+colors]):
        #             body.set_color(color)
        #     else:
        #         viols[item].set_colors([c for c in colors+colors])
        
        boxes = plt.boxplot(m, positions=range(0, len(measurements_list)*2), widths=0.25, flierprops=flierprops, patch_artist=True)
        for patch, color in zip(boxes['boxes'], [c for c in colors+colors]):
            patch.set_facecolor(color+(alpha,))
        for patch in boxes['medians']:
            patch.set_color('black')
        
        l = ['hor.', 'vert.']
        pos = [len(measurements_list)/2-0.5, len(measurements_list)/2+len(measurements_list)-0.5]
        plt.xticks(pos, l, fontsize='small')
        plt.ylabel('PM diameters [mm]')
        
        sub += 1
        
        # calculate statistics
        if len(measurements_list) > 1:
            mh = [[group['hdiam1']+group['hdiam2']] for group in measurements_list]
            mv = [[group['vdiam1']+group['vdiam2']] for group in measurements_list]
            for cbh, cbv, cbn in zip(itertools.combinations(mh, 2), itertools.combinations(mv, 2), combnames):
                # p value
                statistics.loc['p(hor.)', cbn] = scipy.stats.mannwhitneyu(cbh[0][0], cbh[1][0])[1]
                statistics.loc['p(vert.)', cbn] = scipy.stats.mannwhitneyu(cbv[0][0], cbv[1][0])[1]

                # cohen's d
                statistics.loc['d(hor.)', cbn] = cohens_d(cbh[0][0], cbh[1][0])
                statistics.loc['d(vert.)', cbn] = cohens_d(cbv[0][0], cbv[1][0])

                # mean difference
                statistics.loc['MD(hor.)', cbn] = mean_difference(cbh[0][0], cbh[1][0])
                statistics.loc['MD(vert.)', cbn] = mean_difference(cbv[0][0], cbv[1][0])

            # annotate significance
            if len(measurements_list) == 2:
                plt.annotate('p = '+str(np.round(statistics.loc['p(hor.)', cbn], 5)),
                             xytext=(0.5, 0.92),
                             textcoords=('data', 'axes fraction'), 
                             xy=(0.5, 0.9), 
                             xycoords=('data', 'axes fraction'), 
                             ha='center', va='bottom', 
                             arrowprops=dict(arrowstyle='-[, widthB=2'))
                plt.annotate('p = '+str(np.round(statistics.loc['p(vert.)', cbn], 5)),
                             xytext=(2.5, 0.92),
                             textcoords=('data', 'axes fraction'), 
                             xy=(2.5, 0.9), 
                             xycoords=('data', 'axes fraction'), 
                             ha='center', va='bottom', 
                             arrowprops=dict(arrowstyle='-[, widthB=2'))
                _, t = plt.ylim()
                plt.ylim(top=t*1.1)
    
    # plot areas
    if 'area' in list(measurements_list[0].keys()):
        plt.subplot(2, ncols, sub)

        m = [group['area'] for group in measurements_list]  # structure measurements for plot
        # viols = plt.violinplot(m, positions=range(0, len(measurements_list)), showmeans=True, showextrema=True)
        # for item in viols:
        #     if item == 'bodies':
        #         for body, color in zip(viols[item], colors):
        #             body.set_color(color)
        #     else:
        #         viols[item].set_colors(colors)
        
        boxes = plt.boxplot(m, positions=positions, widths=0.5, flierprops=flierprops, patch_artist=True)
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color+(alpha,))
        for patch in boxes['medians']:
            patch.set_color('black')
        
        plt.xticks([]);
        plt.ylabel('PM areas [mm²]')

        sub += 1

        # calculate statistics
        if len(measurements_list) > 1:
            for cb, cbn in zip(itertools.combinations(m, 2), combnames):
                # p value
                statistics.loc['p(area)', cbn] = scipy.stats.mannwhitneyu(cb[0], cb[1])[1]

                # cohen's d
                statistics.loc['d(area)', cbn] = cohens_d(cb[0], cb[1])

                # mean difference
                statistics.loc['MD(area)', cbn] = mean_difference(cb[0], cb[1])

            # annotate significance
            if len(measurements_list) == 2:
                plt.annotate('p = '+str(np.round(statistics.loc['p(area)', cbn], 5)),
                             xytext=(0.5, 0.92),
                             textcoords=('data', 'axes fraction'), 
                             xy=(0.5, 0.9), 
                             xycoords=('data', 'axes fraction'), 
                             ha='center', va='bottom', 
                             arrowprops=dict(arrowstyle='-[, widthB=5'))
                _, t = plt.ylim()
                plt.ylim(top=t*1.1)

    # plot pm numbers
    if 'num' in list(measurements_list[0].keys()):
        plt.subplot(2, ncols, sub)

        m = [group['num'] for group in measurements_list]  # structure measurements for plot
        # viols = plt.violinplot(m, positions=range(0, len(measurements_list)), showmeans=True, showextrema=True)
        # for item in viols:
        #     if item == 'bodies':
        #         for body, color in zip(viols[item], colors):
        #             body.set_color(color)
        #     else:
        #         viols[item].set_colors(colors)
        
        boxes = plt.boxplot(m, positions=range(0, len(measurements_list)), widths=0.5, flierprops=flierprops, patch_artist=True)
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color+(alpha,))
        for patch in boxes['medians']:
            patch.set_color('black')
        
        plt.xticks([]);
        plt.ylabel('PM number')
        
        sub += 1

        # calculate statistics
        if len(measurements_list) > 1:
            for cb, cbn in zip(itertools.combinations(m, 2), combnames):
                # p value
                statistics.loc['p(num)', cbn] = scipy.stats.mannwhitneyu(cb[0], cb[1])[1]

                # cohen's d
                statistics.loc['d(num)', cbn] = cohens_d(cb[0], cb[1])

                # mean difference
                statistics.loc['MD(num)', cbn] = mean_difference(cb[0], cb[1])

            # annotate significance
            if len(measurements_list) == 2:
                plt.annotate('p = '+str(np.round(statistics.loc['p(num)', cbn], 5)),
                             xytext=(0.5, 0.92),
                             textcoords=('data', 'axes fraction'), 
                             xy=(0.5, 0.9), 
                             xycoords=('data', 'axes fraction'), 
                             ha='center', va='bottom', 
                             arrowprops=dict(arrowstyle='-[, widthB=5'))
                _, t = plt.ylim()
                plt.ylim(top=t*1.1)
    
    # plot pm/lv ratio
    if 'pm/lv ratio' in list(measurements_list[0].keys()):
        plt.subplot(2, ncols, sub)

        m = [group['pm/lv ratio'] for group in measurements_list]  # structure measurements for plot
        # viols = plt.violinplot(m, positions=range(0, len(measurements_list)), showmeans=True, showextrema=True)
        # for item in viols:
        #     if item == 'bodies':
        #         for body, color in zip(viols[item], colors):
        #             body.set_color(color)
        #     else:
        #         viols[item].set_colors(colors)
        
        boxes = plt.boxplot(m, positions=range(0, len(measurements_list)), widths=0.5, flierprops=flierprops, patch_artist=True)
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color+(alpha,))
        for patch in boxes['medians']:
            patch.set_color('black')

        plt.xticks([]);
        plt.ylabel('PM/LV ratio')
        
        sub += 1

        # calculate statistics
        if len(measurements_list) > 1:
            for cb, cbn in zip(itertools.combinations(m, 2), combnames):
                # p value
                statistics.loc['p(pm/lv)', cbn] = scipy.stats.mannwhitneyu(cb[0], cb[1])[1]

                # cohen's d
                statistics.loc['d(pm/lv)', cbn] = cohens_d(cb[0], cb[1])

                # mean difference
                statistics.loc['MD(pm/lv)', cbn] = mean_difference(cb[0], cb[1])

            # annotate significance
            if len(measurements_list) == 2:
                plt.annotate('p = '+str(np.round(statistics.loc['p(pm/lv)', cbn], 5)),
                             xytext=(0.5, 0.92),
                             textcoords=('data', 'axes fraction'), 
                             xy=(0.5, 0.9), 
                             xycoords=('data', 'axes fraction'), 
                             ha='center', va='bottom', 
                             arrowprops=dict(arrowstyle='-[, widthB=5'))
                _, t = plt.ylim()
                plt.ylim(top=t*1.1)

    patches = [mpl.patches.Patch(facecolor=c+(alpha,), edgecolor='black', label=l) for c, l in zip(colors, labels)]
    
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncols=len(measurements_list))
    plt.tight_layout()

    # print and save statistics
    if len(measurements_list) > 1:
        print(statistics)
    
    if save_stats:
        stat_path = '../results/stats.csv'
        statistics.to_csv(stat_path)
        print('Statistics are saved to', stat_path+'.')


def quantify_pms(fabry_path, template_array, show_segmentations=False):
    '''
    read Fabry ids from cohort table, create trait-matched control group, quantify papillary muscles
    
    arguments:
        fabry_path:             file path to Fabry CMR scans, compatible with glob()
        template_array:         image array of template for LV cropping
        show_segmentationss:    if True, diagnostic images
    '''
    
    # get fabry participant ids
    nfabry = len(glob(fabry_path))
    print('Fabry patients:\t', nfabry)
    
    # iterate over fabry patients
    failcount = 0     # count failed quantifications
    
    # create measurement dict
    measurements = {'id': [], 'hdiam1': [], 'hdiam2': [], 'vdiam1': [], 'vdiam2': [], 'area': [], 'num': [], 'pm/lv ratio': []}
    
    # create segmentations dict
    segmentations = {'imgs': [], 'pms': [], 'bps': [], 'bpconts': [], 'names': []}  # gets filled with image arrays for diagnostic figure
    
    matched_ids = []  # gets filled with trait-matched control ids to be returned
    
    # for p, patient in fabry_table.iterrows():
    for p, file in enumerate(glob(fabry_path)):

        # get info for this patient
        pid = os.path.basename(file).removesuffix('.nii.gz')
        print('Processing Fabry patient ',str(pid),' (',(p+1), '/', nfabry, ')...', sep='', end='\r')
        
        # segment and quantify PMH
        m, s = get_all_measurements(file, template=template_array)

        if any(v == [] for v in m.values()):
            print('Measurement of Fabry patient image', str(pid), 'failed. This image is omitted from further analysis.')
            continue
        
        measurements['id'].append(pid)
        for key in ['hdiam1', 'hdiam2', 'vdiam1', 'vdiam2', 'area', 'num', 'pm/lv ratio']:
            measurements[key].append(m[key])
        [segmentations[k].append(s[k]) for k in s.keys()]
    
    print('\nPlotting results...                    ', end='\r')
    
    # plot measurements
    plot_measurements([measurements], labels=['Fabry'])

    # plot segmentations
    if show_segmentations:
        plot_segmentation(segmentations['imgs'], segmentations['names'], segmentations['pms'], segmentations['bpconts'], show_bloodpool=True)

    print('Done.                     ')    

    return measurements, segmentations