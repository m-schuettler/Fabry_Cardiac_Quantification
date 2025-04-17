'''
These functions are for the quantification of papillary muscle hypertrophy.
Version for use on UKB RAP.
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
import scipy
import skimage as ski
from statistics import mean, stdev
import os
import pandas as pd
import pydicom
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

    
def thesis_colors():
    # return ['#437080', '#91C5DC', '#78DC9B', '#E7EA5F', '#E0B159']
    return ['#91C5DC', '#437080', '#A19CDA', '#595990', '#D093A2', '#A04F64']

def cctb_colors():
    ## first two colors are used in the CCTB logo,
    ## additional colors are picked to match
    return ['#265596', '#8DCA80', '#FFF165', '#EA725C']
        

def binary_cmap(c):
    return mpl.colors.ListedColormap([(0, 0, 0, 0), c])


def auto_crop_ventricle_3D(images, template, padding=5, return_lims=False):
    '''
    automatically crops the left ventricle in the input CMR image

    arguments:
        images:         list of CMR image arrays
        template:       template array for template matching
        padding:        number of pixels added as padding in x and y directions
        return_lims:    if True, x and y limits used for cropping are returned
    
    returns: 
        crops:          list of arrays 
        fails:          list of indices of arrays where cropping failed
    '''

    crops = []  # gets filled with cropped image arrays
    fails = []  # gets filled with indices of arrays where croppin failed
    lims = []  # gets filled with x and y limits for cropping

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
        lims.append([xmin, xmax, ymin, ymax])

        padded_shape = (template.shape[0]+padding, template.shape[1]+padding, t)

        if crop.shape == padded_shape:
            crops.append(crop)
        else: 
            fails.append(index)
            print('Cropping unsuccessful. Shape is', crop.shape, 'but expected to be', padded_shape)
        
        index += 1

    if return_lims:
        return crops, fails, lims
    else:
        return crops, fails


def sort_pms(binary):
    '''
    sort papillary muscles from binary image

    arguments:
        binary:         2D segmented image
    
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


def plot_segmentation(imgs, names, pms, bps, lvms, colors=[]):
    '''
    plot segmented papillary uscles and optionally blood pool. creates correct number of subplots for any number of images

    arguments: 
        imgs:   list of arrays of cropped CMR images
        names:  list of names for the created subplots
        pms:    list of arrays of papillary muscle segmentations
        bps:    list of contours of blood pools
        lvms:   list of arrays of lv myocard segmentations
        colors: specify list of colors to be used for lvm, bp and pm segmentations, or leave empty for default colors
    '''
    
    if colors == []:
        # colors = get_colors(ibm_colors(), 3)
        colors = get_colors(thesis_colors(), 3)
    elif len(colors) != 3:
        colors = get_colors(colors, 3)
    
    # automatically calculate nrows and ncols
    nimgs = len(imgs)
    if nimgs <= 100:     # only plot first n images
        nplots = nimgs
    else:
        nplots = 100
        print('Only the first', nplots, 'are plotted.')

    ncols = 10
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
        for contour in bps[n]:
            try:
                plt.plot(contour[:, 1], contour[:, 0], color=colors[2], lw=1)
            except:
                pass
        plt.imshow(pms[n], cmap=binary_cmap(colors[0]), alpha=0.4)
        plt.imshow(lvms[n], cmap=binary_cmap(colors[1]), alpha=0.4)

        plt.title(names[n], fontsize=8)
    
    if nplots < nrows*ncols:
        for n in range(0, nrows*ncols-nplots+1):
            plt.subplot(nrows, ncols, nplots+n)
            plt.gca().set_axis_off()

    plt.tight_layout()
    plt.show()


def read_dicom(archive_path, print_output=True, return_pixel_size=False):
    '''
    read dicom files from zip archive into 3D-t array
    
    arguments:
        archive_path:   path to zip file containing dicom files
        print_output:   if True, prints information about current archive, including name and shape
    
    returns:
        array:          array read from dicoms
    '''
    
    # open zip archive
    archive = zipfile.ZipFile(archive_path, 'r')
    file_names = sorted(archive.namelist())
    if print_output:
        print('Processing archive ', os.path.basename(archive_path), '. This archive has ', len(file_names), ' files.', sep='')

    # sort dicoms by series
    series = {}
    for name in file_names:
        if 'manifest' not in name:
            with archive.open(name) as file:
                dicom = pydicom.dcmread(io.BytesIO(file.read()))
                descr = dicom.SeriesDescription
                if 'CINE_segmented_SAX_b' in descr:  # filter out unwanted keys
                    key = int(descr.replace('CINE_segmented_SAX_b', ''))
                    if key in series.keys():
                        t = dicom.TriggerTime
                        series[key] += [[name, t]]
                    else: 
                        t = dicom.TriggerTime
                        series[key] = [[name, t]]
    sorted_series = dict(sorted(series.items(), key=lambda item: item[1]))

    # temp = [sublist[1] for sublist in sorted_series[1]]
    # print('temp is', len(temp), 'long and looks like this:', temp)

    # print('These keys are kept:', series.keys())

    # get attributes
    first_name = sorted_series[list(sorted_series.keys())[0]][0][0]  # get name of first file
    with archive.open(first_name) as file:
        dicom = pydicom.dcmread(io.BytesIO(file.read()))

    X = dicom.Rows                   # number of px in x dimension  ## X and Y are swapped
    Y = dicom.Columns                # number of px in y dimension
    Z = len(series.keys())           # number of z slices
    T = dicom.CardiacNumberOfImages  # number of time points

    dx = float(dicom.PixelSpacing[1])                      # px spacing in x dimension
    dy = float(dicom.PixelSpacing[0])                      # px spacing in y dimension
    
    if dx == dy:
        pixel_size = dx
    else:
        print(os.path.basename(archive_path), 'has different pixel sizes in x and y dimensions! Only x size is used.')
        pixel_size = dx

    if hasattr(dicom, 'SpacingBetweenSlices'):
        dz = float(dicom.SpacingBetweenSlices)             # px spacing in z dimension
    else: 
        print('!! Getting dz does not work yet !!')

    try:
        dt = (sorted_series[list(sorted_series.keys())[0]][1][1] - sorted_series[list(sorted_series.keys())[0]][0][1]) * 1e-3  # temporal spacing
    except IndexError:
        dt = 1

    # check attributes
    if print_output:
        if X != 208:
            print('Something may be wrong here: X=', X, ', but expected to be 208.', sep='')
        if Y != 210:
            print('Something may be wrong here: Y=', Y, ', but expected to be 210.', sep='')
        if Z not in [10, 11]:
            print('Something may be wrong here: Z=', Z, ', but expected to be 10 or 11.', sep='')
        if T != 50:
            print('Something may be wrong here: T=', T, ', but expected to be 50.', sep='')

    # create array and fill with sorted images
    array = np.zeros((X, Y, Z, T), 'float16')  # create empty array in correct size

    files_time = []
    for z in sorted_series.keys():
        for t in range(0, T):
            name = sorted_series[z][t][0]
            with archive.open(name) as file:
                img = pydicom.dcmread(io.BytesIO(file.read())).pixel_array
                array[:, :, z-1, t] = img
    
    if print_output:
        print('Archive ', os.path.basename(archive_path), ' read successfully. The returned array has shape ', array.shape, '.', sep='', end='\n\n')
    
    if return_pixel_size:
        return array, pixel_size
    else:
        return array


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

        rotated = ski.transform.rotate(thislabel, angle=refangle, resize=True, preserve_range=True)
        projection = np.max(rotated, axis=0)
        # vd.append(np.sum(projection>0))
        vd.append(np.sum(projection))

    return hd, vd


def read_and_segment_dicoms(archive_list, tdim=2, template=[], crop=True, show_bloodpool=False, sort=True, returns='diameters', print_output=True, show_images=True, return_slices=False, return_lims=False, return_pixelsize=False):
    ''' 
    segment papillary muscles in CMR images in 2D-t
    version for use in UK Biobank RAP
    
    arguments:
        archive_list:      list of zip file paths or glob
        tdim:              position of temporal dimension of the image arrays
        template:          template as array for template matching
        crop:              should be set to False, if images in images list are already cropped. if True, left ventricle is cropped automatically with crop_ventricle()
        show_bloodpool:    if True, bloodpool segmentation is shown as orange contour in the diagnostic plot
        sort:              if True, pm segmentation is sorted and only contains papillary muscles (sorted by size)
        returns:           select, what should be returned:
                            'diameters':        horizontal and vertical diameters are calculated and returned in a dictionary
                            'areas':            pm areas in mm²
                            any other value:    nothing is returned
        print_output:      if True, print output. if False, only print important output
        show_images:       if True, show diagnostic figures
        return_slices:     if True, returns selected Z and T slices as a list
        return_lims:       if True, returns x and y limits used for cropping
        return_pixelsize:  if True, returns pixel size

    returns:
        imgs:           list of cropped image arrays
        segmentations:  dictionary containing pm and blood pool segmentations as arrays
        if returns is 'diameters':
            diameters:          dictionary containing measurements of horizontal and vertical diameters
        if returns is 'areas':
            pmareas:            list containing pm areas in mm²
        if return_slices:
            selected_slices:    nested list of selected Z and T slices
        if return_lims:
            crop_lims:          nested list of x and y limits used for cropping
        if return_pixelsize:
            px_list:            list of pixel sized for conversion to mm
    '''

    if print_output:
        print(len(archive_list), 'archive(s) found.')

    namelist = []   # gets filled with file names for diagnostic plot
    tslc = []       # gets filled with selected T slices
    imgs = []       # gets filled with cropped images for diagnostic plot
    pms = []        # gets filled with pm segmentations for diagnostic plot
    bps = []        # gets filled with pool contours for diagnostic plot
    crop_lims = []  # gets filled with x and y limits for cropping
    px_list = []    # gets filled with pixel sizes
    failcount = 0   # count failed diameter measurements

    if returns == 'diameters':
        diams = {'h': [], 'v': []}  # gets filled with horizontal and vertial diameters
    elif returns == 'areas':
        pmareas = [] # gets filled with papillary muscle areas
    pools = []  # gets filled with blood pool segmentations

    if return_slices:
        selected_slices = []  # gets filled with selected Z and T slices
    
    z = 4  # Z slice to be selected
    
    for archive_path in archive_list:

        if print_output:
            print('Processing archive ', archive_list.index(archive_path)+1, '/', len(archive_list), '...', sep='', end='\r')
        
        # read dicom image
        name = os.path.basename(archive_path).removesuffix('.zip')
        name = os.path.basename(archive_path).removesuffix('_2_0')
        image, px = read_dicom(archive_path, print_output=False, return_pixel_size=True)
        image = image[:, :, z, :]  # select z slice
        px_list.append(px)

        # crop images
        if crop:
            template = template.astype(np.float32)
            image = image.astype(np.float32)
            if return_lims:
                img, fail, c = auto_crop_ventricle_3D([image], template, return_lims=True)
                crop_lims.append(c[0])
            else:
                img, fail = auto_crop_ventricle_3D([image], template)
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
            
            # preproc = ski.morphology.closing(sliceimg, footprint=np.ones(shape=(3, 3)))
            
            threshold = ski.filters.threshold_li(sliceimg)
            binary = sliceimg > threshold
            binary = ski.segmentation.clear_border(binary)

            labels, nlabels = ski.measure.label(binary, return_num=True)
            if nlabels != 0:
                largest_blob = labels == np.argmax(np.bincount(labels[binary]))

                largest_blob = ski.morphology.binary_closing(largest_blob, footprint=np.ones(shape=(11, 11)), mode='min')
                # largest_blob = ski.morphology.binary_closing(largest_blob)
                bloodpool = ski.morphology.remove_small_holes(largest_blob, area_threshold=100)

                # bloodpool = ski.morphology.binary_erosion(bloodpool, footprint=np.ones(shape=(3, 3)))
                bloodpool = ski.morphology.binary_erosion(bloodpool)

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

        if len(pms) <= 100:      ## if nplots is changed, also change this number!!
            namelist.append(name)
            pms.append(pm)
            try:
                bps.append(ski.measure.find_contours(bloodpool))
            except:
                print('!! bloodpool type:', type(bloodpool))
                bps.append(ski.measure.find_contours(pm))
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
        namelist = [namelist[i]+' ('+tslc[i]+'/50)' for i in range(0, min([len(namelist), len(tslc)]))]
        plot_segmentation(imgs, namelist, pms, bps, show_bloodpool=show_bloodpool)
        
    # set up returns
    segs = {'pm': pms, 'bp': bps}
    
    return_list = [imgs, segs]
    if returns == 'diameters':
        return_list.append(diams)
        if failcount != 0 and print_output:
            print('Measurement failed for', failcount, 'out of', len(imgs), 'images.')
    if returns == 'areas':
        return_list.append(pmareas)
    if return_slices:
        return_list.append(selected_slices)
    if return_lims:
        return_list.append(crop_lims)
    if return_pixelsize:
        return_list.append(px_list)
    
    return return_list

        
def get_all_measurements(image_id, template, tdim=2):
    '''
    Measure all available measurements for image in path

    arguments:
        image_id:       participant ID
        template:       template array for cropping
        tdim:           index of temporal dimension of the image arrays (not counting Z)
    
    returns:
        measurements:   dictionary containing measurements
        segmentations:  dictionary containing img, segmentations and name for segmentation plot
    '''

    # check ukbb_cardiac installation, break if not found
    if not os.path.isfile('/opt/notebooks/ukbb_cardiac/demo_pipeline_2.py'):
        if not os.path.exits('/opt/notebooks/ukbb_cardiac'):
            print('!! Installation of \'ukbb_cardiac\' module was not found. Please install the forked repo (https://github.com/BioMeDS/ukbb_cardiac) to \'/opt/notebooks/ukbb_cardiac/\' and try again.')
        print('!! \'demo_pipeline_2.py\' was not found. Please download the file from the project folder to \'/opt/notebooks/ukbb_cardiac/\' and try again.')
        return

    # initiate measurements and segmentations dicts
    measurements = {'hdiam1':       [], 
                    'hdiam2':       [], 
                    'vdiam1':       [], 
                    'vdiam2':       [], 
                    'area':         [], 
                    'num':          [],
                    'pm/lv ratio':  [],
                    'ma':           [],
                    'mwt':          []}
    segmentations = {'imgs': [], 'bps': [], 'pms': [], 'lvs': [], 'names': []}

    # get path to zip archive containing CMR scan DICOMs
    img_path = glob('/mnt/project/Bulk/Heart MRI/Short axis/'+str(image_id)[:2]+'/'+str(image_id)+'*_2_0.zip')[0]
    
    # measure diameters
    img, seg, d = read_and_segment_dicoms([img_path], tdim=tdim, template=template, returns='diameters', print_output=False, show_images=False)

    if d['h'] == []:
        print('Measurement of image', str(image_id), 'failed. This image is omitted from further analysis.')
        return measurements, segmentations
    
    flat = [i for sublist in d.values() for i in sublist]   # append diameters to measurements
    for i, l in zip(flat, list(measurements.keys())[:4]):
        measurements[l] = i
    
    # measure pm area, num and pm/lv ratio
    img, seg, a, s, l, px = read_and_segment_dicoms([img_path], tdim=2, template=template, sort=False, print_output=False, show_images=False, returns='areas', return_slices=True, return_lims=True, return_pixelsize=True)
    l = l[0]
    px = px[0]

    measurements['area'] = a[0]

    _, n = ski.measure.label(seg['pm'][0], return_num=True, connectivity=1)
    measurements['num'] = n

    bp_sum = np.sum(seg['bp'][0])
    pm_sum = np.sum(seg['pm'][0])
    measurements['pm/lv ratio'] = pm_sum/bp_sum


    # save segmentations
    segmentations['imgs'] = img[0]
    segmentations['bps'] = seg['bp'][0]
    segmentations['pms'] = seg['pm'][0]
    
    # measure ma and mwt
    tmp_path = '/opt/notebooks/tmp/'        # prepare img file for LV segmentation
    id_path = tmp_path+str(image_id)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    if not os.path.exists(id_path):
        os.mkdir(id_path)
    img = read_dicom(img_path, print_output=False)
    img = np.rot90(img, k=1)
    img = np.flipud(img)
    img = nib.Nifti1Image(img.astype(np.float32), affine=np.eye(4))
    nib.save(img, id_path+'/sa.nii.gz')

    os.system('export PYTHONPATH=/opt/notebooks:$PYTHONPATH && cd /opt/notebooks/ukbb_cardiac/ && python3 demo_pipeline_2.py {0}'.format(tmp_path));
    
    z, t = s[0] # unpack selected Z and T slices from PM segmentation
    lv_seg = nib.load(id_path+'/seg_sa.nii.gz').get_fdata()
    lv_seg = lv_seg == 2
    lv_seg = lv_seg[l[2]:l[3], l[0]:l[1], z, t].T
    
    measurements['ma'] = np.sum(lv_seg)*px*px
    
    mwt = pd.read_csv(id_path+'/wall_thickness_ED_max.csv').iloc[-1]['Thickness_Max']  ## this is (probably?) in mm already
    
    measurements['mwt'] = mwt
    
    # save segmentations
    segmentations['lvs'] = lv_seg
    name = str(image_id)+' (z='+str(z)+', t='+str(t)+')'
    segmentations['names'] = name

    [os.remove(file) for file in glob(id_path+'/*')]    # delete id_path contents and id_path, they interfere with future LV segmentations
    os.rmdir(id_path)

    return measurements, segmentations


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

    nsubs = sum([m in measurements_list[0] for m in ['hdiam1', 'area', 'num', 'pm/lv ratio', 'ma', 'mwt']])
    ncols = int((nsubs+1)/2)
    fig, _ = plt.subplots(1, ncols, figsize=(ncols*3, 5))

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
        
        boxes = plt.boxplot(m, positions=range(0, len(measurements_list)*2), flierprops=flierprops, patch_artist=True)
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
        
        boxes = plt.boxplot(m, positions=range(0, len(measurements_list)), widths=0.5, flierprops=flierprops, patch_artist=True)
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
            
    if 'ma' in list(measurements_list[0].keys()):
        plt.subplot(2, ncols, sub)
        
        m = [group['ma'] for group in measurements_list]  # structure measurements for plot
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
        plt.ylabel('myocardial area [mm²]')
        
        sub += 1
        
        # calculate statistics
        if len(measurements_list) > 1:
            for cb, cbn in zip(itertools.combinations(m, 2), combnames):
                # p value
                statistics.loc['p(ma)', cbn] = scipy.stats.mannwhitneyu(cb[0], cb[1])[1]

                # cohen's d
                #statistics.loc['d(ma)', cbn] = cohens_d(cb[0], cb[1])
                statistics.loc['d(ma)', cbn] = cohens_d(cb[0], cb[1])

                # mean difference
                statistics.loc['MD(ma)', cbn] = mean_difference(cb[0], cb[1])

            # annotate significance
            if len(measurements_list) == 2:
                plt.annotate('p = '+str(np.round(statistics.loc['p(ma)', cbn], 5)),
                             xytext=(0.5, 0.92),
                             textcoords=('data', 'axes fraction'), 
                             xy=(0.5, 0.9), 
                             xycoords=('data', 'axes fraction'), 
                             ha='center', va='bottom', 
                             arrowprops=dict(arrowstyle='-[, widthB=5'))
                _, t = plt.ylim()
                plt.ylim(top=t*1.1)
            
    if 'mwt' in list(measurements_list[0].keys()):
        plt.subplot(2, ncols, sub)
        
        m = [group['mwt'] for group in measurements_list]  # structure measurements for plot
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
        plt.ylabel('max. wall thickness [mm]')
        
        sub += 1
        
        # calculate statistics
        if len(measurements_list) > 1:
            for cb, cbn in zip(itertools.combinations(m, 2), combnames):
                # p value
                statistics.loc['p(mwt)', cbn] = scipy.stats.mannwhitneyu(cb[0], cb[1])[1]

                # cohen's d
                #statistics.loc['d(ma)', cbn] = cohens_d(cb[0], cb[1])
                statistics.loc['d(mwt)', cbn] = cohens_d(cb[0], cb[1])

                # mean difference
                statistics.loc['MD(mwt)', cbn] = mean_difference(cb[0], cb[1])

            # annotate significance
            if len(measurements_list) == 2:
                plt.annotate('p = '+str(np.round(statistics.loc['p(mwt)', cbn], 5)),
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
        stat_path = '/opt/notebooks/results/stats.csv'
        statistics.to_csv(stat_path)
        print('Statistics are saved to', stat_path+'.')


def find_matches(df, matchtraits, nmatches, template, match_behaviour='softmatch'):
    '''
    Trait-match controls, return IDs

    arguments:
        df:                 control_table containing ID and trait columns
        matchtraits:        dictionary of traits and values to be matched
        nmatches:           number of IDs to be matched
        template:           template array for cropping
        match_behaviour:    match mode:
                                'omit':         use only direct matches, don't compensate for missing matches
                                'duplicate':    duplicate direct matches to compensate for missing matches
                                'softmatch':    use closest matches for numerical traits (e. g. age)  to compensate for missing matches
    
    returns:
        mids:               matched IDs
        failed_mids:        IDs with failed measurements
        measurements:       dataframe containing measurements of matches
        segmentations:      dictionary containing img, segmentations and name for segmentation plot
    '''

    mids = []       # gets filled with matched IDs
    failed_ids = [] # gets filled with ID with failed measurements

    traitnames = list(matchtraits.keys())

    # initiate measurements and segmentations dicts
    measurements = {'hdiam1':       [], 
                    'hdiam2':       [], 
                    'vdiam1':       [], 
                    'vdiam2':       [], 
                    'area':         [], 
                    'num':          [],
                    'pm/lv ratio':  [], 
                    'ma':           [],
                    'mwt':          []}
    segmentations = {'imgs': [], 'bps': [], 'pms': [], 'lvs': [], 'names': []}
    
    if match_behaviour == 'softmatch':
        for trait, value in matchtraits.items():
            if isinstance(value, numbers.Number):
                softtrait = trait
                break
        else:
            match_behaviour == 'duplicate'
            print('Soft-matching selected, but no numerical trait is found. \'Duplicate\' match behaviour used instead.')

    offset = 0   # offset for softmatched numerical traits
    while len(mids) < nmatches:
        
        if offset == 1 and match_behaviour == 'softmatch':
            print('Soft-matching with trait ', softtrait, '.', sep='')
        
        # get matches
        if match_behaviour == 'softmatch':
            softrange = [matchtraits[softtrait]-offset, matchtraits[softtrait]+offset]

            comparison = pd.Series(True, index=df.index)
            for trait, value in matchtraits.items():
                if trait == softtrait:
                    comparison &= df[trait].between(softrange[0], softrange[1])
                else:
                    comparison &= df[trait] == value
        else:
            comparison = (df[traitnames] == pd.Series(matchtraits)).all(axis=1)  # True for ids where df and matchtraits match
        matches = df.loc[comparison, 'Participant ID'].tolist()
        matches = [match_id for match_id in matches if not match_id in mids or match_id in failed_ids]  # all ids that match the traits and haven't been used before


        # try segmentation and quantification
        for match_id in matches:
            if len(mids) >= nmatches:
                ## while loop does not break as long as the for loop is still running
                ## so check number of mids here, as well
                break
            m, s = get_all_measurements(match_id, template=template)
            if not any(v == [] for v in m.values()):     # only use, if all measurements worked
                mids.append(match_id)
                [measurements[k].append(m[k]) for k in measurements.keys()]
                [segmentations[k].append(s[k]) for k in segmentations.keys()]
            else:
                failed_ids.append(match_id)

        if match_behaviour == 'omit':
            print(len(mids), 'matches found.')
            continue
        if match_behaviour == 'duplicate':
            print(len(mids), ' matches found. Duplicating to reach ', nmatches, '.', sep='')
            missing = nmatches - len(mids)
            [measurements[k].extend([m for _, m in zip(range(missing), itertools.cycle(segmentations[k][-len(mids):]))]) for k in measurements.keys()]
            continue
        
        offset += 1

    if match_behaviour == 'softmatch':
        print('Offset for soft-matching is ±', offset-1, '.              ', sep='')
            
    return mids, failed_ids, measurements, segmentations


def quantify_pms(fabry_table_path, control_table_path, template_array, traits=[2, 3], nmatches=3, match_behaviour='softmatch', show_segmentations=False):
    '''
    Read Fabry IDs from cohort table, create trait-matched control group, segement and quantify papillary muscles, plot measurements and segmentations
    Measures these values:
        'diameters':    measure horizontal and vertical diameters in mm (according to 10.1186/s12872-023-03463-w)
        'areas':        measure cross-section area in mm²
        'numbers':      count PM objects
        'pm/lv ratios': segment LVs (using Wenjia Bai's ukbb_cardiac), measure pm/lv ratio

    arguments:
        fabry_table_path:   file path to table created from fabry imaging cohort containing traits to be matched
        control_table_path: file path to table created from control pool imaging cohort containing traits to be matched
        template_array:     image array of template for LV cropping
        traits:             column indices of fabry_table of traits to be matched
        nmatches:           number of control group participants to be matched per fabry participant
        match_behaviour:    match mode:
                                'omit':         use only direct matches, don't compensate for missing matches
                                'duplicate':    duplicate direct matches to compensate for missing matches
                                'softmatch':    use closest matches for numerical traits (e. g. age)  to compensate for missing matches
        show_segmentations: if True, plot segmentations of bloodpool, PMs and LV
        measure:            select desired measurements as string or list of strings:
                                'diameters':    measure horizontal and vertical diameters in mm (according to 10.1186/s12872-023-03463-w)
                                'areas':        measure cross-section area in mm²
                                'numbers':      count PM objects
                                'pm/lv ratios': segment LVs (using Wenjia Bai's ukbb_cardiac), measure pm/lv ratio

    returns:
        f_measurements: dataframe containing measurements for fabry group
        c_measurements: dataframe containing measurements for control group
        matches:        dictionary containing ids of matched controls for each fabry id
    '''

    # read fabry_table and control_table
    fabry_table = pd.read_csv(fabry_table_path)
    traitnames = [fabry_table.columns[t] for t in traits]
    control_table = pd.read_csv(control_table_path)

    print('___ '*25, end='\n\n')
    print('\tFabry cases:\t', fabry_table.shape[0])
    if nmatches == 0:
        print('\tNo control group matching.')
    else:
        print('\tMatches each:\t', nmatches)
        print('\tTraits matched:\t', traitnames)
    print('___ '*25, end='\n')

    # create measurements dicts
    keys = ['id', 'hdiam1', 'hdiam2', 'vdiam1', 'vdiam2', 'area', 'num', 'pm/lv ratio', 'ma', 'mwt']
    f_measurements = {key: [] for key in keys}
    c_measurements = {key: [] for key in keys+['match for']}

    # create segmentations dict
    ## to save image arrays for segmentations plot
    segmentations = {'imgs': [], 'bps': [], 'pms': [], 'lvs': [], 'names': []}

    # check ukbb_cardiac installation, break if not found
    if not os.path.isfile('/opt/notebooks/ukbb_cardiac/demo_pipeline_2.py'):
        if not os.path.exists('/opt/notebooks/ukbb_cardiac'):
            print('\n!! Installation of \'ukbb_cardiac\' module was not found. Please install the forked repo (https://github.com/BioMeDS/ukbb_cardiac) to \'/opt/notebooks/ukbb_cardiac/\' and try again.')
        print('!! \'demo_pipeline_2.py\' was not found. Please download the file from the project folder to \'/opt/notebooks/ukbb_cardiac/\' and try again.')
        return

    # iterate over fabry ids
    for p in range(fabry_table.shape[0]):

        # get info for this patient
        patient = fabry_table.loc[p]
        pid = patient['Participant ID']
        print('\nProcessing Fabry patient ',str(pid),' (',(p+1), '/', fabry_table.shape[0], ')...', sep='')

        # segment and quantify PMH
        print('Segmenting and quantifying PMH...', end='\r')
        m, s = get_all_measurements(pid, template=template_array)

        if any(v == [] for v in m.values()):
            print('Measurement of Fabry patient image', str(pid), 'failed. This image is omitted from further analysis.')
            print([k for k, v in zip(list(m.keys()), list(m.values())) if v == []])
            continue
        
        f_measurements['id'].append(pid)
        for key in ['hdiam1', 'hdiam2', 'vdiam1', 'vdiam2', 'area', 'num', 'pm/lv ratio', 'ma', 'mwt']:
            f_measurements[key].append(m[key])
        s['names'] = '(F) '+s['names']
        [segmentations[k].append(s[k]) for k in list(s.keys())]
        
        # match controls
        if nmatches != 0:
            print('Finding and quantifying matching controls...', end='\r')
            matchtraits = dict([[t, patient[t]] for t in traitnames])   # get traits for this patient

            mids, failed_ids, m, s = find_matches(control_table, matchtraits, nmatches, template_array, match_behaviour=match_behaviour)

            c_measurements['id'].extend(mids)
            c_measurements['match for'].extend([pid]*len(mids))
            for key in ['hdiam1', 'hdiam2', 'vdiam1', 'vdiam2', 'area', 'num', 'pm/lv ratio', 'ma', 'mwt']:
                c_measurements[key].extend(m[key])

            for i in mids + failed_ids:   # remove all tested ids so they aren't matched again
                control_table = control_table[control_table['Participant ID'] != i]
    
    print('\nPlotting results...                    ', end='\r')
    
    # plot measurements
    if nmatches == 0:
        plot_measurements([f_measurements], labels=['Fabry'])
    else:
        plot_measurements([c_measurements, f_measurements], labels=['control', 'Fabry'])

    # plot segmentations
    if show_segmentations:
        plot_segmentation(segmentations['imgs'], segmentations['names'], segmentations['pms'], segmentations['bps'], segmentations['lvs'])

    print('Done.                     ')    
    
    return c_measurements, f_measurements, segmentations