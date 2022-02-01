import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks




def compute_cell_statistics(img_data, circle_info, stat_to_compute='mean'):
    
    '''
    This function computes the sizes and average intensities of all the macropinosomes detected
    in the image img_data, given the circle parameteres stored in the list circle_info
    
    Arguments
    =============
    `img_data` [2-D np.ndarray]:
        image data (assumes a single channel, and hence a 2D matrix of image data)
    `circle info` [list]:
        a list of tuples, where each tuple stores the (center_x, center_y, radius) of each detected macropinosome
    `stat_to_compute` [str]:
        a string indicating which statistic to use to measure intensity of pixels within a macropinosome. Options are 'median' or 'mean'
    
    Returns
    ===========
    average intensity values for each macropinosome
    
    '''

    h, w = img_data.shape
    Y, X = np.ogrid[:h, :w]

    intensities = np.zeros(len(circle_info))
    
    for ii, cell_info in enumerate(circle_info):

        center_y, center_x, radius = cell_info

        dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)

        mask_temp = dist_from_center <= radius*0.5

        if stat_to_compute == 'median':
            intensities[ii] = np.median(img_data[mask_temp])
        elif stat_to_compute == 'mean':
            intensities[ii] = img_data[mask_temp].mean()
    
    return intensities




def create_contour_mask(imdata, contours, pts_threshold = 700):

    """This function creates cell masks within detected contours"""

    '''
    Arguments
    =============
    `imdata` [numpy array]:
        - image
    `contours` [numpy array]:
        - image

    Returns
    =============
    `contour_mask` [numpy array]:
        - creates binary cell masks based on contour info (only contours about a size threshold are considered to be cells)

    ''' 

    contour_mask = np.zeros(imdata.shape, dtype=bool)
    # For each list of contour points...
    for i in range(len(contours)):
        # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(imdata)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        if len(pts[0]) >= pts_threshold:
            contour_mask[pts[0], pts[1]]=True
    
    return contour_mask




def display_img(image_data, figsize = (16,9), cmap = 'gray'):
    '''
    Wrapper function for displaying images
    '''
    plt.figure(figsize=figsize)
    plt.imshow(image_data,cmap=cmap)




def extract_boxes(img_data, coordinates_list, dims,max_x, max_y):

    """This function crops 50x50 boxes around macropinosomes (detected with blob_dog) centers"""

    '''
    Arguments
    =============
    `img_data` [numpy array]:
        - image
    `coordinates_list` [numpy array]:
        - list of macropinosomes coordinates produced with blob_dog
    `dims` [numpy array]:
        - box dimentions
    `max_x`, `max_y` [int]:
        - image shape info

    Returns
    =============
    `all_boxes` [numpy array]:
        - array of cropped boxes

    ''' 

    all_boxes = []
    for center_i in coordinates_list:
        y, x = center_i
        box_i = img_data[ max(0, int(y - dims[0]/2)) : min(max_y, int(y + dims[0]/2)), 
                        max(0, int(x - dims[1]/2)) : min(max_x, int(x + dims[1]/2))]
        all_boxes.append(box_i)
    
    return all_boxes




def hough_circle_finder(extracted_boxes, num_peaks_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100):
    
    """This function detects circles within cropped boxes"""

    '''
    Arguments
    =============
    `extracted_boxes` [numpy array]:
        - array of cropped boxes
    `num_peaks_to_find` [int]:
        - number of circles that should be found
    `hough_radii` [numpy array]:
        - array of possible radii
    `sigma` [float]:
        - standard deviation of the Gaussian filter
    `low_threshold` [float]:
        - lower bound for hysteresis thresholding (linking edges)
    `high_threshold` [float]:
        - higher bound for hysteresis thresholding (linking edges)

    Returns
    =============
    `hough_res` [numpy array]:
        - array of circle coordinates found for each cropped box

    ''' 

    hough_res=[]
    num_boxes = len(extracted_boxes)
    for i in range(num_boxes):
        edges_i = canny(extracted_boxes[i], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        hough_res_i = hough_circle(edges_i, hough_radii)
        _, cx, cy, radii = hough_circle_peaks(hough_res_i, hough_radii,
                                            total_num_peaks=num_peaks_to_find)
        hough_res.append( (cx, cy, radii) )

    return hough_res     




def filter_circles(all_circles, num_peaks_to_find, dims, center_deviation_tolerance = 5, near_center_threshold = 2, radius_pct = 95):

    """This function selects true macropinosomes"""

    '''
    Arguments
    =============
    `all_circles` [numpy array]:
        - array of circle coordinates found for each cropped box, return of hough_circle_finder function
    `num_peaks_to_find` [int]:
        - number of circles that were found in hough_circle_finder function
    `dims` [numpy array]:
        - box dimensions
    `center_deviation_tolerance` [int]:
        - maximal distance from center of the box
    `near_center_threshold` [int]:
        - minimal number of circles that have to lie within center_deviation_tolerance
    `radius_pct` [int]:
        - percentile value for final radius calculation

    Returns
    =============
    `good_circles` [numpy array]:
        - coordinates of true macropinosomes
    `box_idx` [numpy array]:
        - indexes of boxes containing true macropinosomes
   
    ''' 

    center_coord = [dims[0]/2, dims[1]/2] # coordinate of the center of each box (in box-relative coordinates)
    good_circles = []
    box_idx = []

    for idx, circle_set in enumerate(all_circles):
        near_center_counter = 0
        good_pk_idx = []
        for pk_i in range(num_peaks_to_find):
            cx = circle_set[0][pk_i]
            cy = circle_set[1][pk_i]
            distance_from_center = np.sqrt( (center_coord[0] - cx)**2 + (center_coord[1] - cy)**2 )
            if distance_from_center <= center_deviation_tolerance:
                near_center_counter += 1
                good_pk_idx.append(pk_i)
        if near_center_counter >= near_center_threshold: 
            new_circle_coordinates = ()
            cx_new = 0
            cy_new = 0
            radius_new = []
            for pk_i in good_pk_idx:
                cx_new += circle_set[0][pk_i]
                cy_new += circle_set[1][pk_i]
                radius_new.append(circle_set[2][pk_i])
            new_circle_coordinates = (cx_new/len(good_pk_idx), cy_new/len(good_pk_idx), np.percentile(radius_new, radius_pct))
            good_circles.append(new_circle_coordinates)
            box_idx.append(idx)
    
    return good_circles, box_idx




def show_box_with_circle(box_data, box_idx, circle_coordinates, idx2inspect = None):
    
    """This function plots selected box with macropinosome and final circle that was fitted in it"""

    '''
    Arguments
    =============
    `box_data` [numpy array]:
        - array containing all cropped boxes
    `box_idx` [numpy array]:
        - indexes of boxes containing true macropinosomes (output of filter_circles function)
    `circle_coordinates` [numpy array]:
        - coordinates of true macropinosomes (output of filter_circles function)
    `idx2inspect` [int]:
        - box index that has to be plotted

    Returns
    =============
    Plot of the selected cropped box with a mcaropinosome and fitted circle
   
    ''' 

    if idx2inspect is None:
        idx2inspect = np.random.randnint(len(box_idx))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(box_data[box_idx[idx2inspect]], cmap = 'gray', interpolation = 'bicubic')
    ax[1].imshow(box_data[box_idx[idx2inspect]], cmap = 'gray', interpolation = 'bicubic')

    x, y, r = circle_coordinates[idx2inspect]
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax[1].add_patch(c)
    plt.show()
    print('Now showing circle found in box %d...\n'%box_idx[idx2inspect])
