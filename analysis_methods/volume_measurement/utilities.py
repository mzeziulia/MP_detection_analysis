import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks


def crop_image(img_data, pixel_crop = 50):

    return img_data[pixel_crop:-pixel_crop,pixel_crop:-pixel_crop]

def compute_cell_statistics(img_data, circle_info, stat_to_compute='mean'):
    '''
    This function computes the sizes and average intensities of all the macropinosomes detected
    in the image img_data, given the circle parameteres stored in the list circle_info
    Parameters:
    ----------
    img_data [2-D np.ndarray]:
        image data (assumes a single channel, and hence a 2D matrix of image data)
    circle info [list]:
        a list of tuples, where each tuple stores the (center_x, center_y, radius) of each detected macropinosome
    stat_to_compute [str]:
        a string indicating which statistic to use to measure intensity of pixels within a macropinosome. Options are 'median' or 'mean'
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

def create_cell_masks(imdata, contours, pts_threshold = 700):

    all_masks = np.zeros(imdata.shape, dtype=bool)[:,:,np.newaxis]
    # For each list of contour points...
    for i in range(len(contours)):
        mask_i = np.zeros(imdata.shape, dtype=bool)
        # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(imdata)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        if len(pts[0]) >= pts_threshold:
            mask_i[pts[0], pts[1]]=True
            all_masks = np.concatenate( (all_masks, mask_i[:,:,np.newaxis]), axis = 2)

    return all_masks[:,:,1:]

def display_img(image_data, figsize = (16,9), cmap = 'gray'):
    '''
    Wrapper function for displaying images
    '''
    plt.figure(figsize=figsize)
    plt.imshow(image_data,cmap=cmap)

def extract_boxes(img_data, coordinates_list, dims,max_x, max_y):

    all_boxes = []
    for center_i in coordinates_list:
        y, x = center_i
        box_i = img_data[ max(0, int(y - dims[0]/2)) : min(max_y, int(y + dims[0]/2)), 
                        max(0, int(x - dims[1]/2)) : min(max_x, int(x + dims[1]/2))]
        all_boxes.append(box_i)
    
    return all_boxes

def hough_circle_finder(extracted_boxes, num_peaks_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100):

    hough_res=[]
    num_boxes = len(extracted_boxes)
    for i in range(num_boxes):
        edges_i = canny(extracted_boxes[i], sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        hough_res_i = hough_circle(edges_i, hough_radii)
        _, cx, cy, radii = hough_circle_peaks(hough_res_i, hough_radii,
                                            total_num_peaks=num_peaks_to_find)
        hough_res.append( (cx, cy, radii) )

    return hough_res     

def filter_circles(all_circles, num_peaks_to_find, dims, center_deviation_tolerance = 3, near_center_threshold = 4, radius_pct = 85):

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
