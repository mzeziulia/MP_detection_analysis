import cv2
import numpy as np

def create_contour_mask(imdata, contours, pts_threshold = 700):
    '''
    Docstring @TODO
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

def create_cell_masks(imdata, contours, pts_threshold = 700):
    '''
    Docstring @TODO
    '''

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

