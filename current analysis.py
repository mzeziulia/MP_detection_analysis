# %% Imports
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import datetime as dt
import csv
import math
from scipy import sparse

import skimage
from skimage import img_as_float, filters
from skimage import io
from skimage.util import img_as_ubyte
from skimage.feature import blob_dog, canny, shape_index
from skimage.segmentation import inverse_gaussian_gradient
from skimage.transform import hough_circle, hough_circle_peaks

# %% Define auxiliary functions

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

def extract_boxes(img_data, coordinates_list, dims):

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

# %% Run analysis 
# os.chdir('Y:/group/Mariia/macrophages/BMDMS macropinocytosis/images/20200520_WT vs AZOR KO 60x oil, z stacks, 3x3 field/z projected max intensity')
# tif_file='MAX_06_WT_30_zstack_3x3004-1.tif'

# os.chdir('Y:/group/Mariia/macrophages/BMDMS macropinocytosis/images/20200514_WT vs AZOR KO/cropped images/12_wt_674')
# tif_file='3.tif'
last_frame=14
first_frame=4

conditions=['WT_200uMClGBI10min_100nMBafA_NaCl', 'C5KO_200uMClGBI10min_100nMBafA_NaCl', 'C5KO_100nMBafA_NaCl_TMR', 'WT_100nMBafA_NaCl', 'WT_NaCl_TMR', 'C5KO_NaCl_TMR']
# os.chdir("Z:/image analysis/cropped images/12032020/08_ko")
date_dir='Y:/group/mzeziulia/macrophages/BMDM/BMDMS macropinocytosis/images/20210714/z projected'
# csv_name = os.path.splitext('result_09')[0] + '.csv'
for condition in conditions:
    csv_name = os.path.join(date_dir,'volume_%s.csv'%condition)
    row_list = [['Filename', '5min radii', '6min', '7min', '8min', '9min', '10min', '11min', '12min', '13min', '14min', '15min  radii', '5min intensities', '6min', '7min', '8min', '9min', '10min', '11min', '12min', '13min', '14min', '15min  intensities']]
    for tif_file in glob.glob(os.path.join(date_dir,'*_%s*'%condition)):

        # figure_dimensions = (15, 5) # found from experience that if the width (first entry of the tuple) is 3x bigger than
        #                     # the height (second entry of the tuple) then the space between first and second rows of 
        #                     # subplots is optimal. Play around with this though.

        # fig, axes = plt.subplots(2, 6, figsize=figure_dimensions, sharex=True, sharey=True)
        # ax = axes.ravel()
        # fig.delaxes(ax[-1]) # get rid of the last subplot (since there are only 11 slices total)

        # bottom_row_shift_amount = 0.07 # amount to shift the bottom row, horizontally -- play around with this

        cell_cropped_fullstack = io.imread(tif_file)

        frame15 = cell_cropped_fullstack[last_frame,:,:]
        # frame15 = crop_image((frame15/256).astype('uint8'),50)
        img_16bit=frame15.copy()
        img_8bit = img_as_ubyte(img_16bit)
        max_y, max_x = img_8bit.shape
        # display_img(img_8bit)

        max_pctile = 95
        max_val = np.percentile(img_16bit.flatten(), max_pctile)
        max_limited = img_16bit.copy()
        max_limited[img_16bit>max_val] = max_val

        #display_img(max_limited)

        threshold_pctile = 15
        threshold_d = np.percentile(img_16bit.flatten(), threshold_pctile)

        img_hist_limited = max_limited.copy()
        img_hist_limited[img_hist_limited < threshold_d] = threshold_d

        #display_img(img_hist_limited)

        temp = img_as_float(img_hist_limited)
        gimage = inverse_gaussian_gradient(temp)

        edge_pctile = 30
        threshold_g = np.percentile(gimage.flatten(),edge_pctile)
        img_thresholded_g = gimage < threshold_g
        gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
        #display_img(gimage_8bit_thr)

        contours, hierarchy = cv2.findContours(gimage_8bit_thr,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        gimage_8bit_contours = cv2.drawContours(gimage_8bit_thr, contours, -1, (0, 255, 0), 3)
        #display_img(gimage_8bit_contours)

        img_binar = create_contour_mask(img_16bit, contours, pts_threshold = 800)
        #display_img(img_binar)

        img_16bit_cleaned = img_16bit.copy()
        img_16bit_cleaned[img_binar == 0] = 0

        blobs = blob_dog(img_16bit, min_sigma = 1, max_sigma=15, threshold=.01)

        blobs_list=[]
        for blob_info in blobs:
            y,x,r = blob_info
            if img_16bit_cleaned[int(y), int(x)] > 0: # this makes sure you only include blobs whose center pixel is on the mask  
                blobs_list.append((y,x))

        # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
        # ax = axes.ravel()
        # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
        # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
        # for filtered_blob in blobs_list:
        #     y, x = filtered_blob
        #     c = plt.Circle((x, y), 10, color='red', linewidth=2, fill=False)
        #     ax[1].add_patch(c)

        bounding_box_dims = [50, 50] # set the height and width of your bounding box
        all_boxes = extract_boxes(img_16bit, blobs_list, bounding_box_dims)

        num_circles_to_find = 5
        hough_radii = np.arange(3, 35)
        hough_res = hough_circle_finder(all_boxes, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)

        good_circles, good_box_idx = filter_circles(hough_res, num_circles_to_find, bounding_box_dims, 
                                        center_deviation_tolerance = 5, near_center_threshold = 2, radius_pct = 95)  


        radii=np.array(good_circles)[:,2]
        good_circles_original_info = np.hstack( (np.array(blobs_list)[good_box_idx,:], radii[:,np.newaxis]) )
        intensities = compute_cell_statistics(img_16bit, good_circles_original_info)
        good_circles_coordinates = np.array(blobs_list)[good_box_idx,:]
        
        # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
        # ax = axes.ravel()

        # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
        # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')

        # for i in range(len(good_circles_original_info)):
        #     y,x,r = good_circles_original_info[i]
        #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        #     ax[1].add_patch(c)

    #show_box_with_circle(all_boxes, good_box_idx, good_circles, idx2inspect = np.random.randint(len(good_box_idx)))

        final_radii = np.empty([last_frame-first_frame+1, len(radii)])
        final_intensities= np.empty([last_frame-first_frame+1, len(intensities)])

        final_radii[len(final_radii)-1]=radii
        final_intensities[len(final_radii)-1]=intensities

        # ax[10].imshow(frame15, cmap = 'gray', interpolation = 'bicubic')

        # for blob in good_circles_original_info:
        #     y, x, r = blob
        #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        #     ax[10].add_patch(c)
        #     ax[10].set_axis_off()

        # curr_pos = ax[10].get_position()
        # new_pos = [curr_pos.x0 + bottom_row_shift_amount, curr_pos.y0,  curr_pos.width, curr_pos.height]
        # ax[10].set_position(new_pos)


        for stack in reversed(range(first_frame, last_frame)):
            frame = cell_cropped_fullstack[stack,:,:]
            # frame = crop_image((frame/256).astype('uint8'),50)
            img_16bit=frame.copy()
            img_8bit = img_as_ubyte(img_16bit)
            max_y, max_x = img_8bit.shape
            # display_img(img_8bit)

            max_pctile = 95
            max_val = np.percentile(img_16bit.flatten(), max_pctile)
            max_limited = img_16bit.copy()
            max_limited[img_16bit>max_val] = max_val

            #display_img(max_limited)

            threshold_pctile = 15
            threshold_d = np.percentile(img_16bit.flatten(), threshold_pctile)

            img_hist_limited = max_limited.copy()
            img_hist_limited[img_hist_limited < threshold_d] = threshold_d

            #display_img(img_hist_limited)

            temp = img_as_float(img_hist_limited)
            gimage = inverse_gaussian_gradient(temp)

            edge_pctile = 30
            threshold_g = np.percentile(gimage.flatten(),edge_pctile)
            img_thresholded_g = gimage < threshold_g
            gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
            #display_img(gimage_8bit_thr)

            contours, hierarchy = cv2.findContours(gimage_8bit_thr,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            gimage_8bit_contours = cv2.drawContours(gimage_8bit_thr, contours, -1, (0, 255, 0), 3)
            #display_img(gimage_8bit_contours)

            img_binar = create_contour_mask(img_16bit, contours, pts_threshold = 800)
            #display_img(img_binar)

            img_16bit_cleaned = img_16bit.copy()
            img_16bit_cleaned[img_binar == 0] = 0

            blobs = blob_dog(img_16bit, min_sigma = 1, max_sigma=15, threshold=.01)

            blobs_list=[]
            for blob_info in blobs:
                y,x,r = blob_info
                if img_16bit_cleaned[int(y), int(x)] > 0: # this makes sure you only include blobs whose center pixel is on the mask  
                    blobs_list.append((y,x))

            # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
            # ax = axes.ravel()
            # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
            # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
            # for filtered_blob in blobs_list:
            #     y, x = filtered_blob
            #     c = plt.Circle((x, y), 10, color='red', linewidth=2, fill=False)
            #     ax[1].add_patch(c)

            bounding_box_dims = [50, 50] # slightly begger boxes
            all_boxes_next = extract_boxes(img_16bit, blobs_list, bounding_box_dims) #boxes based on coordinates of blobs in frame 5
            num_circles_to_find = 5
            hough_radii = np.arange(3, 35)
            hough_res_next = hough_circle_finder(all_boxes_next, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)
            good_circles_next, good_box_idx_next = filter_circles(hough_res_next, num_circles_to_find, bounding_box_dims, 
                                        center_deviation_tolerance = 5, near_center_threshold = 1, radius_pct = 95)  
            radii_next=np.array(good_circles_next)[:,2]

            good_circles_original_info = np.hstack( (np.array(blobs_list)[good_box_idx_next,:], radii_next[:,np.newaxis]) )

            correct_coordinates = np.zeros([len(good_circles_coordinates), 3])
            acceptable_distance = 30
            for circles_old in range(len(good_circles_coordinates)):
                distance_old=acceptable_distance
                for circles_new in range(len(good_circles_original_info)):
                    if np.absolute(good_circles_original_info[circles_new,2] - final_radii[stack-(first_frame-1),circles_old]) < 0.3*good_circles_original_info[circles_new,2]:
                        if np.absolute(good_circles_coordinates[circles_old,0] - good_circles_original_info[circles_new,0]) < acceptable_distance and np.absolute(good_circles_coordinates[circles_old][1] - good_circles_original_info[circles_new][1]) < acceptable_distance:
                            distance = np.sqrt( (good_circles_coordinates[circles_old][0] - good_circles_original_info[circles_new][0])**2 + (good_circles_coordinates[circles_old][1] - good_circles_original_info[circles_new][1])**2 )
                            if distance<=distance_old:
                                distance_old=distance
                                correct_coordinates[circles_old][0]=good_circles_original_info[circles_new][0]
                                correct_coordinates[circles_old][1]=good_circles_original_info[circles_new][1]
                                correct_coordinates[circles_old][2]=good_circles_original_info[circles_new][2]
            # list_to_delete=[]
            # for i in range(len(correct_coordinates)):
            #     if correct_coordinates[i][2]==0:
            #         list_to_delete.append(i)
            
            # for delete in list_to_delete:
            #     correct_coordinates=np.delete(correct_coordinates, delete, 0)
            intensities_next = compute_cell_statistics(img_16bit, correct_coordinates)

            # for i in range(len(intensities_next)):
            #     if np.absolute(intensities_next[i]-final_intensities[stack-3,i])>2000:
            #         correct_coordinates[i,:]=0
            
            for i in range(len(correct_coordinates)):
                good_circles_coordinates[i][0]=correct_coordinates[i][0]
                good_circles_coordinates[i][1]=correct_coordinates[i][1]
                if correct_coordinates[i][2]>0:
                    final_intensities[stack-first_frame][i]=intensities_next[i]
                    final_radii[stack-first_frame][i]=correct_coordinates[i][2]
                else:
                    final_intensities[stack-first_frame][i]=np.nan
                    final_radii[stack-first_frame][i] = np.nan


            # if len(good_circles_coordinates) > len(correct_coordinates):
            #     a=len(good_circles_coordinates)-len(correct_coordinates)
            #     good_circles_coordinates_new = np.delete(good_circles_coordinates, np.s_[a+1:len(good_circles_coordinates)], 0)  
            # else:
            #     good_circles_coordinates_new = good_circles_coordinates
                
            # good_circles_coordinates = good_circles_coordinates_new

            # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
            # ax = axes.ravel()
            # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
            # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
            # for filtered_blob in correct_coordinates:
            #     y, x,r = filtered_blob
            #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            #     ax[1].add_patch(c)

            # ax[stack-4].imshow(frame, cmap = 'gray', interpolation = 'bicubic')

            # for blob in correct_coordinates:
            #     y, x, r = blob
            #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            #     ax[stack-4].add_patch(c)
            #     ax[stack-4].set_axis_off()

            # if stack-4 >= 6:
            #     curr_pos = ax[stack-4].get_position()
            #     new_pos = [curr_pos.x0 + bottom_row_shift_amount, curr_pos.y0,  curr_pos.width, curr_pos.height]
            #     ax[stack-4].set_position(new_pos)

        
        # plt.savefig(os.path.join(figure_path,'KO.png'),dpi = 425)




        final_radii=final_radii.T
        final_intensities=final_intensities.T

        for i in range (len(final_radii)):
            if final_radii[i,0] < 6:
                final_radii[i,:]=np.nan
                final_intensities[i,:]=np.nan
        
        for i in range (len(final_radii)):
            counter=0
            for j in range (len(final_radii[i])):
                if np.isnan(final_radii[i][j]) == True:
                    counter=counter+1
            if counter>=6:
                final_radii[i,:]=np.nan
                final_intensities[i,:]=np.nan

        # i=0
        # while i < len(final_radii):
        #     if np.isnan(final_radii[i,10]):
        #         final_radii=np.delete(final_radii, i, 0)
        #         final_intensities=np.delete(final_intensities, i, 0)
        #         i=i-1
        #     i=i+1
        
        num_mps, num_timesteps = final_radii.shape

        nan_mps = []
        for mp_i in range(num_mps):
            if np.isnan(final_radii[mp_i,10]):
                nan_mps.append(mp_i)
        
        final_radii=np.delete(final_radii,np.array(nan_mps),0)
        final_intensities=np.delete(final_intensities,np.array(nan_mps),0)

        # i=0
        # while i < len(final_intensities):
        #     if math.isnan(final_intensities[i][10]) == True:
        #         final_intensities=np.delete(final_intensities, i, 0)
        #         i=i-1
        #     i=i+1

        row_list=np.concatenate((final_radii,final_intensities),axis=1)
        my_string = tif_file.rsplit('/')[-1]

        column_vector = np.array([my_string]*len(row_list)).reshape(-1,1)

        row_list=np.hstack((column_vector, row_list))

        with open(csv_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(row_list)



# %%





    #show_box_with_circle(all_boxes_next, good_box_idx_next, good_circles_next, idx2inspect = np.random.randint(len(good_box_idx)))

    # correct_coordinates=np.empty([len(good_box_idx_next), 3])
    # for center in range(len(good_circles_next)):
    #     x,y,r = good_circles_next[center]
    #     a=good_box_idx_next[center] 
    #     if y>=bounding_box_dims_next[0]/2:
    #         correct_new_y=good_circles_coordinates[a][0]+(y-bounding_box_dims[0]/2)
    #     else:
    #         correct_new_y=good_circles_coordinates[a][0]-(bounding_box_dims[0]/2-y)
    #     if x>=bounding_box_dims_next[0]/2:
    #         correct_new_x=good_circles_coordinates[a][1]+(x-bounding_box_dims[0]/2)
    #     else:
    #         correct_new_x=good_circles_coordinates[a][1]-(bounding_box_dims[0]/2-x)
        
    #     correct_coordinates[center] = [correct_new_y, correct_new_x, radii_next[center]]
    
    # intensities_next = compute_cell_statistics(img_16bit_next, correct_coordinates)


    # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
    # ax = axes.ravel()

    # ax[0].imshow(frame, cmap = 'gray', interpolation = 'bicubic')
    # ax[1].imshow(frame, cmap = 'gray', interpolation = 'bicubic')

    # for i in range(len(correct_coordinates)):
    #     y,x,r = correct_coordinates[i]
    #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    #     ax[1].add_patch(c)








    for i in range(len(correct_coordinates)):
        if np.sqrt((correct_coordinates[i][2] - final_radii[stack-3][i])**2) <=2 or np.sqrt((intensities_next[i] - final_intensities[stack-4][i])**2) <=500:
            good_circles_coordinates[i][0]=correct_coordinates[i][0]
            good_circles_coordinates[i][1]=correct_coordinates[i][1]
            final_radii[stack-4][i]=correct_coordinates[i][2]
            final_intensities[stack-4][i]=intensities_next[i]
        else:
            good_circles_coordinates = np.delete(good_circles_coordinates, i, 0)
            final_radii[stack-4][i]=np.nan
            final_intensities[stack-4][i]=np.nan           












            img_8bit = img_as_float(img_8bit)
            gimage = inverse_gaussian_gradient(img_8bit)
            # display_img(gimage)

            edge_pctile = 30
            threshold_g = np.percentile(gimage.flatten(),edge_pctile)
            img_thresholded_g = gimage < threshold_g
            gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
            # display_img(gimage_8bit_thr)

            contours, hierarchy = cv2.findContours(gimage_8bit_thr,  
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            gimage_8bit_contours = cv2.drawContours(gimage_8bit_thr, contours, -1, (0, 255, 0), 3)
            # display_img(gimage_8bit_contours)
            # all_cell_masks = create_cell_masks(img_16bit, contours, pts_threshold = 700)
            # display_img(all_cell_masks[:,:,12])

            img_binar = create_contour_mask(img_16bit, contours, pts_threshold = 800)
            # display_img(img_binar)
    
            img_16bit_cleaned = img_16bit.copy()
            img_16bit_cleaned[img_binar == 0] = 0
            # display_img(img_16bit_cleaned)
            
            blobs = blob_dog(img_16bit, min_sigma = 1, max_sigma=15, threshold=.01)
            blobs_list=[]
            for blob_info in blobs:
                y,x,r = blob_info
                if img_16bit_cleaned[int(y), int(x)] > 0: # this makes sure you only include blobs whose center pixel is on the mask  
                    blobs_list.append((y,x))
            bounding_box_dims = [50, 50] # set the height and width of your bounding box
            all_boxes = extract_boxes(img_16bit, blobs_list, bounding_box_dims)
            num_circles_to_find = 5
            hough_radii = np.arange(5, 50)
            hough_res = hough_circle_finder(all_boxes, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)
            good_circles_frame5, good_box_idx_frame5 = filter_circles(hough_res, num_circles_to_find, bounding_box_dims, 
                                            center_deviation_tolerance = 5, near_center_threshold = 2, radius_pct = 85)  
            radii_frame5=np.array(good_circles_frame5)[:,2]
            good_circles_original_idx_frame5 = np.hstack( (np.array(blobs_list)[good_box_idx_frame5,:], radii_frame5[:,np.newaxis]) )
            frame5_good_circles_coordinates = np.array(blobs_list)[good_box_idx_frame5,:]
            
            #Loop through all next frames

            if len(good_box_idx_frame5)>0:

                for stack in range(6,15):
                    frame = cell_cropped_fullstack[stack,:,:]
                    img_16bit=frame.copy()
                    img_8bit = img_as_ubyte(img_16bit)
                    max_y, max_x = img_8bit.shape
                    # display_img(img_8bit)
                
                    img_8bit = img_as_float(img_8bit)
                    gimage = inverse_gaussian_gradient(img_8bit)
                    # display_img(gimage)

                    edge_pctile = 30
                    threshold_g = np.percentile(gimage.flatten(),edge_pctile)
                    img_thresholded_g = gimage < threshold_g
                    gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
                    # display_img(gimage_8bit_thr)

                    bounding_box_dims = [60, 60] # slightly begger boxes
                    all_boxes = extract_boxes(img_16bit, frame5_good_circles_coordinates, bounding_box_dims) #boxes based on coordinates of blobs in frame 5
                    num_circles_to_find = 5
                    hough_radii = np.arange(5, 50)
                    hough_res = hough_circle_finder(all_boxes, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)
                    good_circles_next, good_box_idx_next = filter_circles(hough_res, num_circles_to_find, bounding_box_dims, 
                                                    center_deviation_tolerance = 15, near_center_threshold = 2, radius_pct = 85)  
                    radii=np.array(good_circles_next)[:,2]

                    # Problematic area. Finding coordinates on newly detected blobs on the whole image

                    correct_coordinates=np.empty([len(good_box_idx_next), 2])
                    for center in range(len(good_circles_next)):
                        y,x,r = good_circles_next[center]
                        a=good_box_idx_next[center] 
                        if y>=bounding_box_dims[0]/2:
                            correct_new_y=frame5_good_circles_coordinates[a][0]+(y-bounding_box_dims[0]/2)
                        else:
                            correct_new_y=frame5_good_circles_coordinates[a][0]-(bounding_box_dims[0]/2-y)
                        if x>=bounding_box_dims[0]/2:
                            correct_new_x=frame5_good_circles_coordinates[a][1]+(x-bounding_box_dims[0]/2)
                        else:
                            correct_new_x=frame5_good_circles_coordinates[a][1]-(bounding_box_dims[0]/2-x)
                        correct_coordinates[center]=[correct_new_y, correct_new_x]
                        
                        frame5_good_circles_coordinates = correct_coordinates

                        show_box_with_circle(all_boxes, good_box_idx_next, good_circles_next, idx2inspect = np.random.randint(len(good_box_idx)))


                        fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
                        ax = axes.ravel()

                        ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
                        ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')

                        for i in range(len(correct_coordinates)):
                            y,x = correct_coordinates[i]
                            r=radii[i]
                            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                            ax[1].add_patch(c)
                        
                        fig.suptitle(tif_file, fontsize=20)

                    # END OF NEW PART. DIDN'T MODIFY ANYTHING STARTING FROM THIS POINT YET
                    if len(good_box_idx) > 0:
                        radii=np.array(good_circles)[:,2]
                        good_circles_original_idx = np.hstack( (np.array(blobs_list)[good_box_idx,:], radii[:,np.newaxis]) )
                        intensities = compute_cell_statistics(img_16bit, good_circles_original_idx)

                        fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
                        ax = axes.ravel()

                        ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
                        ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')

                        for i in range(len(good_circles_original_idx)):
                            y,x,r = good_circles_original_idx[i]
                            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
                            ax[1].add_patch(c)
                        
                        fig.suptitle(tif_file, fontsize=20)
                    
                        radii_weights = radii/radii.sum() # create weights as probabilities
                        mean_dot_radius = radii.dot(radii_weights)
                        
                        mean_radius=np.mean(radii)
                        max_intensity=np.max(intensities)

                        median_radius=np.median(radii)
                        median_intensities=np.median(intensities)
                        max_radius=np.max(radii)
                        row_list.append([tif_file.rsplit('/')[-1], max_radius, mean_radius, mean_dot_radius, max_intensity, len(good_box_idx)])
                    else:
                        row_list.append([tif_file.rsplit('/')[-1], 0, 0, 0, 0, len(good_box_idx)])
    

    with open(csv_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

# %%
figure_path ='Z:/1st TAC'

# chunk_idx = (slice(1000,2000), slice(1000,2000))

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

axes[0,0].imshow(img_16bit, cmap = 'gray', interpolation='bicubic')
axes[0,0].set_title('Original image')
# axes[0,1].imshow(gimage, cmap = 'gray', interpolation='bicubic')
# axes[0,1].set_title('Inverse gaussian filter')

axes[0,1].imshow(img_hist_limited, cmap = 'gray', interpolation='bicubic')
axes[0,1].set_title('Thresholded image')

axes[0,2].imshow(gimage_8bit_thr, cmap = 'gray', interpolation='bicubic')
axes[0,2].set_title('Inverse gaussian')

axes[1,0].imshow(gimage_8bit_contours, cmap = 'gray', interpolation='bicubic')
axes[1,0].set_title('Inverse gaussian with drawn contours')

axes[1,1].imshow(img_binar, cmap = 'gray', interpolation='bicubic')
axes[1,1].set_title('Final segmentation')

axes[1,2].imshow(img_16bit_cleaned, cmap = 'gray', interpolation='bicubic')
axes[1,2].set_title('Background-subtracted image')
# axes[2,1].imshow(img_hist_limited[chunk_idx], cmap = 'gray', interpolation='bicubic')
# axes[2,1].set_title('Contrast-enhanced image')

plt.subplots_adjust(wspace=None, hspace=None)

fig.suptitle('Background segmentation and subtraction', fontsize=20)

plt.savefig(os.path.join(figure_path,'figure1b.png'),dpi = 425)

# %%
fig,axes = plt.subplots(2,2, figsize = (10,10), sharex = True, sharey = True)

axes[0,0].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
axes[0,0].set_title('Original image')
axes[0,1].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
axes[0,1].set_title('DoG blob detection')
for blob_i in blobs:
    y, x, r = blob_i.copy()
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    axes[0,1].add_patch(c)
axes[1,0].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
axes[1,0].set_title('Original image')
axes[1,1].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
axes[1,1].set_title('Blobs found on segmentation mask')
for filtered_blob in blobs_list:
    y, x = np.array(filtered_blob).copy()
    c = plt.Circle((x, y), 10, color='red', linewidth=2, fill=False)
    axes[1,1].add_patch(c)

plt.subplots_adjust(wspace=None, hspace=None)
fig.suptitle('Initial blob detection', fontsize=20)
plt.savefig(os.path.join(figure_path,'figure2b.png'),dpi = 425)

# %%
# idx2inspect = 0

# fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(all_boxes[good_box_idx[idx2inspect]], cmap = 'gray', interpolation = 'bicubic')
# ax[1].imshow(all_boxes[good_box_idx[idx2inspect]], cmap = 'gray', interpolation = 'bicubic')

# x, y, r = good_circles[idx2inspect]
# c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
# ax[1].add_patch(c)
# plt.show()

fig,axes = plt.subplots(2,2, figsize = (10,10))

axes[0,0].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
axes[0,0].set_title('Original image')
# axes[0,1].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
# axes[0,1].set_title('With final blob estimates')
# for i in range(len(good_circles_original_idx)):
#     y, x, r = good_circles_original_idx[i].copy()
#     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
#     axes[0,1].add_patch(c)
# axes[1,0].imshow(all_boxes[good_box_idx[0]], cmap = 'gray', interpolation = 'bicubic')
axes[0,1].imshow(all_boxes[good_box_idx[0]], cmap = 'gray', interpolation = 'bicubic')

x, y, r = good_circles[0]
c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
axes[0,1].add_patch(c)
# plt.show()

# axes[1,0].imshow(all_boxes[0], cmap = 'gray', interpolation = 'bicubic')
# x, y, r = good_circles[50]
# c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
# axes[1,0].add_patch(c)

axes[1,0].imshow(all_boxes[good_box_idx[1]], cmap = 'gray', interpolation = 'bicubic')
x, y, r = good_circles[1]
c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
axes[1,0].add_patch(c)

axes[1,1].imshow(all_boxes[good_box_idx[3]], cmap = 'gray', interpolation = 'bicubic')
x, y, r = good_circles[3]
c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
axes[1,1].add_patch(c)

# axes[2,1].imshow(all_boxes[good_box_idx[8]], cmap = 'gray', interpolation = 'bicubic')
# x, y, r = good_circles[8]
# c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
# axes[2,1].add_patch(c)

plt.subplots_adjust(wspace=None, hspace=None)
fig.suptitle('Final estimates and examples', fontsize=20)

plt.savefig(os.path.join(figure_path,'figure3b.png'),dpi = 425)


# %%
fig,axes = plt.subplots(3,2, figsize = (10,15))

axes[0,0].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
axes[0,0].set_title('Original image')
axes[0,1].imshow(img_16bit, cmap = 'gray',interpolation='bicubic')
axes[0,1].set_title('With final blob estimates')
for i in range(len(good_circles_original_idx)):
    y, x, r = good_circles_original_idx[i].copy()
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    axes[0,1].add_patch(c)
# axes[1,0].imshow(all_boxes[good_box_idx[0]], cmap = 'gray', interpolation = 'bicubic')
axes[1,0].imshow(all_boxes[good_box_idx[0]], cmap = 'gray', interpolation = 'bicubic')
# axes[1,1].imshow(all_boxes[good_box_idx[0]], cmap = 'gray', interpolation = 'bicubic')

x, y, r = good_circles[0]
c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
axes[1,0].add_patch(c)
# plt.show()

# axes[1,0].imshow(all_boxes[0], cmap = 'gray', interpolation = 'bicubic')
# x, y, r = good_circles[50]
# c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
# axes[1,0].add_patch(c)

axes[1,1].imshow(all_boxes[good_box_idx[8]], cmap = 'gray', interpolation = 'bicubic')
x, y, r = good_circles[8]
c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
axes[1,1].add_patch(c)

axes[2,0].imshow(all_boxes[good_box_idx[7]], cmap = 'gray', interpolation = 'bicubic')
x, y, r = good_circles[7]
c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
axes[2,0].add_patch(c)

axes[2,1].imshow(all_boxes[good_box_idx[6]], cmap = 'gray', interpolation = 'bicubic')
x, y, r = good_circles[6]
c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
axes[2,1].add_patch(c)

plt.subplots_adjust(wspace=None, hspace=None)
fig.suptitle('Final estimates and examples', fontsize=20)

plt.savefig(os.path.join(figure_path,'figure3c T9.png'),dpi = 425)

# %%
figure_dimensions = (15, 5) # found from experience that if the width (first entry of the tuple) is 3x bigger than
                      # the height (second entry of the tuple) then the space between first and second rows of 
                      # subplots is optimal. Play around with this though.

fig, axes = plt.subplots(2, 6, figsize=figure_dimensions, sharex=True, sharey=True)
ax = axes.ravel()
fig.delaxes(ax[-1]) # get rid of the last subplot (since there are only 11 slices total)

bottom_row_shift_amount = 0.07 # amount to shift the bottom row, horizontally -- play around with this

for ii in range(4,14):

    processed_slice = my_array[ii,:,:] # this is a stand-in for the analysis that takes place on each of the slices

    num_particles = np.random.randint(0,50) # this is the number of particles detected -- a function of your analysis algorithms

    cell_info_array = np.zeros( (num_particles, 3) ) # generate the array of y,x,r data (y position, x position, and radius)
    cell_info_array[:,0] = np.random.randint(0,size_y,num_particles,dtype=int)
    cell_info_array[:,1] = np.random.randint(0,size_x,num_particles,dtype=int)
    cell_info_array[:,2] = np.random.rand(num_particles)

    ax[ii].imshow(processed_slice, cmap = 'gray', interpolation = 'bicubic')

    for blob in cell_info_array:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax[ii].add_patch(c)
        ax[ii].set_axis_off()

    if ii >= 6:
        curr_pos = ax[ii].get_position()
        new_pos = [curr_pos.x0 + bottom_row_shift_amount, curr_pos.y0,  curr_pos.width, curr_pos.height]
        ax[ii].set_position(new_pos)

plt.draw()
