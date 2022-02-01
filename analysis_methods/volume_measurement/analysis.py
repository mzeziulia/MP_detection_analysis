from . import utilities
import os
import glob
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
import csv
from skimage import img_as_float
from skimage.util import img_as_ubyte
from skimage.feature import blob_dog
from skimage.segmentation import inverse_gaussian_gradient



def run_volume_measurement (date_dir, last_frame, first_frame, conditions):

    """Function for macropinosome detection and circle fitting"""

    '''
    Arguments
    =============
    `date_dir` [str]:
        - directory containing files for analysis
    `last_frame` [int]:
        - number of the last frame that has to be analyzed
    `first_frame` [int]:
        - number of the first frame that has to be analyzed
    `conditions` [array]:
        - array containing listed conditions for analysis

    Returns
    ===========
    Writes in csv file radii and intensities of detected macropinosomes at every fame (first list of radii from first frame to last frame, then list of intensities from first frma to the last one)

    '''   

    for condition in conditions:
        csv_name = os.path.join(date_dir,'volume_%s.csv'%condition)
        row_list = [['Filename', '5min radius', '6min radius', '7min radius', '8min radius', '9min radius', '10min radius', '11min radius', '12min radius', '13min radius', '14min radius', '15min radius', '5min intensity', '6min intensity', '7min intensity', '8min intensity', '9min intensity', '10min intensity', '11min intensity', '12min intensity', '13min intensity', '14min intensity', '15min intensity']]
        for tif_file in glob.glob(os.path.join(date_dir,'*%s*'%condition)):

            cell_cropped_fullstack = io.imread(tif_file) # Read the image

            frame15 = cell_cropped_fullstack[last_frame,:,:] # Copy the last frame for analysis
            img_16bit=frame15.copy()
            img_8bit = img_as_ubyte(img_16bit)
            max_y, max_x = img_8bit.shape
            # utilities.display_img(img_8bit)

            ### Truncate intensity histograms at 95th and 15th percentiles ###

            max_pctile = 95
            max_val = np.percentile(img_16bit.flatten(), max_pctile)
            max_limited = img_16bit.copy()
            max_limited[img_16bit>max_val] = max_val
            # utilities.display_img(max_limited)

            threshold_pctile = 15
            threshold_d = np.percentile(img_16bit.flatten(), threshold_pctile) 

            ### Image segmentation ###

            img_hist_limited = max_limited.copy()
            img_hist_limited[img_hist_limited < threshold_d] = threshold_d
            # utilities.display_img(img_hist_limited)

            temp = img_as_float(img_hist_limited)
            gimage = inverse_gaussian_gradient(temp)

            ### Image thresholding ###

            edge_pctile = 30
            threshold_g = np.percentile(gimage.flatten(),edge_pctile)
            img_thresholded_g = gimage < threshold_g
            gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
            # utilities.display_img(gimage_8bit_thr)

            ### Contour search ###

            contours, hierarchy = cv2.findContours(gimage_8bit_thr,  
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            gimage_8bit_contours = cv2.drawContours(gimage_8bit_thr, contours, -1, (0, 255, 0), 3)
            # utilities.display_img(gimage_8bit_contours)

            img_binar = utilities.create_contour_mask(img_16bit, contours, pts_threshold = 800)
            # utilities.display_img(img_binar)

            img_16bit_cleaned = img_16bit.copy()
            img_16bit_cleaned[img_binar == 0] = 0

            ### First "dirty" macropinosome (circles) search ###

            blobs = blob_dog(img_16bit, min_sigma = 1, max_sigma=15, threshold=.01)

            blobs_list=[]
            for blob_info in blobs:
                y,x,r = blob_info
                if img_16bit_cleaned[int(y), int(x)] > 0: # this makes sure only blobs whose center pixel is on the mask are included
                    blobs_list.append((y,x))

            # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
            # ax = axes.ravel()
            # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
            # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
            # for filtered_blob in blobs_list:
            #     y, x = filtered_blob
            #     c = plt.Circle((x, y), 10, color='red', linewidth=2, fill=False)
            #     ax[1].add_patch(c)

            ### Crop images into 50x50 boxes around each macropinosome center detected with blob_dog and finding edges within each box. ### 
            ### Macropinosome is true if at least 1 of detected circles lies less than or equal to 5 pixels ###

            bounding_box_dims = [50, 50] # set the height and width of your bounding box
            all_boxes = utilities.extract_boxes(img_16bit, blobs_list, bounding_box_dims, max_x, max_y)

            num_circles_to_find = 5
            hough_radii = np.arange(3, 35)
            hough_res = utilities.hough_circle_finder(all_boxes, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)

            good_circles, good_box_idx = utilities.filter_circles(hough_res, num_circles_to_find, bounding_box_dims, 
                                            center_deviation_tolerance = 5, near_center_threshold = 2, radius_pct = 95)  


            radii=np.array(good_circles)[:,2]
            good_circles_original_info = np.hstack( (np.array(blobs_list)[good_box_idx,:], radii[:,np.newaxis]) )
            intensities = utilities.compute_cell_statistics(img_16bit, good_circles_original_info)
            good_circles_coordinates = np.array(blobs_list)[good_box_idx,:]
            
            ### Shows localization of macropinosomes ###
            
            # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
            # ax = axes.ravel()

            # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
            # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')

            # for i in range(len(good_circles_original_info)):
            #     y,x,r = good_circles_original_info[i]
            #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            #     ax[1].add_patch(c)

            ### Example of cirle fitting in the macropinosome ###
            # utilities.show_box_with_circle(all_boxes, good_box_idx, good_circles, idx2inspect = np.random.randint(len(good_box_idx)))

            final_radii = np.empty([last_frame-first_frame+1, len(radii)])
            final_intensities= np.empty([last_frame-first_frame+1, len(intensities)])

            final_radii[len(final_radii)-1]=radii
            final_intensities[len(final_radii)-1]=intensities


            ### The same analysis is repeated for each time frame ### 

            for stack in reversed(range(first_frame, last_frame)):
                frame = cell_cropped_fullstack[stack,:,:]
                img_16bit=frame.copy()
                img_8bit = img_as_ubyte(img_16bit)
                max_y, max_x = img_8bit.shape
                # utilities.display_img(img_8bit)

                max_pctile = 95
                max_val = np.percentile(img_16bit.flatten(), max_pctile)
                max_limited = img_16bit.copy()
                max_limited[img_16bit>max_val] = max_val
                # utilities.display_img(max_limited)

                threshold_pctile = 15
                threshold_d = np.percentile(img_16bit.flatten(), threshold_pctile)

                img_hist_limited = max_limited.copy()
                img_hist_limited[img_hist_limited < threshold_d] = threshold_d
                # utilities.display_img(img_hist_limited)

                temp = img_as_float(img_hist_limited)
                gimage = inverse_gaussian_gradient(temp)

                edge_pctile = 30
                threshold_g = np.percentile(gimage.flatten(),edge_pctile)
                img_thresholded_g = gimage < threshold_g
                gimage_8bit_thr=img_as_ubyte(img_thresholded_g)
                # utilities.display_img(gimage_8bit_thr)

                contours, hierarchy = cv2.findContours(gimage_8bit_thr,  
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                gimage_8bit_contours = cv2.drawContours(gimage_8bit_thr, contours, -1, (0, 255, 0), 3)
                # utilities.display_img(gimage_8bit_contours)

                img_binar = utilities.create_contour_mask(img_16bit, contours, pts_threshold = 800)
                # utilities.display_img(img_binar)

                img_16bit_cleaned = img_16bit.copy()
                img_16bit_cleaned[img_binar == 0] = 0

                blobs = blob_dog(img_16bit, min_sigma = 1, max_sigma=15, threshold=.01)

                blobs_list=[]
                for blob_info in blobs:
                    y,x,r = blob_info
                    if img_16bit_cleaned[int(y), int(x)] > 0: # this makes sure you only include blobs whose center pixel is on the mask  
                        blobs_list.append((y,x))

                ### This function only showes localization of macropinosomes, doesn't reflect true radius, only center of the vesicle ###
                # fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
                # ax = axes.ravel()
                # ax[0].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
                # ax[1].imshow(img_16bit, cmap = 'gray', interpolation = 'bicubic')
                # for filtered_blob in blobs_list:
                #     y, x = filtered_blob
                #     c = plt.Circle((x, y), 7, color='red', linewidth=2, fill=False)
                #     ax[1].add_patch(c)

                bounding_box_dims = [50, 50]
                all_boxes_next = utilities.extract_boxes(img_16bit, blobs_list, bounding_box_dims, max_x, max_y) #boxes based on coordinates of blobs in frame 5
                num_circles_to_find = 5
                hough_radii = np.arange(3, 35)
                hough_res_next = utilities.hough_circle_finder(all_boxes_next, num_circles_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100)
                good_circles_next, good_box_idx_next = utilities.filter_circles(hough_res_next, num_circles_to_find, bounding_box_dims, 
                                            center_deviation_tolerance = 5, near_center_threshold = 1, radius_pct = 95)  
                radii_next=np.array(good_circles_next)[:,2]

                good_circles_original_info = np.hstack( (np.array(blobs_list)[good_box_idx_next,:], radii_next[:,np.newaxis]) )


                ### Macropinosome is considered to be the same only if the distance between centers of macropinosomes from consecutive time frames did not exceed 30px ###
                
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
                intensities_next = utilities.compute_cell_statistics(img_16bit, correct_coordinates)
                
                for i in range(len(correct_coordinates)):
                    good_circles_coordinates[i][0]=correct_coordinates[i][0]
                    good_circles_coordinates[i][1]=correct_coordinates[i][1]
                    if correct_coordinates[i][2]>0:
                        final_intensities[stack-first_frame][i]=intensities_next[i]
                        final_radii[stack-first_frame][i]=correct_coordinates[i][2]
                    else:
                        final_intensities[stack-first_frame][i]=np.nan
                        final_radii[stack-first_frame][i] = np.nan

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

            num_mps, num_timesteps = final_radii.shape

            nan_mps = []
            for mp_i in range(num_mps):
                if np.isnan(final_radii[mp_i,10]):
                    nan_mps.append(mp_i)
            
            final_radii=np.delete(final_radii,np.array(nan_mps),0)
            final_intensities=np.delete(final_intensities,np.array(nan_mps),0)

            row_list=np.concatenate((final_radii,final_intensities),axis=1)
            my_string = tif_file.rsplit('/')[-1]

            column_vector = np.array([my_string]*len(row_list)).reshape(-1,1)

            row_list=np.hstack((column_vector, row_list))


            ### Writing data in csv ###
            with open(csv_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)

