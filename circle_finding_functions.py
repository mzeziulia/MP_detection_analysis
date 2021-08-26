import numpy as np
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

def extract_boxes(img_data, coordinates_list, dims):
    '''
    Docstring @TODO
    '''

    all_boxes = []
    for center_i in coordinates_list:
        y, x = center_i
        box_i = img_data[ max(0, int(y - dims[0]/2)) : min(max_y, int(y + dims[0]/2)), 
                        max(0, int(x - dims[1]/2)) : min(max_x, int(x + dims[1]/2))]
        all_boxes.append(box_i)
    
    return all_boxes

def hough_circle_finder(extracted_boxes, num_peaks_to_find, hough_radii, sigma = 4, low_threshold=0, high_threshold=100):
    '''
    Docstring @TODO
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

def filter_circles(all_circles, num_peaks_to_find, dims, center_deviation_tolerance = 3, near_center_threshold = 4, radius_pct = 85):
    '''
    Docstring @TODO
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
    '''
    Docstring @TODO
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

