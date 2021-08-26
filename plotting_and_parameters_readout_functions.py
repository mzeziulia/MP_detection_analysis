import matplotlib.pyplot as plt
import numpy as np


def display_img(image_data, figsize = (16,9), cmap = 'gray'):
    '''
    Wrapper function for displaying images
    '''
    plt.figure(figsize=figsize)
    plt.imshow(image_data,cmap=cmap)


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

