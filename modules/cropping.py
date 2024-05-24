# placeholder file for all methods to do with scan cropping
# including experimental methods 

import cv2
import re
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter

def find_cropping_coordinates(image: np.ndarray) -> [int, int, int, int]:
    """
    Find cropping coordinates based on white pixels in the input scan.

    Args:
        image (np.ndarray): Binary scan.

    Returns:
        tuple[int, int, int, int]: Coordinates for cropping (top, bottom, left, right).
    """
    # Find leftmost and rightmost bounds with white pixels
    leftmost_bound: int = (np.sum(image, axis=0) > 0).argmax()
    rightmost_bound: int = image.shape[1] - (np.sum(image, axis=0) > 0)[::-1].argmax()
    width: int = rightmost_bound - leftmost_bound
    
    top_whites: list[int] = []
    bottom_whites: list[int] = []
    
    # Divide the image into thirds horizontally and find white pixels in each third
    third_indices: list[tuple[int, int]] = [(leftmost_bound, leftmost_bound + int(width / 3)), 
                                            (leftmost_bound + int(width / 3), leftmost_bound + 2 * int(width / 3)), 
                                            (leftmost_bound + 2 * int(width / 3), rightmost_bound)]
    
    for start, end in third_indices:
        small_im: np.ndarray = image[:, start:end]
        white_pixels: np.ndarray = np.array(np.where(small_im == 255))
        first_white_pixel_h: int = white_pixels[:, 0][0]
        last_white_pixel_h: int = white_pixels[:, -1][0]
        
        top_whites.append(first_white_pixel_h)
        bottom_whites.append(last_white_pixel_h)
    
    top: int = int(np.median(top_whites))
    bottom: int = int(np.median(bottom_whites))
    height: int = bottom - top
    
    left_whites: list[int] = []
    right_whites: list[int] = []
    
    # Divide the image into thirds vertically and find white pixels in each third
    third_indices: list[tuple[int, int]] = [(top, top + int(height / 3)), 
                                            (top + int(height / 3), top + 2 * int(height / 3)), 
                                            (top + 2 * int(height / 3), bottom)]
    
    for start, end in third_indices:
        small_im: np.ndarray = image[start:end, :]
        white_pixels: np.ndarray = np.array(np.where(small_im == 255))
        first_white_pixel_w: int = white_pixels[:, 0][1]
        last_white_pixel_w: int = white_pixels[:, -1][1]
        
        left_whites.append(first_white_pixel_w)
        right_whites.append(last_white_pixel_w)
    
    return (top, bottom, int(np.median(left_whites)), int(np.median(right_whites)))

def crop_scan(filename: str, SOURCE_PATH: str, DEST_PATH: str) -> None:
    """
    Crop scan based on intensity thresholding and connected components analysis.

    Args:
        filename (str): The name of the input BUS scan file.
        SOURCE_PATH (str): Where the input BUS scan file can be found.
        DEST_PATH (str): Where the output (cropped) BUS scan should be saved.

    Returns:
        None
    """
    # Construct the full file path
    f: str = os.path.join(SOURCE_PATH, filename)
    
    # Check if the file exists and is a PNG image
    if os.path.isfile(f) and '.png' in filename:
        # Open the image and convert it to grayscale
        im: Image.Image = Image.open(f)
        image: np.ndarray = np.asarray(im.convert('L'))
        im: np.ndarray = np.asarray(im)
        
        mask3: np.ndarray = np.zeros_like(image)
        
        # Determine threshold for scan background based on mode
        mode: int = scipy.stats.mode(image, axis=None, keepdims=False)[0]
        
        mask3[image <= mode + 10] = 0
        mask3[image > mode + 10] = 255

        # Creating kernel for erosion and dilation
        kernel: np.ndarray = np.ones((3, 3), np.uint8)

        # Run 2 iterations of erosion and dilation on the mask
        for _ in range(2):
            mask3 = cv2.dilate(mask3, kernel)
            mask3 = cv2.erode(mask3, kernel)

        # Connected components analysis to find the largest component
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask3, connectivity=4)
        sizes: np.ndarray = stats[:, -1]

        # Identify the largest connected component
        max_label: int = 1
        try:
            max_size: int = sizes[1]
            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]
                    
            img2: np.ndarray = np.zeros(output.shape)
            img2[output == max_label] = 255
        
        except IndexError:
            img2: np.ndarray = np.ones(output.shape) * 255
        
        mask: np.ndarray = img2
        
        # Get top, bottom, left, and right coordinates based on white pixels
        top, bottom, left, right = find_cropping_coordinates(mask)
        top: int
        bottom: int
        left: int
        right: int
            
        # Crop the original image based on calculated coordinates
        cropped_image: np.ndarray = im[top:bottom, left:right]
        
        # Save the cropped image to the destination path
        Image.fromarray(cropped_image).save(os.path.join(DEST_PATH, filename))
        
def crop_to_bbox(im: Image.Image, bbox_list: list, show_thresh: bool = False):
    
    # Number of cropping boxes per row
    size = 6
    
    width, height = im.size
    box_width = width // size
    box_height = height // size
    
    box_nums = [] 
    flatten = []
    [flatten.extend(sublist) for sublist in bbox_list]
    
    
    # Stores all the boxes the bounding boxes reside in
    for i in range(size):
        for j in range(size):
            box_left = j * box_width
            box_upper = i * box_height
            box_right = box_left + box_width
            box_lower = box_upper + box_height
            
            for coord in flatten:
                if box_left <= coord[0] <= box_right and box_upper <= coord[1] <= box_lower:
                    box_nums.append((i,j))             
    
    # box_nums.append((1,1))
    box_nums = list(set(box_nums))   
    print("box_nums: ", box_nums)
    
    times_cropped_height = 0
    times_cropped_width = 0
    
    # Check if there are any boxes to crop in the same row or column
    i_vals, j_vals = zip(*box_nums)
    print("unpacked: ", i_vals, j_vals)
    
    same_value_coords_i = []
    same_value_coords_j = []
    for i, (x, y) in enumerate(box_nums):
        if i_vals.count(x) > 1:
            same_value_coords_i.append(x)
        if j_vals.count(y) > 1:
            same_value_coords_j.append(y)

    same_value_coords_i = list(set(same_value_coords_i))
    same_value_coords_j = list(set(same_value_coords_j))
    if same_value_coords_i or same_value_coords_j:
        print("i to crop:", same_value_coords_i)
        print("j to crop:", same_value_coords_j)
        
        # crop off the top/bottom
        for i in same_value_coords_i:
            times_cropped_height += i
            box_upper = i * box_height
            box_lower = box_upper + box_height
            if i < size / 2:
                im = im.crop((0, box_lower, im.width, im.height))
            if i >= size /2:
                im = im.crop((0, 0, im.width, box_upper))
                
        # crop off the right/left
        for j in same_value_coords_j:
            times_cropped_width += j
            box_left = j * box_width
            box_right = box_left + box_width
            if j < size / 2:
                print("cropa")
                im = im.crop((box_right, 0, im.width, im.height))
            else:
                print("cropb")
                im = im.crop((0, 0, box_left, im.height))
    
    box_nums = [(x, y) for x, y in box_nums if x not in same_value_coords_i and y not in same_value_coords_j]
    print("box_nums after removing ones in same row/col: ", box_nums)
    
    # Crop the rest of the boxes
    for box in box_nums:
        i, j = box
        i -= times_cropped_height 
        j -= times_cropped_width
        box_left = j * box_width
        box_upper = i * box_height
        box_right = box_left + box_width
        box_lower = box_upper + box_height
        
        # middle of the grid ex (2,2) (1,1)
        if i == j:
            # crop off the left if anno is on the left side. crop off the right otherwise
            if i < size / 2:
                if 0 < box_right < im.width:
                    print("crop1")
                    im = im.crop((box_right, 0, im.width, im.height))
                    times_cropped_width += j
                    
            else:
                if 0 < box_left < im.height:
                    print("crop2")
                    im = im.crop((0, 0, box_left, im.height))
                    times_cropped_width += j
                
        # lower triangle ex (1,0) (4,2)
        elif i > j:
            # crop off the left if anno is on the left side. crop off the bottom otherwise
            if i < size / 2:
                if 0 < box_right < im.width:
                    print("crop3")
                    im = im.crop((box_right, 0, im.width, im.height))
                    times_cropped_width += j
            else:
                if 0 < box_upper < im.height:
                    print("crop4")
                    im = im.crop((0, 0, im.width, box_upper))
                    times_cropped_height += i
        
        # upper triangle ex (0,2) (1,4)
        else:
            # crop off the top if anno is on the left side. crop off the right otherwise
            if i < size / 2:
                if 0 < box_lower < im.height:
                    print(im.height)
                    print("crop5")
                    im = im.crop((0, box_lower, im.width, im.height))
                    times_cropped_height += i
            else:
                if 0 < box_left < im.width:
                    print("crop6")
                    im = im.crop((0, 0, box_left, im.height))
                    times_cropped_width += j
                    
    return im