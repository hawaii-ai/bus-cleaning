# placeholder file for all methods remaining 

import cv2
import re
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter

def is_scan_valid(im: Image.Image) -> int:
    """
    Identifies black images based on the percentage of black pixels in the image.
    
    Args:
        im (Image.Image): PIL RGB Image object of the scan.
        
    Returns:
        int: If the scan is invalid. 
             Returns 0 if the scan is invalid.
             Returns 1 if the scan is NOT invalid (is valid).
    """
    img_g: np.ndarray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    height: int
    width: int
    height, width = img_g.shape
    
    # Count black pixels in the image
    n_zeros: int = np.count_nonzero(img_g < 5)
    
    # If more than 75% of the image is black, mark it as 'DROP', else keep the filename
    if n_zeros > 0.75 * height * width:
        return 0
    else:
        return 1
    
def num_views_in_scan(im: Image.Image) -> int:
    """
    Determines the number of views in a given BUS scan PNG.
    
    Args:
        im (Image.Image): PIL RGB Image object of the scan.
        
    Returns:
        int: Number of views in the scan. 
             Returns 1 if the scan contains a single view.
             Returns 2 if the scan contains two views.
    """
    img2: np.ndarray = np.array(im)
    img_g: np.ndarray = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    
    # Define color ranges for green and teal masks
    lower_green: np.ndarray = np.array([148, 226, 164], np.double)
    upper_green: np.ndarray = np.array([150, 228, 166], np.double)
    green_mask: np.ndarray = cv2.inRange(img2, lower_green, upper_green)
    
    lower_teal: np.ndarray = np.array([0, 152, 152], np.double)
    upper_teal: np.ndarray = np.array([0, 154, 154], np.double)
    teal_mask: np.ndarray = cv2.inRange(img2, lower_teal, upper_teal)
    
    # Identify the edges in the image
    edges: np.ndarray = cv2.Canny(img_g, 25, 100, apertureSize=3)
    height: int
    width: int
    height, width = edges.shape
    # Identify slices spanning the dividing line
    slice1: np.ndarray  = edges[:, int(width/2)]
    slice2: np.ndarray  = edges[:, int(width/2) + 10]
    slice3: np.ndarray  = edges[:, int(width/2)- 10]
    
    # Idenitfy if we have a single-view elastography scan
    if((green_mask == 255).sum() >= height):
        return 1
    if((teal_mask == 255).sum() < 150 and (teal_mask == 255).sum() > 0):
        return 1
    
    # Pull out topmost slice of the image 
    top_slice: np.ndarray = edges[0, :]
    
    # Do we have a) less than 10 edge pixels in the top slice 
    # and b) more than 3 edge pixels in the top slice?
    if (((top_slice == 255).sum() < 10) and ((top_slice == 255).sum() > 3)):
        # denoise the Canny edge results
        res = np.array(tuple(i/10 * j/10 for i, j in zip(top_slice, top_slice[1:])))
        # if we have less than 3 pixels which edge pixels 
        if ((res != 0).sum() < 3):
            return 1
        else:
            return 2
        
    # Is the image at least as wide as it is tall? 
    if (width < height*0.75):
        return 1

    # if more than 100 pixels down the center of the image are recognized as an edge 
    if (((slice1 == 255).sum() > 100) and ((slice1 == 255).sum() > (slice2 == 255).sum() + 10) and ((slice1 == 255).sum() > (slice3 == 255).sum() + 10)):
        return 2
    else:
        return 1
    
def is_scan_grayscale(im: Image.Image) -> bool:
    """
    Check if all channels (R, G, and B) are the same for every pixel in the given RGB image, 
    indicating that it is grayscale.

    Args:
        im (Image.Image): RGB PIL image.

    Returns:
        bool: True if all channels are the same for every pixel, False otherwise.
    """
    # Get pixel data
    pixels = im.load()
    
    # Iterate through each pixel and check if R, G, and B channels are the same
    width, height = im.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            if r != g or r != b or g != b:
                return False
            
    # If all pixels have the same R, G, and B values
    return True

def is_not_B_mode(im: Image.Image, show_thresh: bool = False):
    """
    Identifies if an image is a B-mode scan or non-B-mode scan (doppler and elastography).

    Args:
        im (Image.Image): RGB PIL image.
        show_thresh (bool): If True, show prints and images for debugging

    Returns:
        int: If the scan has bloodflow Doppler highlighting. 
             Returns 1 if the scan has Doppler highlighting.
             Returns 0 if the scan does NOT have Doppler highlighting.
    """
    if is_scan_grayscale(im):
        return [0, 'gray'] if show_thresh else 0

    array_im: np.ndarray = np.array(im)
    img_hsv: np.ndarray = cv2.cvtColor(array_im, cv2.COLOR_RGB2HSV)
    width, height = im.size

    # Thresholding for different colors to detect Doppler or lesion
    thresh_green: np.ndarray = cv2.inRange(img_hsv, (36, 50, 70), (89, 255, 255))
    thresh_orange: np.ndarray = cv2.inRange(img_hsv, (10, 50, 70), (24, 255, 255))
    thresh_yellow: np.ndarray = cv2.inRange(img_hsv, (25, 50, 70), (35, 255, 255))
    thresh_red: np.ndarray = cv2.inRange(img_hsv, (0, 50, 70), (9, 255, 255))
    thresh_blue: np.ndarray = cv2.inRange(img_hsv, (90, 50, 70), (128, 255, 255))
    thresh_white: np.ndarray = cv2.inRange(img_hsv, (0, 0, 130), (180, 18, 255))

    # Combine all color thresholds for Doppler image
    thresh_img: np.ndarray = thresh_orange + thresh_green + thresh_yellow + thresh_red + thresh_blue
    
    detected_boxes = []
    
    # If the image isnt all black
    if not np.all((thresh_white == 0), axis=None):
        
        # Look for boxes in green and white. running green first is more accurate
        # to minimize risk of counting tissue as a box
        box_colors = [thresh_green, thresh_white]
        for color in box_colors:
            
            # Dilate the image
            kernel = np.ones((6, 6), np.uint8)
            dilatedimg = cv2.dilate(color, kernel, iterations=1)
            
            if show_thresh:
                print("dilated color threshold:")
                display(Image.fromarray(dilatedimg))     
            testimg = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Find contours
            contours, hierarchy = cv2.findContours(dilatedimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                
                approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True), True)
                
                if show_thresh:
                    cv2.drawContours(testimg, approx, -1, (0, 255, 0), 2)
                
                # Classify as a box if there are 4 or 5 points forming of a significant area
                if ((len(approx) == 4 or len(approx) == 5) and cv2.contourArea(approx) > 0.06*width*height):
                    
                    if show_thresh:
                        x, y, w, h = cv2.boundingRect(approx)
                        cv2.putText(testimg, str(len(approx)), (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        print("points forming significant area:")
                        display(Image.fromarray(testimg))
                            
                    return [1, 'box'] if show_thresh else 1

            # Edge detection
            linestest = np.zeros((height, width, 3), dtype=np.uint8)
            
            dst = cv2.Canny(dilatedimg, 50, 200, None, 3)
        
            lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 150, minLineLength=50, maxLineGap=10)
            
            filtered_lines = []
            height_threshold = height * 0.4
            width_threshold = width * 0.4
            
            if lines is not None:
                for line in lines:
                    
                    # Extract line endpoints
                    x1, y1, x2, y2 = line[0]
                    
                    if show_thresh:
                        cv2.line(linestest, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Calculate line length using endpoints distance formula
                    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    
                    # Calculate slope (avoid division by zero)
                    if x2 - x1 != 0:
                        slope = abs((y2 - y1) / (x2 - x1))
                    else:
                        slope = np.inf  # Assign infinite slope for vertical lines
                        
                    # Classify lines based on slope for perfect horizontal or vertical lines
                    if slope < 0.1:
                        if line_length >= width_threshold:
                            filtered_lines.append(line)
                    elif slope > np.pi / 2:
                        if line_length >= height_threshold:
                            filtered_lines.append(line)
                            
            if show_thresh:
                print("vertical/horizontal lines detected > 40% of width or height: ")
                display(Image.fromarray(linestest))
                            
            if len(filtered_lines) != 0:
                return [1, 'cut off box'] if show_thresh else 1
        
    # Cropped image for blood flow highlighting checking (color)
    width_crop = int(width * 0.1)
    height_crop = int(height * 0.1)
    bottom_crop = int(height * 0.2)
    cropped_img = thresh_img[height_crop:height-bottom_crop, width_crop:width-width_crop]
    if show_thresh:
        print("cropped image for color checking: ")
        display(Image.fromarray(cropped_img))
    # Check if there are values which match our HSV color values covering more than 0.5% of the image
    if((cropped_img == 255).sum() > (cropped_img.shape[0]*cropped_img.shape[1]*0.005)):
        return [1, 'color'] if show_thresh else 1
    
    return [0, 'nothing'] if show_thresh else 0

def B_mode_wrapper(im: Image.Image, show_thresh: bool = False):
    """
    Identifies if an image is a B-mode scan or non-B-mode scan, taking into account
    double scans.

    Args:
        im (Image.Image): PIL RGB Image object of the scan.
        show_thresh (bool): If True, show prints and images for debugging

    Returns:
        results (arr): d
    """
    array_im: np.ndarray = np.array(im)
    width, height = im.size
    
    if num_views_in_scan(im) == 2:
        
        if show_thresh:
            print("double scan\n")
        
        half_width = width // 2
        left_half = array_im[:, :half_width]
        right_half = array_im[:, half_width:]
        
        result = is_not_B_mode(Image.fromarray(left_half), show_thresh)
        
        if result[0] == 0:
            result = is_not_B_mode(Image.fromarray(right_half), show_thresh)
            
    else:
        result = is_not_B_mode(im, show_thresh)
        
    return result

def enhance_image(im: Image.Image, show_thresh: bool = False) -> int:
    """
    Enhances image to prepare for lesion annotation detection

    Args:
        im (Image.Image): RGB PIL image.
        show_thresh (bool): If True, show prints and images for debugging

    Returns:
        Image.Image: The enhanced image
    """
    dilate = 1
    
    width, height = im.size
    
    # crop a percentage off each edge of the image to get rid of extraneous overlays
    # original size of the image is retained, just filled with black
    crop_percentage = 0.15
    crop_pixels_height = int(height * crop_percentage)
    crop_pixels_width = int(width * crop_percentage)
    cropped_image = np.array(im)[crop_pixels_height:height-crop_pixels_height, crop_pixels_width:width-crop_pixels_width]
    new_image = np.zeros_like(im)
    new_image[crop_pixels_height:height-crop_pixels_height, crop_pixels_width:width-crop_pixels_width] = cropped_image
    new_image = Image.fromarray(new_image)
    
    # fill with black pixels
    
    new_image = new_image.filter(ImageFilter.FIND_EDGES)
    if show_thresh:
        display(new_image)
    
    array_im: np.ndarray = np.array(new_image)
    array_im[crop_pixels_height - 1:crop_pixels_height + 1,:] = False
    array_im[height - crop_pixels_height - 1:height - crop_pixels_height + 1,:] = False
    array_im[:,crop_pixels_width - 1:crop_pixels_width + 1] = False
    array_im[:,width - crop_pixels_width - 1:width - crop_pixels_width + 1] = False
    
    array_im = cv2.cvtColor(array_im, cv2.COLOR_RGB2GRAY)
    
    _, binary_image = cv2.threshold(array_im, 210, 255, cv2.THRESH_BINARY)
    
    enhance = (Image.fromarray(binary_image)).filter(ImageFilter.MaxFilter)
    
    enhance = np.array(enhance)
    for i in range(dilate):
        enhance = cv2.dilate(enhance, np.ones((2, 2), np.uint8))
    enhance = Image.fromarray(enhance)
    
    if show_thresh:
        display(enhance)
    
    return enhance

def detect_anno(im: Image.Image, show_thresh: bool = False):
    """
    Detects coordinates of lesion annotations in an image

    Args:
        im (Image.Image): RGB PIL image.
        show_thresh (bool): If True, show prints and images for debugging

    Returns:
        coord_list (arr): A list of the coordinates for the bounding boxes of the lesion annotations in the
                          following format: [top_left, top_right, bottom_right, bottom_left]
    """
    
    # can add option to pass in text coordinates, if it overlaps with coord_list then don't return that coord (+-1 or a little more)
    
    coord_list = []
    double_check_list = []
    
    width, height = im.size
    
    contours, _ = cv2.findContours(np.uint8(im), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    im = np.array(im)
    rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    for cnt in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(cnt)
        
        cond = ((h <= 70) and (h >= 10)) and ((w <= 70) and (w >= 10))
        if cond:
            
            top_left = (x, y)
            top_right = (x + w, y)
            bottom_right = (x + w, y + h)
            bottom_left = (x, y + h)

            coord_list.append([top_left, top_right, bottom_right, bottom_left])

            cv2.rectangle(rgb_image, (x, y), (x+w, y+h), (0,0,255), 2)
            
    # cv2.drawContours(rgb_image, contours, -1, (0,255,0), 3)
    
    # Display grid lines for cropping (to be used for debugging with crop_to_bbox)
    # if show_thresh:
    #     size = 6
    #     box_width = width // size
    #     box_height = height // size
    #     for i in range(size):
    #         for j in range(size):
    #             box_left = j * box_width
    #             box_upper = i * box_height
    #             box_right = box_left + box_width
    #             box_lower = box_upper + box_height
    #             cv2.rectangle(rgb_image, (box_left, box_upper), (box_right, box_lower), (255, 255, 255), 2)
                
    if show_thresh:
        display(Image.fromarray(rgb_image))
        
    return coord_list

def detect_anno_BUSI(im: Image.Image, show_thresh: bool = False):
    
    coord_list = detect_anno(im, show_thresh)
    im = np.array(im)

    # if we didn't detect anything, we're going to run through an additional check 
    if len(coord_list) < 1:
        # additional checking step for finding the cross-style annotations which we found this failed on 
        lines = cv2.HoughLinesP(im, rho=1, theta= np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)
        if lines is not None:
            lines = lines[:, 0, :] 

            for line1, line2 in zip(lines, lines[1:]):
                image_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                x1a, y1a, x2a, y2a = line1
                cv2.line(image_color, (x1a, y1a), (x2a, y2a), (0, 0, 255), 2)
                x1b, y1b, x2b, y2b = line2
                cv2.line(image_color, (x1b, y1b), (x2b, y2b), (255, 0, 0), 2)

                intersection = find_intersection(line1, line2)
                if intersection:
                    coord_list = [min(x1a, x1b), min(y1a, y1b), max(x2b, x2a), max(y2a, y2b)]
                    
                if show_thresh:
                    image_pil = Image.fromarray(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
                    display(image_pil)

    return coord_list

def find_intersection(line1, line2):
    """
    Find the intersection of two lines given in the form ((x1, y1, x2, y2), (x1, y1, x2, y2)).
    Returns the intersection point as (x, y) or None if the lines do not intersect within the line segments.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Convert lines to vectors
    A = np.array([[x2 - x1, x3 - x4],
                  [y2 - y1, y3 - y4]])
    
    # Check if lines are parallel
    if np.linalg.det(A) == 0:
        return None

    B = np.array([[x3 - x1],
                  [y3 - y1]])

    t, s = np.linalg.solve(A, B)

    # Check if the intersection point lies on both line segments
    if 0 <= t <= 1 and 0 <= s <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return (intersection_x, intersection_y)
    
    return None
