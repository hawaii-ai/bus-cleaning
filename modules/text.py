# placeholder file for all methods to do with text extraction and 
# classification into categories

import cv2
import re
import numpy as np

from PIL import Image, ImageFont, ImageDraw, ImageFilter
import easyocr

def grab_prev_word_helper(starting_index: int, text: str):
    """
    Starts at a staring_index and traverses backwards through the text to find the left and right indices of the
    previous number of the starting word includes - and . characters with the number

    Args:
        starting_index (int): The first index of the starting word
        text (str): The text you are parsing through

    Returns:
        rl_index (arr): An array with index [0] being the left index and index [1] being the right index of the previous word
    """
    right_index = starting_index
    
    # get the index of the first char to the left of starting word (right_index)
    while right_index - 1 >= 0:
        right_index = right_index - 1
        if text[right_index] != ' ':
            break
            
    # get the index at the left end of target word (left_index)
    left_index = right_index
    while left_index - 1 >= 0:
        left_index = left_index - 1
        if text[left_index] == ' ':
            left_index = left_index + 1
            break
            
    rl_index = [left_index, right_index]
            
    return rl_index

def check_laterality(orig_str: str, show_thresh: bool):
    """
    Returns 'RIGHT' or 'LEFT' if there is an indication of either in orig_str. Removes indicators of laterality
    from the text and returns a new text string.

    Args:
        orig_str (str): The original string
        show_threash (bool): If True, show print statements for debugging

    Returns:
        new_info (arr): index 0 is the attribute that was searched for
        index 1 is the new string without those indicators
    """
    new_info = ['', orig_str]
    
    right_patt = r'(?:\b(?:[rp]\w*ht|rt)\b)|(?:\b[br]\w+\s*[br]reast\b)'
    left_patt = r'(?:\b(?:[lit]t|\w*(?:eft|fft))\b|\b[it]\+\B)|(?:\B\+\s*[br]reast)'
    
    rl_patt = [['RIGHT', right_patt], ['LEFT', left_patt]]
    for patt in rl_patt:
        matches = re.findall(patt[1], orig_str)
        if matches:
            new_info[1] = re.sub(patt[1], '', orig_str)
            new_info[0] = patt[0]
            
            if show_thresh:
                print("\nlaterality matches: ", matches)
                print('new text: ' + new_info[1])
                
            break
    return new_info

def check_axilla(orig_str: str, show_thresh: bool):
    """
    Returns '1' if there is an indicator of it in orig_str and '0' otherwise. Removes indicators of
    axilla from the text and returns a new text string

    Args:
        orig_str (str): The original string
        show_threash (bool): If True, show print statements for debugging

    Returns:
        new_info (arr): index 0 is the attribute that was searched for
        index 1 is the new string without those indicators
    """
    new_info = ['0', orig_str]
    
    axill_patt = r'\baxt[ih]\s+[\[\]\|il]?\s*[\[\]\|il]?4\b|\bax\w*[il]\w*\b'
    matches = re.findall(axill_patt, orig_str)
    if matches:
        new_info[1] = re.sub(axill_patt, '', orig_str)
        new_info[0] = '1'
        
        if show_thresh:
            print("\naxilla matches: ", matches)
            print('new text: ' + new_info[1])
            
    return new_info
            
def check_orientation(orig_str: str, show_thresh: bool):
    """
    Returns 'ANTIRADIAL', 'RADIAL', 'TRANSVERSE', 'SAGITTAL', 'OBLIQUE' if there is an indicator of it in orig_str.
    Removes indicators of transducer orientation from the text and returns a new text string

    Args:
        orig_str (str): The original string
        show_threash (bool): If True, show print statements for debugging

    Returns:
        new_info (arr): index 0 is the attribute that was searched for
        index 1 is the new string without those indicators
    """
    new_info = ['', orig_str]
    
    antirad_patt = r'\b[4a]\s*\w*r[4a][dn][li]?\b|[4a]nti'
    rad_patt = r'\br[4a][dn][li]?\w*\b'
    trans_patt = 'trans?|trns|trv'
    sag_patt = 'sa[gn]'
    obq_patt = 'obq'   

    orientation_patt = [['ANTIRADIAL', antirad_patt], ['RADIAL', rad_patt], ['TRANSVERSE', trans_patt], ['SAGITTAL', sag_patt], ['OBLIQUE', obq_patt]]
    for patt in orientation_patt:
        matches = re.findall(patt[1], orig_str)
        if matches:
            new_info[1] = re.sub(patt[1], '', orig_str)
            # can also filter out rad here since we would have just looked at anti
            new_info[1] = re.sub(rad_patt, '', new_info[1])
            new_info[0] = patt[0]
            
            if show_thresh:
                print("\norientation matches: ", matches)
                print('new text: ' + new_info[1])
                
            break
    return new_info

def check_cmfn(orig_str: str, show_thresh: bool):
    """
    Returns cmfn as 'X CMFN' where X is the number if there is an indicator of it in orig_str. Removes indicators of cmfn
    from the text and returns a new text string

    Args:
        orig_str (str): The original string
        show_threash (bool): If True, show print statements for debugging

    Returns:
        new_info (arr): index 0 is the attribute that was searched for
        index 1 is the new string without those indicators
    """
    new_info = ['', orig_str]
    
    cmfn_patt = r'(?:(?:[as\d]\s*\.\s*|\d)?[as\d]\s*-\s*)?(?:[as\d]\s*\.\s*|\d)?[as\d]\s*(?:c\s*[mn]?\s*f(?:\s*t)?\s*[nm]|cm[\w+|\/!]*n|fn|cm\s*f\w*\s*(?:t\w*\s*)?n\w*)'
    cmfn_patt_end = r'c\s*[mn]?\s*f(?:\s*t)?\s*[nm]|cm[\w+|\/!]*n|fn|cm\s*f\w*\s*(?:t\w*\s*)?n\w*'
    matches = re.findall(cmfn_patt, orig_str)
    
    if len(matches) > 0:
        
        # parse the number from the string and reformat
        match = matches[0]
        substring = ''
        for index, char in enumerate(match):
            
            # changes a to 4 and s to 5 to account for scanning errors
            match = match[:index] + '4' + match[index + 1:] if char == 'a' else match[:index] + '5' + match[index + 1:] if char == 's' else match
            if char != 'a' and char != 's' and char.isalpha():
                substring = match[:index]
                break

        if substring != '':
            # fix decimal count if needed
            if substring.count('.') == 1 and substring.count('-') == 1:
                substring = substring[orig_str.index('-'):]
            
            # remove any leading 0s and strip spaces
            substring = substring.lstrip('0').replace(" ", "")

            # if there's a dash at the front after this, remove it
            substring = substring[1:] if substring != '' and substring[0] == '-' else substring
            
            # if above 30, then that's not possible so set it to nothing
            substring = '' if substring.isdigit() and int(substring) >= 30 else substring
            
            if substring != '':
                new_info[0] = substring + ' CMFN'
                new_info[1] = re.sub(substring + r'\s*' + cmfn_patt_end, '', orig_str)

            if show_thresh:
                print("\ncmfn matches: ", matches)
                
    # filter out all instances of cmfn
    new_info[1] = re.sub(cmfn_patt_end, '', new_info[1])
    
    if show_thresh:
        print('new text: ' + new_info[1])
    
    return new_info

def check_lesion_dist_meas(orig_str: str, show_thresh: bool):
    """
    Returns '1' if there is an indicator of a lesion measurement annotation in orig_str, and '0' otherwise. Removes
    indicators of cmfn from the text and returns a new text string. This removal helps with the accuracy of scanning
    for clock, thus is designed to come before it

    Args:
        orig_str (str): The original string
        show_threash (bool): If True, show print statements for debugging

    Returns:
        new_info (arr): index 0 is the attribute that was searched for
        index 1 is the new string without those indicators
    """
    new_info = ['0', orig_str]

    all_matches = re.findall('cm', orig_str)
    new_info[0] = '1' if len(all_matches) > 0 else '0'
    for x in range(0, len(all_matches)):
        matches = re.search('cm', orig_str)
        
        prev_word_rl_index = grab_prev_word_helper(matches.span()[0], orig_str)
        left_index = prev_word_rl_index[0]
        right_index = prev_word_rl_index[1]
                
        if not matches.span()[0] == right_index == left_index:
            new_info[1] = orig_str[:left_index] + orig_str[matches.span()[1]:]
            
            if show_thresh:
                print("\nlesion dist matches (cm): ", all_matches)
                print('new text: ' + new_info[1])
                
    # Do the same with 'm m' pattern since some use millimeters
    if new_info[0] != '1':
        mm_pattern = r'\bm\s?m\b'
        matches = re.findall(mm_pattern, new_info[1])
        if matches:
            new_info[1] = re.sub(mm_pattern, '', new_info[1])
            new_info[0] = '1'
            
            if show_thresh:
                print("\nlesion dist matches (mm): ", matches)
                print('new text: ' + new_info[1])
            
    return new_info

def check_clock(orig_str: str, show_thresh: bool):
    """
    Returns the clock position value if there is an indicator of it in orig_str. Removes indicators of clock position
    from the text and returns a new text string. Check both analog and "o'clock" forms of writing clock position

    Args:
        orig_str (str): The original string
        show_threash (bool): If True, show print statements for debugging

    Returns:
        new_info (arr): index 0 is the attribute that was searched for
        index 1 is the new string without those indicators
    """
    new_info = ['', orig_str]
    
    clock_patt = r'\b[01]?\s?[0-9]\s?[:.;]?\s?[0qa3o]\s?[0qao]\b'
    clock_patt_alt = r'\b[1-9](?:(?:[03]\s?0)|(?:[q3]\s?q)|(?:[a3]\s?a))\b'
    analog = False
    matches = re.findall(clock_patt, orig_str)
    alt_matches = re.findall(clock_patt_alt, orig_str)
    
    if show_thresh:
        print('\nclock matches: ', matches)
        print('alt clock matches (3 digits ex.600): ', alt_matches)
        
    # will only take it if str len >= 4 (this to not accidentally take in decimals)
    if (matches and len(matches[-1]) >= 4) or alt_matches:
        
        matches = matches if matches else alt_matches
        
        # grab last two chars
        no_space = matches[-1].replace(' ', '')
        hour = no_space[:-2]            
        hour = re.sub('[:.]', '', hour)
        
        # remove all non number chars
        hour = re.sub(r'\D', '', hour)
        
        # remove any leading 0s
        hour = hour.lstrip('0').replace(" ", "")
        
        #check for 30 min or 00 min and set
        new_info[0] = hour + ':30' if no_space[-2:][0] == '3' else hour + ':00'
        analog = True
        
        # sub whole string out to clean
        new_info[1] = re.sub(clock_patt, '', orig_str)
    
    # Check for clock position (o'clock form)
    if not analog:
        oclock_patt = r'[0o]\'clock\b'
        matches = re.search(oclock_patt, orig_str)
        if matches:
            prev_word_rl_index = grab_prev_word_helper(matches.span()[0], orig_str)
            left_index = prev_word_rl_index[0]
            right_index = prev_word_rl_index[1]

            if not matches.span()[0] == right_index == left_index:
                substring = text[left_index:right_index+1]
                
                # remove all non number chars
                substring = re.sub(r'\D', '', substring)
                
                # remove any leading 0s
                substring = substring.lstrip('0').replace(" ", "")
                
                new_info[0] = substring + ':00'
                # sub whole string out to clean
                new_info[1] = orig_str[:left_index] + orig_str[matches.span()[1]:]
                
                if show_thresh:
                    print('\nclock matches (oclock): ', matches)
                
    if show_thresh:
        print('new text: ' + new_info[1])
        
    return new_info
        
def check_procedural_imaging(orig_str: str, show_thresh: bool):
    """
    Returns a '1' if there is an indicator of procedural imaging in orig_str, and '0' otherwise. This means biopsy or implant
    presence. Removes indicators of procedural imaging from the text and returns a new text string.

    Args:
        orig_str (str): The original string
        show_threash (bool): If True, show print statements for debugging

    Returns:
        new_info (arr): index 0 is the attribute that was searched for
        index 1 is the new string without those indicators
    """
    new_info = ['0', orig_str]
    
    procedural_patt = ['[br]x', '[br]iop', 'stereo', 'marquee', 'core', 'tru', 'pass', 'fire', 'celero', r'pre\w*\b', r'\bpost\b', 's[il\/]p', 'coil', 'ribbon', 'wing', 'bard', 'twirl', 'clip', 'tumark', 'vision']
    for patt in procedural_patt:
        matches = re.search(patt, orig_str)
        if matches:
            new_info[0] = '1'
            new_info[1] = re.sub(patt, '', new_info[1])
            
            if show_thresh:
                print("\nprocedural imaging match: ", matches.group())
                print('new text: ' + new_info[1])
    return new_info
                

def text_helper(text: list, show_thresh: bool = False):
    """
    Gets important information from the text in the following format:
        [laterality, orientation, cmfn, clock position, axilla flag, lesion dist measurement flag, prodecural imaging flag, misc text, full scanned string]

    Args:
        text (list): The list of text to check
        show_thresh (bool): Make true to print for debugging

    Returns:
        results (arr): List of results; each index corresponds to a certain attribute
        if there is nothing found for a particular attribute, the value will be ''
    """
    # Initialize results array with the number of possible attributes text can have
    # [laterality, orientation, cmfn, clock position, axilla flag, lesion dist measurement flag, prodecural imaging flag, misc, full string]
    results = [''] * 9
    
    # full string
    text = " ".join(text)
    results[8] = text
    
    # Make text lowercase
    text = text.lower()
    
    if show_thresh:
        print('\nraw scanned text:')
        print(text)
    
    # Laterality
    results[0], text = check_laterality(text, show_thresh)

    # Filter out the word breast
    text = re.sub(r'\b[br]r[a-z]*t?\b', '', text)
    
    # Axilla
    results[4], text = check_axilla(text, show_thresh)
    
    # Orientation
    results[1], text = check_orientation(text, show_thresh)
    
    # Cmfn
    results[2], text = check_cmfn(text, show_thresh)
    
    # Lesion dist measurement
    # Also serves to filter out cm and their numbers which helps accuracy of clock scanning
    results[5], text = check_lesion_dist_meas(text, show_thresh)
    
    # clock
    results[3], text = check_clock(text, show_thresh)
    
    # procedural imaging
    results[6], text = check_procedural_imaging(text, show_thresh)
    
    # misc
    text = " ".join(text.split())
    results[7] = text
    
    return results


def get_text_attributes(filename: str, show_thresh: bool = False):
    """
    Identifies the location of text in a scan and parses out relevant information from get_text_attributes

    Args:
        filename (str): The name of the scan
        show_thresh (bool): If True, show prints and images for debugging

    Returns:
        results (arr): List of results, with the bounding box coordinates for each text box, the original
                       scanned text, and the return value from get_text_attributes
    """
    
    # Read image
    f = filename
    img = Image.open(f)
    width, height = img.size
    im = np.array(img)
    
    # Add black padding at bottom to help with accuracy
    for x in range(0, 70):
        im[0] = 0
        appending_black = [im[0]]
        im = np.append(im, appending_black, axis=0)
        
    # Create arrays to hold text
    orig_text = []
    cleaner_text = []
    
    # Results array
    print_results = []
    bounding_box_coord = []
    
    # Create reader
    reader = easyocr.Reader(['en'], gpu=True)
    
    # Read text
    full_text = reader.readtext(im, x_ths=2.4, paragraph=True, mag_ratio=2.1, low_text=0.2, link_threshold=0.35)
    
    for t in full_text:
        bbox, text = t
        bbox = np.asarray(bbox, dtype='int')
        
        orig_text.append(text)
        
        # bbox[0], bbox[1], bbox[2], bbox[3]
        # tp left, tp right, btm right, btm left

        # filter out "dead" text (keep text in the bottom 7/8ths of image and > 2 chars)
        if bbox[2][1] > height/8 and len(text) > 2 and re.match(r'^\W+$', " ".join(text.split())) is None:
            cv2.rectangle(im, bbox[0], bbox[2], (255, 255, 255), 2)
            cleaner_text.append(text)
            
            # Round down coordinates that go over the max height
            if bbox[2][1] > height:
                bbox[2][1] = height

            # Add text and coordinates to result array)
            print_results.append([[bbox[0], bbox[1], bbox[2], bbox[3]], text])
            bounding_box_coord.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            
        if show_thresh:
            # Draw circles where the cropping coordinates are
            cv2.circle(im, bbox[0], 5, (255, 255, 255), 2)
            cv2.circle(im, bbox[2], 5, (255, 255, 255), 2)
            
    # Get important info from text
    text_attributes = text_helper(cleaner_text, show_thresh)
        
    # Format original text
    orig_text = ' '.join(orig_text)
    
    if show_thresh:
        print('\nresults from each box:')
        for r in print_results:
            bbox, text = r
            print("top left: %s, bottom right: %s, text: %s" %(bbox[0], bbox[1], text))
        
        print('\nattributes array:')
        print(text_attributes)
        print("[laterality, orientation, cmfn, clock position, axilla flag, lesion dist measurement flag, prodecural imaging flag, misc, full string]")
        
        display(Image.fromarray(im))
    
    return bounding_box_coord, orig_text, text_attributes