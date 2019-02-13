### Functions to help in preprocessing ###
import numpy as np
import cv2

def image_load_resize(filename, resize_width):
    '''Takes in filename as string and returns a square NumPy array of size resize_width'''
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (resize_width, resize_width), interpolation=cv2.INTER_AREA)
    return image_resized

def clean_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 20
    high_threshold = 50
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 3  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    y_1 = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope  = (y2 - y1) / (x2 - x1)
            if slope < 0.06 and slope > -0.06:
                region = gray.shape[0] - (gray.shape[0] * 0.30)
                if y1 > region:
                    y_1.append(y1)    
            else:
                continue
    return y_1

def remove_words_from_image(image_path):
    '''
    Read in image from path, remove words, and returns processed image
    '''
    
    image = cv2.imread(image_path)
    
    # Crop the image and check if it has white pixels
    bottom_percent = 0.25
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    img = image[bottom:image.shape[0], :]
    
    n_white_pix = np.sum(img >= 250)
    
    if n_white_pix >= 90000:
        y1 = clean_image(image)
        
        # Crop image
        if len(y1) != 0:
            min_y1 = min(y1)
            image_new = image[0:min_y1, 0:image.shape[1]]
            
            return image_new
        else:
            return image
    else:        
        return image