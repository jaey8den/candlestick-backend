import cv2 as cv

def get_upload_details(img):

    # Convert image to gray and blur it
    src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))

    threshold = 94 # used a slider to find a optimal value
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE) # CCOMP > TREE, simpler mode as we only want to count candlesticks

    candlestick_contours = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:   # has no parent i.e remove contours within contours
            candlestick_contours.append(cnt)
    
    # find longest candlestick
    max_height = 0
    for cnt in candlestick_contours:
        x, y, w, h = cv.boundingRect(cnt)
        if h > max_height:
            max_height = h

    return len(candlestick_contours), max_height