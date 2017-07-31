import cv2
import numpy as np
from matplotlib import pyplot as plt
import pygame # For cropping image
pygame.init()

# Parameters
cam_port = 0 # Which camera to use? 0 is the fixed webcam, 1 is the usb webcam
cam_cal_photo = 'Calibrate.png' # Name of the calibration photo
cam_cal_crop = 'Calibrate_crop.png' # Name of the cropped calibration photo
photo_size_x = 720 # Photo size in x direction
photo_size_y = 600 # Photo size in y direction


# Functions
def nothing(x):
    pass

def setup(path):
    # Load new image into file
    px = pygame.image.load(path)
    # Scale the image
    px = pygame.transform.scale(px, (photo_size_x, photo_size_y))
    # Initialize a window or screen for display
    screen = pygame.display.set_mode( px.get_rect()[2:] )
    # draw one image onto another
    screen.blit(px, px.get_rect())
    # Update the full display Surface to the screen
    pygame.display.flip()
    return screen, px

def displayImage(screen, px, topleft, prior):
    # ensure that the rect always has positive width, height
    x, y = topleft
    width =  pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current
    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()


def mainLoop(screen, px):
    # Read the image
    img = cv2.imread(cam_cal_photo)
    img = cv2.resize(img, (photo_size_x, photo_size_y))

    # Initialize the extrema of the color
    hm = 255
    hp = 0
    sm = 255
    sp = 0
    vm = 255
    vp = 0

    topleft = prior = None
    end_sampling = 0

    while end_sampling!=1: # keep reading
        for event in pygame.event.get():
            # print event

            if event.type == pygame.KEYUP:
                end_sampling = 1

            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                    left = topleft[0]
                    upper = topleft[1]
                    # print topleft
                else:
                    # right, lower = event.pos
                    bottomright = event.pos

                    right = bottomright[0]
                    lower = bottomright[1]

                    # ensure output rect always has positive width, height
                    if right < left:
                        left, right = right, left
                    if lower < upper:
                        lower, upper = upper, lower

                    crop_img = img[upper:lower, left:right]
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                    # cv2.imwrite(cam_cal_crop, crop_img)

                    if crop_img[:, :, 0].min() < hm:
                        hm = crop_img[:, :, 0].min()
                    if crop_img[:, :, 0].max() > hp:
                        hp = crop_img[:, :, 0].max()

                    if crop_img[:, :, 1].min() < sm:
                        sm = crop_img[:, :, 1].min()
                    if crop_img[:, :, 1].max() > sp:
                        sp = crop_img[:, :, 1].max()

                    if crop_img[:, :, 2].min() < vm:
                        vm = crop_img[:, :, 2].min()
                    if crop_img[:, :, 2].max() > vp:
                        vp = crop_img[:, :, 2].max()

                    print hm, hp, sm, sp, vm, vp

                    topleft = None
                    prior = None
            if topleft:
                # First function of this script file
                prior = displayImage(screen, px, topleft, prior)
    return hm, hp, sm, sp, vm, vp



# Main function below
if __name__ == "__main__":
    # print 'hi'
    # Create track bars
    # cv2.namedWindow('Trackbar', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Trackbar', 300, 200)
    # cv2.createTrackbar('erode', 'Trackbar', 1, 20, nothing)
    # cv2.createTrackbar('dilate', 'Trackbar', 1, 20, nothing)
    # cv2.createTrackbar('r-', 'Trackbar', 1, 50, nothing)
    # cv2.createTrackbar('r+', 'Trackbar', 1, photo_size_y, nothing)
    # cv2.createTrackbar('dmin', 'Trackbar', 1, 50, nothing)

    # Open the camera to take a photo and save it with the name cam_cal_photo
    camera = cv2.VideoCapture(cam_port)
    # retval, im = camera.read()
    # cv2.imshow('captured image', im)
    # cv2.imwrite(cam_cal_photo, im)

    # # Open the captured image, let user crop image and find color values
    # screen, px = setup(cam_cal_photo)
    # # Display the taken image, let user crop and get coordinates of cropped image
    # hm, hp, sm, sp, vm, vp = mainLoop(screen, px)
    #
    # # Close the calibration image
    # pygame.display.quit()


    # Open camera to live stream and identify cutter
    while True:
        (grabbed, frame) = camera.read()
        frame = cv2.resize(frame, (photo_size_x,photo_size_y))
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = np.float32(img2)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        frame[dst > 0.01 * dst.max()] = [0, 0, 255]



        # img1 = cv2.imread('text.png', cv2.COLOR_BGR2GRAY)  # queryImage
        #img1 = cv2.imread(cam_cal_photo, cv2.COLOR_BGR2GRAY)  # trainImage
        # Initiate SIFT detector
        #orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT
        #kp1, des1 = orb.detectAndCompute(img1, None)
        #kp2, des2 = orb.detectAndCompute(img2, None)
        # create BFMatcher object
        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        #matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        #matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

        # plt.imshow(img3), plt.show()













        #sift = cv2.xfeatures2d.SIFT_create()
        #detector = cv2.FeatureDetector_create("SIFT")
        # detector = cv2.xfeatures2d.SURF_create()
        # detector = cv2.SIFT()
        #orb = cv2.ORB_create()

        #kps = detector.detect(gray)
        #kp = sift.detect(gray, None)
        #gray = cv2.drawKeypoints(gray, kp, gray)

        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lower = np.array([hm, sm, vm])
        # upper = np.array([hp, sp, vp])
        # mask = cv2.inRange(hsv, lower, upper) # 0 or 1, is the pixel in that color range or not
        # mask_result = cv2.bitwise_and(frame, frame, mask=mask) # suppress the pixels that are not in the color range
        #
        # # Morphological transformation
        # erode = cv2.getTrackbarPos('erode', 'Trackbar')
        # dilate = cv2.getTrackbarPos('dilate', 'Trackbar')
        # kernel = np.ones((2, 2), np.uint8)
        # mask1 = cv2.erode(mask, kernel,iterations=erode)     # erodes away the boundaries of foreground object (removes white noise but also shrinks object)
        # mask2 = cv2.dilate(mask1, kernel, iterations=dilate) # dilate the object back without noise
        # #mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #
        # rm = cv2.getTrackbarPos('r-', 'Trackbar')
        # rp = cv2.getTrackbarPos('r+', 'Trackbar')
        # _, contours, hierarchy = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # make a mask copy, because function modifies the input
        # frame_circles = frame.copy()
        # # Find minimum enclosing circle for all contours
        # for mycontour in contours:
        #     ((x, y), radius) = cv2.minEnclosingCircle(mycontour)
        #     if radius > rm and radius < rp:
        #         cv2.circle(frame_circles, (int(x), int(y)), int(radius), (0, 255, 0), thickness=5)
        #         cv2.circle(frame_circles, (int(x), int(y)), 2, (0, 255, 0), thickness=5)
        #
        #
        # distance_min = cv2.getTrackbarPos('dmin', 'Trackbar')
        # circles = cv2.HoughCircles(mask2, cv2.HOUGH_GRADIENT, 2, 500)
        # #circles = cv2.HoughCircles(mask2, cv2.HOUGH_GRADIENT, 2, distance_min, minRadius=rm, maxRadius=rp)
        # output = frame.copy()
        # # ensure at least some circles were found
        # if circles is not None:
        #     circles = np.round(circles[0, :]).astype("int")
        #     # loop over the (x, y) coordinates and radius of the circles
        #     for (x, y, r) in circles:
        #         cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        ###############################################
        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('camera', photo_size_x, photo_size_y)
        cv2.imshow('camera', frame)

        # cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('camera', photo_size_x, photo_size_y)
        # cv2.imshow('camera', gray)

        # cv2.namedWindow('mask_result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('mask_result', photo_size_x, photo_size_y)
        # cv2.imshow('mask_result', mask_result)
        #
        # cv2.namedWindow('mask2', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('mask2', photo_size_x, photo_size_y)
        # cv2.imshow('mask2', mask2)
        #
        # cv2.namedWindow('frame_circles', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame_circles', photo_size_x, photo_size_y)
        # cv2.imshow('frame_circles', frame_circles)
        #
        # cv2.namedWindow('frame_circles2', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame_circles2', photo_size_x, photo_size_y)
        # cv2.imshow('frame_circles2', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()