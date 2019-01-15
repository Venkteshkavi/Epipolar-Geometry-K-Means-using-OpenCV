import cv2
import numpy as np 
from matplotlib import pyplot as plt
import random
UBIT = 'venktesh'
np.random.seed(sum([ord(c) for c in UBIT]))

#IMAGE READING AND CONVERSION TO GRAY SCALE FOR PROCESSING
def image_reading():
    img1 = cv2.imread('mountain1.jpg')
    img2 = cv2.imread('mountain2.jpg')
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return img1,img2,gray1,gray2
#SIFT DETECTION
def find_keypoints():
    img1,img2,gray1,gray2 = image_reading()
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    #DRAWING KEYPOINTS
    task1_sift1 = cv2.drawKeypoints(gray1,kp1,img1)
    task1_sift2 = cv2.drawKeypoints(gray2,kp2,img2)
    #cv2.imshow('SIFT_Mountain1.jpg',task1_sift1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow('SIFT_Mountain2.jpg',task1_sift2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite('task1_sift1.jpg',task1_sift1)
    cv2.imwrite('task1_sift2.jpg',task1_sift2)
    return des1,des2,kp1,kp2,gray1,gray2
#FLANN
def draw_keypoints():
    new_matches = []
    img1,img2,gray1,gray2 = image_reading()
    des1,des2,kp1,kp2,gray1,gray2 = find_keypoints()
    flann_algorithm = 1
    index_params = dict(algorithm = flann_algorithm, trees =5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    #Showing all matched points

    required_points = []
    matchcount = 10
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            required_points.append(m)
    if(len(required_points) > matchcount):
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in required_points]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in required_points]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        print('HOMOGRAPHY MATRIX', H)
        new_required_points = random.sample(required_points,10)
        matchesMask = mask.ravel().tolist() 
        random_matches = random.sample(matchesMask,10)        
        draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = random_matches , flags = 2)
        draw_params1 = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = matchesMask, flags = 2 )
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,new_required_points,None,**draw_params)
        img3_1 = cv2.drawMatches(img1,kp1,img2,kp2,required_points,None,**draw_params1)
        #plt.imshow(img3,'gray'),plt.show()
        #plt.imshow(img3_1,'gray'),plt.show()
        cv2.imwrite('task1_matches.jpg',img3) 
        cv2.imwrite('task1_matches_knn.jpg',img3_1)
    return H
    #plt.imshow(result,'Warped Image'),plt.show()

def warping_images():
    M = draw_keypoints()
    img1,img2,gray1,gray2 = image_reading()
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, M)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    result = cv2.warpPerspective(img1, Ht.dot(M), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2
    cv2.imwrite('Warped Image.jpg',result)
    #cv2.imshow('Warped Images',result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #plt.imshow(result,'Warped Image'),plt.show()

    
def main_func():
    warping_images()

main_func()

#REFERENCE = https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
#GIT HUB REPOSITARIES
#WARPING - https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/13074597
