import cv2
import numpy as np 
from matplotlib import pyplot as plt
import random
import math
from random import randrange
UBIT = 'venktesh'
np.random.seed(sum([ord(c) for c in UBIT]))

#IMAGE READING AND CONVERSION TO GRAY SCALE FOR PROCESSING
def image_reading():
    img1 = cv2.imread('tsucuba_left.png')
    img2 = cv2.imread('tsucuba_right.png')
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
    #cv2.imshow('SIFT_tscuba_left.png',task1_sift1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow('SIFT_tscuba_right.png',task1_sift2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite('task2_sift1.jpg',task1_sift1)
    cv2.imwrite('task2_sift2.jpg',task1_sift2)
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
        #print('HOMOGRAPHY MATRIX', H)
        #new_required_points = random.sample(required_points,10)
        matchesMask = mask.ravel().tolist() 
        #random_matches = random.sample(matchesMask,10)        
        #draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = random_matches , flags = 2)
        draw_params1 = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = matchesMask, flags = 2 )
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,new_required_points,None,**draw_params)
        img3_1 = cv2.drawMatches(img1,kp1,img2,kp2,required_points,None,**draw_params1)
        #plt.imshow(img3_1,'gray'),plt.show()
        #plt.imshow(img3_1,'gray'),plt.show()
        #cv2.imwrite('task1_matches.png',img3_1) 
        cv2.imwrite('task2_matches_knn.png',img3_1)
    return required_points,kp1,kp2,matches

def fundamental_matrix_calc():
    fund_pt1 = []
    fund_pt2= []
    img1,img2,gray1,gray2 = image_reading()
    required_points,kp1,kp2,matches = draw_keypoints()
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.75*n.distance :
            fund_pt2.append(kp2[m.trainIdx].pt)
            fund_pt1.append(kp1[m.queryIdx].pt)
    
    fund_pt1 = np.int32(fund_pt1)
    fund_pt2 = np.int32(fund_pt2)
    F, mask1 = cv2.findFundamentalMat(fund_pt1,fund_pt2,cv2.FM_RANSAC) #FM_LMEDS
    fund_pt1 = fund_pt1[mask1.ravel()==1]
    fund_pt2 = fund_pt2[mask1.ravel()==1]
    fund_pt1,fund_pt2
    print('Fundamental Matrix \n',F)
    return gray1,gray2,fund_pt1,fund_pt2,F

def drawlines(img1,img2,lines,fund_pt1,fund_pt2):
    random_index = []
    new_local_arr1 = []
    new_local_arr2 = []
    updt = []
    updt1 = []
    count = -1
    #rnd = np.zeros((10,3),dtype=float)
    for i in range(10):
        random_index.append(randrange(len(lines)))
    random_index.sort()
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    #random_lines = np.random.choice(lines,10)
    #random_fundpt1 = np.random.choice(fund_pt1,10)
    #random_fundpt2 = np.random.choice(fund_pt2,10)
    i = 0
    for r,fund_pt1,fund_pt2 in zip(lines,fund_pt1,fund_pt2):
        count = count + 1
        if count in random_index:
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1]])
            x1,y1 = map(int, [c, -(r[2] + r[0] * c)/r[1]])
            cv2.line(img1, (x0,y0), (x1,y1), color,1)
            cv2.circle(img1,tuple(fund_pt1),5,color,-1)
            cv2.circle(img2,tuple(fund_pt2),5,color,-1)
            i = i+1
    return img1,img2


def drawing_epilines():
    gray1,gray2,fund_pt1,fund_pt2,F = fundamental_matrix_calc()
    lines1 = cv2.computeCorrespondEpilines(fund_pt2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(gray1,gray2,lines1,fund_pt1,fund_pt2)

    lines2 = cv2.computeCorrespondEpilines(fund_pt1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawlines(gray2,gray1,lines2,fund_pt1,fund_pt2)

    #plt.subplot(121),plt.imshow(img5)
    #plt.subplot(122),plt.imshow(img3)
    #plt.show()
    cv2.imwrite('task2_epi_left.jpg',img5)
    cv2.imwrite('task2_epi_right.jpg',img3)

def disparity():
    img1,img2,gray1,gray2 = image_reading()
    #Computing disparity map using stereo block matchig algo
    create_stereo = cv2.StereoSGBM_create(0,64, preFilterCap = 10,blockSize = 20, speckleWindowSize = 91, speckleRange = 20)
    disparity = create_stereo.compute(gray1,gray2)
    #plt.imshow(disparity,'gray')
    #plt.show()
    #plt.savefig('disparity.png',cmap = plt.cm.gray)
    plt.imsave('disparity.png',disparity,cmap = plt.cm.gray)
    #cv2.imwrite('Disparity.png',disparity)

def main_func():
    drawing_epilines()
    disparity()

main_func()

#REFERENCES : https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html
# GIT - HUB REPOSITERIES