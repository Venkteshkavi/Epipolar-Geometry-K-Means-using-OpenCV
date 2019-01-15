import cv2
from copy import deepcopy
import numpy as np
#import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
UBIT = 'venktesh'
np.random.seed(sum([ord(c) for c in UBIT]))
#from tqdm import tqdm



# Euclidean Distance Caculator
def dist(a, b, ax=1):
     return np.linalg.norm(a - b, axis=ax)

def clusters():
    j = 0
    # Importing the dataset
    data = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
    #data.head()
    t = [0,1,2]
    # Getting the values and plotting it
    f1 = data[:,0]
    f2 = data[:,1]
    #X = np.array(list(zip(f1, f2)))
    X = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
    # Number of clusters    # for i in range(0,):
    #     if(clusters[])
    k = 3
    # X coordinates of random centroids
    #C_x = np.random.randint(0, np.max(X)-20, size=k)
    #C_x = data[:,0]
    #C_y = data[:,1]
    # Y coordinates of random centroids
    #
    #C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    C = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
    print("Initial Centroids")
    C_1 = C[:,0]
    C_2 = C[:,1]
    print(C)
    # Plotting along with the Centroids
    #plt.scatter(f1, f2, c='#050505', s=7)
    #plt.scatter(C_1,C_2)
    #plt.scatter(C_x, C_y, marker='^', s=200, c='g')
    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    #Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(data))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
            #plt.show(clusters[k] for k in range(i))
        # Storing the old centroid values
        C_old = deepcopy(C)
        #cent = []
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            #plt.text(plot_x,plot_y,'points')
            C[i] = np.mean(points, axis=0)   
            new_centroid_x = C[i][0]
            new_centroid_y = C[i][1]
            plot_x,plot_y = zip(*points)
            size_x = len(plot_x)
            size_y = len(plot_y)
            if(i==0):
                plt.scatter(new_centroid_x,new_centroid_y, c= 'r')
                #plt.text(new_centroid_x,new_centroid_y,'   ' + '(' + str(new_centroid_x) + ' , ' + str(new_centroid_y) + ')')
            elif(i == 1):
                plt.scatter(new_centroid_x,new_centroid_y, c= 'g')
                #plt.text(new_centroid_x,new_centroid_y,'   ' + '(' + str(new_centroid_x) + ' , ' + str(new_centroid_y) + ')')
            else:
                plt.scatter(new_centroid_x,new_centroid_y, c= 'b')
                #plt.text(new_centroid_x,new_centroid_y,'   ' + '(' + str(new_centroid_x) + ' , ' + str(new_centroid_y) + ')')
  
            if(i == 0):
                plt.scatter(plot_x,plot_y,marker = '^', c = 'r',s = 300, edgecolors='w')
                for k in range(0,size_x):
                    plt.text(plot_x[k],plot_y[k],'   ' + '(' + str(plot_x[k]) + ' , ' + str(plot_y[k]) + ')')
            elif(i==1):
                plt.scatter(plot_x,plot_y,marker = '^', c = 'g',s = 300, edgecolors='w')
                for k in range(0,size_x):
                    plt.text(plot_x[k],plot_y[k],'   ' + '(' + str(plot_x[k]) + ' , ' + str(plot_y[k]) + ')')
            else:
                plt.scatter(plot_x,plot_y,marker = '^', c = 'b',s = 300, edgecolors='w')
                for k in range(0,size_x):
                    plt.text(plot_x[k],plot_y[k],'   ' + '(' + str(plot_x[k]) + ' , ' + str(plot_y[k]) + ')')
            plt.savefig('scatter_plot' + str(j) + '.png')
            j = j+1
        error = dist(C, C_old, None)
        #plt.show()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    print("final centroids")
    print(C)

def eucl_dist(a,b,ax=1):
    return np.linalg.norm(a-b, axis=ax)

def color_quantization(num_cluster,q):
    color_img = cv2.imread('baboon.jpg')
    #print(color_img.shape)
    h,w = color_img.shape[:2]
    image_array = np.reshape(color_img,(w*h,3))
    #init_centroid = np.zeros((3,3))
    # print(h,w)
    # reshaped_img = color_img.reshape(h,w,3)
    # reshaped_img.shape
    #Clusters 
    data = image_array
    #Number of Clusters used
    #num_cluster = 5

    C_x = np.random.randint(0, np.max(image_array)-20, size=num_cluster)
    C_y = np.random.randint(0,np.max(image_array)-20, size=num_cluster)
    C_z = np.random.randint(0,np.max(image_array)-20, size=num_cluster)
    #C_m = np.random.randint(0,len(imagearray), size=num_cluster)

    init_centroid = np.array(list(zip(C_x, C_y,C_z)), dtype=np.float32)
    print('initial centroid \n',init_centroid)
    prev_centroid = np.zeros((num_cluster,3))
    clusters = np.zeros(len(data))
    #Euclidean Distance
    error = eucl_dist(init_centroid,prev_centroid,None)
    while(error != 0):
        for i in range(len(data)):
            euclidean_distance = eucl_dist(data[i], init_centroid)
            cluster = np.argmin(euclidean_distance)
            clusters[i] = cluster
        prev_centroid = deepcopy(init_centroid)
        for i in range(num_cluster):
            # for j in range(len(data)):
            #     if(cluster[j] == i ):
            #         data[j] = cluster[i]
            points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
            init_centroid[i] = np.mean(points,axis=0)
            #print(init_centroid[i])
            # new_centroid_x = init_centroid[i][0]
            # new_centroid_y = init_centroid[i][1]
            # plot_x,plot_y = zip(*points)
            # size_x = len(plot_x)
            # size_y = len(plot_y)
        error = eucl_dist(init_centroid,prev_centroid,None)
    for i in range(num_cluster):
        points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
    print("final centroid")
    print(init_centroid)
    init_centroid = np.asarray(init_centroid)
    kp = np.reshape(clusters,(512,512))
    final_image = np.zeros((512,512,3))
    for i in range(0,512):
        for j in range(0,512):
            c = kp[i][j]
            k = int(c)
            final_image[i][j] = init_centroid[k]
    #plt.imshow(kp,)
    #plt.show()
    cv2.imwrite('baboon' + str(q) + '.png',final_image)
    #plt.imshow(final_image,)
    #plt.show()
    #print('Clusters \n',clusters)


def main_func():
    clusters()
    print('**** COLOR QUANTIZATION 1 ****')
    color_quantization(3,1)
    print('')
    print('**** COLOR QUANTIZATION 2 ****')
    color_quantization(5,2)
    print('')
    print('**** COLOR QUANTIZATION 3 ****')
    color_quantization(10,3)
    print('')
    print('**** COLOR QUANTIZATION 4 ****')
    color_quantization(20,4)

main_func()

#REFERENCES : OPENCV Documentation
#https://github.com/mubaris/friendly-fortnight