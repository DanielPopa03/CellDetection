import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

for i in range(10,100):
   
    number = "{:04d}".format(i)
   
    picture = np.array(cv2.imread(f"ds1/train/img/cls/{number}.tif",cv2.IMREAD_COLOR))
    
    
    vechea = picture
    shape = vechea.shape

    picture = picture.reshape((picture.shape[0]*picture.shape[1], picture.shape[2]))
    dbscan = DBSCAN(eps=0.4, min_samples=10)
    labels = dbscan.fit_predict(picture)

    mask = np.array(picture)
    mask[labels != -1] = np.array([0,0,0])
    mask.shape = vechea.shape
    mask = mask.astype(np.uint8)
    
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)) 
    plt.show()
    
    cells = np.argwhere(np.any(mask != [0, 0, 0], axis=-1))
    normalized_pixels = cells / np.array(vechea.shape[0:2])
    print(normalized_pixels)
    dbscan = DBSCAN(eps=0.005 ,min_samples=25)
    labels_cells = dbscan.fit_predict(normalized_pixels)
    cells = cells[labels_cells != -1]
    labels_cells = labels_cells[labels_cells != -1]
    print(labels_cells)
    plt.scatter(cells[:, 0], cells[:, 1], c=labels_cells, cmap='viridis', edgecolors='k')
    plt.title("DBSCAN Clustering")
    plt.show()


