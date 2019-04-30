from sklearn import neighbors
from sklearn import datasets
import cv2
import numpy as np

co = np.array([[1, 1], [2, 1], [3, 1], [1, 2], [2, 2], [3, 2], [1, 3], [2, 3], [3, 3]])
label = np.array(range(9))
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
knn.fit(co,label)
predictedLabel = knn.predict([[0, 0], [4, 0], [0, 4], [4, 4]])
print (predictedLabel,iris['target_names'][predictedLabel])