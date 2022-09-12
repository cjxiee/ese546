from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sklearn as skl

import matplotlib.pyplot as plt
import cv2
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# load the dataset
ds = fetch_openml('mnist_784', as_frame=False)
# create the 10000 samples dataset and split it into 8:2
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)
# print(len(x[0]))

# down-sampling the image
a = x[0].reshape((28, 28))
print(a)
plt.imshow(a)
plt.savefig('a')
b = cv2.resize(a, (14,14))
plt.imshow(b)
plt.savefig('b')

# create the classifier
clf = skl.svm.SVC(C=1.0, kernel='rbf', gamma='auto')
clf.fit(x,y)








# cv2.waitKey(0)