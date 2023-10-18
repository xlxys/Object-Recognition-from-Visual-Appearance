import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA


# Read face image from zip file on the fly
faces = {}

# Open a ZIP file: class zipfile.ZipFile(...)
with zipfile.ZipFile("archive.zip") as facezip:
    for filename in facezip.namelist():  
    #ZipFile.namelist() Return a list of archive members by name.
        if not filename.endswith(".pgm"):
            continue # not a face picture
        with facezip.open(filename) as image:
            # If we extracted files from zip, we can use cv2.imread(filename) instead
            faces[filename] = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            #faces[filename] is an array of image values
# Show sample faces using matplotlib
fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
faceimages = list(faces.values())[-16:] # take last 16 images
for i in range(16):
    axes[i%4][i//4].imshow(faceimages[i], cmap="gray")
print("Showing sample faces")
plt.show()

# Print some details
faceshape = list(faces.values())[0].shape
print("Face image shape:", faceshape)

classes = set(filename.split("/")[0] for filename in faces.keys())
print("Number of classes:", len(classes))
print("Number of images:", len(faces))

# Take classes 1-39 for eigenfaces, keep entire class 40 and
# image 10 of class 39 as out-of-sample test
facematrix = []
facelabel = []
for key,val in faces.items():
    if key.startswith("s40/"):
    #     continue # this is our test set
    # if key == "s39/10.pgm":
    #     continue # this is our test set
        facematrix.append(val.flatten())
        facelabel.append(key.split("/")[0])

# Create a NxM matrix with N (=389=400-10-1) images and M pixels per image
facematrix = np.array(facematrix)
print(facematrix.shape)



# Apply PCA and take first K principal components as eigenfaces
#pca = PCA().fit(facematrix)
#print(pca)

n_components = 10
pca = PCA(n_components)
pca.fit(facematrix)
print(pca.explained_variance_ratio_)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.title('Explained Variance Ratio')
plt.show()

eigenfaces = pca.components_[:n_components]

# # Show the first 16 eigenfaces
# fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
# for i in range(10):
#     axes[i%4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
# print("Showing the eigenfaces")
# plt.show()


sum = 0

for i in range(10):
    sum += eigenfaces[i].reshape(faceshape)

plt.plot(sum)
plt.show()

# # Show the first 16 eigenfaces
# fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
# for i in range(16):
#     axes[i%4][i//4].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
# print("Showing the eigenfaces")
# plt.show()

# Generate weights as a KxN matrix where K is the number of eigenfaces 
# and N the number of samples
weights = eigenfaces @ (facematrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)


# Test on out-of-sample image of existing class
query = faces["s40/10.pgm"].reshape(1,-1)

print (query)

# valp, vecp = np.linalg.eig(query)

query_weight = eigenfaces @ (query - pca.mean_).T

rec = query_weight * eigenfaces

plt.plot(rec)
plt.show()





# # Test on out-of-sample image of existing class
# query = faces["s39/10.pgm"].reshape(1,-1)
# query_weight = eigenfaces @ (query - pca.mean_).T
# euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)

# best_match = np.argmin(euclidean_distance)
# print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
# # Visualize
# fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
# axes[0].imshow(query.reshape(faceshape), cmap="gray")
# axes[0].set_title("Query")
# axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
# axes[1].set_title("Best match")
# plt.show()

# # Test on out-of-sample image of new class
# query = faces["s40/8.pgm"].reshape(1,-1)
# query_weight = eigenfaces @ (query - pca.mean_).T
# euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
# best_match = np.argmin(euclidean_distance)
# print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
# # Visualize
# fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
# axes[0].imshow(query.reshape(faceshape), cmap="gray")
# axes[0].set_title("Query")
# axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
# axes[1].set_title("Best match")
# plt.show()




