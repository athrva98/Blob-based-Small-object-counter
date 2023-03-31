# Blob-based-Small-object-counter
This is an old project of mine which uses a blob detector to detect and count the number of pipes loaded on a vehicle. This detector has also worked well with counting number of cells in an image (for micro-biology applications).

# Features

This small object detector is multi-scale and rotation invariant. The user specifies the number of (approx) scales of objects that can be found in the image and the detector uses this information to categorize the detections based on histograms and remove any outliers. Since the histograms are constructed based on the sizes of the detected keypoints, this method of outlier detection and refinement fails for outliers that have roughly  the same size as the true detections. To this I suggest using a neural network or alternatively any feature extraction process to get the features of all the detections. Post which outliers can be detected based on the mutual structural similarity between the detections. As a small project (as this project was intended), I deem the results as satisfactory. Any suggestions are welcome.

# Images and Results

1. Input Image \
![alt text](https://github.com/NonStopEagle137/Blob-based-Small-object-counter/blob/main/Images/input_blob.jpg?raw=true)
2. Detections \
![alt text](https://github.com/NonStopEagle137/Blob-based-Small-object-counter/blob/main/Images/blob_counter.jpg?raw=true)


