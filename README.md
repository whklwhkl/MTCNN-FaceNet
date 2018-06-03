# Real-time Faces Detection and Recognition System: Integrating MTCNN and FaceNet

This repository contains:

1. `detection.py` : the main code of **MTCNN** used for face dectection. We take advantage of open source pre-train model published [here](https://github.com/TropComplique/mtcnn-pytorch)

2. `recognition.py` : the main code of **FaceNet**. We use pre-trained model published [here](https://github.com/davidsandberg/facenet)

3. Folder `TrainImage` : pre-processed images of class

4. Folder `src` : the scorce code of MTCNN

## Run 
### Required packages:

* python3
* torch
* tensorflow
* OpenCV(cv2)
* numpy
* scipy
* pillow

### Dowload FaceNet pre-train weight
To correct run this code, you need to dowload faceNet pre-train weight from [here](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) and unzip it in the current folder

### Command(Terminal)

```bash
$ python FaceId.py -i YOUR_TEST_IMAGE(e.g. IMG_1819.jpg)
```

### Import as Package:

```python
from FaceId import main
main("path/to/test_image", "path/to/result_image")
```

**Note**: we recommend this way of running our code, if you want to test the time needed to finish the task multiple times.

Of course, our main function can take additional arguments, like the KNN parameter`K` and the output text configuration parameter `columns` which is the number of columns when displaying the recognition result as a table. In addition we have `width` and `height` to control the size of the actual size of image involved in the process, given an image, especially a large one.