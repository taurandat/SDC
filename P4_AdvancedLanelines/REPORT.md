# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the **camera calibration matrix** and **distortion coefficients** given a set of **chessboard images**.
* Apply a **distortion correction** to **raw images**.
* Use **color transforms**, **gradients**, etc., to create a **thresholded binary image**.
* Apply a **perspective transform** to rectify **binary image** (**"birds-eye view"**).
* Detect **lane pixels** and fit to **find the lane boundary**.
* Determine the **curvature** of the lane and **vehicle position** with respect to center.
* Warp the **detected lane boundaries** back onto the original image.
* Output **visual display of the lane boundaries** and **numerical estimation of lane curvature and vehicle position**.

---

## Camera Calibration and Distortion Correction

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is the method `calculate_calibration` contained in the file [`camera.py`](camera.py), which has been shown below.

```python
    def calculate_calibration(self, images, pattern_size, retain_calibration_images):
        """
        Prepares calibration settings.

        Parameters
        ----------
        images                      : Set of calibration images.
        pattern_size                : Calibration pattern shape.
        retain_calibration_images   : Flag indicating if we need to preserve calibration images.
        """
        # Prepare object points: (0,0,0), (1,0,0), (2,0,0), ...
        pattern = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        pattern[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        pattern_points = []  # 3d points in real world space
        image_points = []  # 2d points in image plane.

        image_size = None

        # Step through the list and search for chessboard corners
        for i, path in enumerate(images):
            image = mpimg.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points and image points
            if found:
                pattern_points.append(pattern)
                image_points.append(corners)
                image_size = (image.shape[1], image.shape[0])
                if retain_calibration_images:
                    cv2.drawChessboardCorners(image, pattern_size, corners, True)
                    self.calibration_images_success.append(image)
            else:
                if retain_calibration_images:
                    self.calibration_images_error.append(image)

        if pattern_points and image_points:
            _, self.camera_matrix, self.dist_coefficients, _, _ = cv2.calibrateCamera(
                pattern_points, image_points, image_size, None, None
            )
```

In order to calibrate the camera, a list of 20 chessboard images was provided (in camera_cal folder). We used the function `cv2.findChessboardCorners()` to detect the corners of chessboards and save the successfully detected ones to the list `image_points`. We also need `pattern_points` which are the coordinates `(x, y, z)` of the chessboard corners in the world. Here we are assuming the chessboard is fixed on the `(x, y)` plane at `z = 0` (the wall), such that the pattern points are the same for each calibration image. Thus, `pattern` is just a replicated array of coordinates, and `pattern_points` will be appended with a copy of it every time we successfully detect all chessboard corners in a test image.

We then used the output `pattern_points` and `image_points` to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. We applied this distortion correction to the test image using the cv2.undistort() function.

```python
corrected_image = cv2.undistort(image, self.camera_matrix, self.dist_coefficients, None, self.camera_matrix)
```

That gives us the results:




###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

