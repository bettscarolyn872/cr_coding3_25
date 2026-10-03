You may have noticed that we didn't go over lab `b_image_effects.py` in class. 

This file demonstrates **how to create interactive controls (trackbars)** in OpenCV and draw basic shapes on images.

**What it does:**
- Creates a black canvas (512×512 pixels)
- Adds a trackbar (slider) that controls the X position of a green rectangle
- As you drag the slider left/right, the rectangle moves horizontally
- Shows the basics of OpenCV's drawing functions

**OpenCV has built-in functions for drawing:**

- `cv.rectangle()` - Draw rectangles
- `cv.circle()` - Draw circles
- `cv.line()` - Draw lines
- `cv.putText()` - Add text

You'll see trackbars used more meaningfully in `e_blob_image.py` where they control blob detection parameters. This file just shows the basic pattern in a simple context.
