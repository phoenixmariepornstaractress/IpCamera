# The code was originally written by MariyaSha and later developed further by Phoenix Marie.

import cv2
import numpy as np

# Connecting to the phone camera in real-time
# IMPORTANT: Replace this URL with the actual IP address and port provided by your IP Webcam app.
# Example: 'http://192.168.1.68:8080/video'
url = 'https://192.168.1.68:8080/video'
cap = cv2.VideoCapture(url)

# Defining the video type and output for saving the processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (720, 480))

def apply_filters(frame):
    """
    Applies multiple basic filters to enhance video quality.
    Includes grayscale, blur, edge detection, and a white rectangle overlay.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Combines the grayscale and edge-detected images
    combined = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
    # Draws a white rectangle on the combined frame
    cv2.rectangle(combined, (50, 50), (670, 430), (255, 255, 255), 2)
    return combined

def detect_faces(frame):
    """
    Detects faces in the video feed using a pre-trained Haar Cascade classifier.
    Draws green rectangles around detected faces.
    """
    # Load the pre-trained face cascade classifier
    # This file needs to be present in your OpenCV installation's haarcascades directory
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return frame

def apply_sepia(frame):
    """
    Applies a sepia filter to the frame.
    This is done by applying a color transformation matrix.
    """
    # Sepia color transformation matrix
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    # Apply the transformation
    sepia = cv2.transform(frame, sepia_filter)
    # Ensure pixel values are within 0-255 range
    return cv2.convertScaleAbs(sepia)

def apply_pencil_sketch(frame):
    """
    Applies a pencil sketch effect to the frame using OpenCV's built-in function.
    """
    # cv2.pencilSketch returns a grayscale sketch and a color sketch. We use the color sketch.
    gray_sketch, color_sketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return color_sketch

def apply_cartoon_effect(frame):
    """
    Applies a cartoon effect to the frame.
    This involves edge detection and bilateral filtering for color smoothing.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    # Adaptive thresholding to create strong edges
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Bilateral filter to smooth colors while preserving edges
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    # Combine color image with edges as a mask
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_negative(frame):
    """
    Applies a negative (color inversion) filter effect to the frame.
    """
    return cv2.bitwise_not(frame)

def apply_emboss(frame):
    """
    Applies an emboss effect using a convolution kernel.
    """
    # Emboss kernel
    kernel = np.array([[ -2, -1,  0],
                       [ -1,  1,  1],
                       [  0,  1,  2]])
    # Apply the kernel using 2D convolution
    embossed = cv2.filter2D(frame, -1, kernel)
    return embossed

def apply_sharpen(frame):
    """
    Applies a sharpening effect using a convolution kernel.
    """
    # Sharpening kernel
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    # Apply the kernel using 2D convolution
    sharpened = cv2.filter2D(frame, -1, kernel)
    return sharpened

def apply_edge_enhancement(frame):
    """
    Enhances edges using the Laplacian filter.
    """
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    # Convert back to 8-bit unsigned integer
    return cv2.convertScaleAbs(laplacian)

def apply_brightness_contrast(frame, brightness=30, contrast=50):
    """
    Adjusts brightness and contrast of the frame.
    Brightness is added to pixel values, contrast scales them.
    """
    frame = np.int16(frame) # Convert to signed 16-bit to avoid overflow during calculations
    frame = frame * (contrast / 127 + 1) - contrast + brightness
    frame = np.clip(frame, 0, 255) # Clip values to stay within 0-255 range
    return np.uint8(frame) # Convert back to unsigned 8-bit

def apply_blur(frame, ksize=(15, 15)):
    """
    Applies a Gaussian blur effect to the frame.
    `ksize` is the kernel size (width, height) of the blur filter.
    """
    return cv2.GaussianBlur(frame, ksize, 0)

def apply_sobel(frame):
    """
    Applies a Sobel filter for edge detection in both X and Y directions,
    then combines them.
    """
    # Convert to grayscale for Sobel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate gradients along X and Y axis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # Combine the absolute values of the gradients
    return cv2.convertScaleAbs(sobelx + sobely)

def apply_motion_blur(frame):
    """
    Applies a motion blur effect using a linear kernel.
    """
    # Create a motion blur kernel (a horizontal line)
    kernel = np.zeros((15, 15))
    kernel[int((15 - 1) / 2), :] = np.ones(15)
    kernel /= 15 # Normalize the kernel
    # Apply the kernel using 2D convolution
    return cv2.filter2D(frame, -1, kernel)

def apply_invert(frame):
    """
    Inverts colors in the frame. This is identical to `apply_negative`.
    """
    return cv2.bitwise_not(frame)

def apply_vignette(frame):
    """
    Applies a vignette effect to the frame, darkening the corners.
    This is achieved by multiplying the frame with a Gaussian mask.
    """
    rows, cols = frame.shape[:2]
    # Create Gaussian kernels for X and Y dimensions
    kernel_x = cv2.getGaussianKernel(cols, cols / 5)
    kernel_y = cv2.getGaussianKernel(rows, rows / 5)
    # Create a 2D Gaussian mask
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max() # Normalize the mask to 0-1
    vignette = np.copy(frame)
    # Apply the mask to each color channel
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette.astype(np.uint8)

def apply_glitch(frame):
    """
    Applies a simple glitch effect by randomly shifting horizontal segments of the frame.
    """
    glitch_frame = np.copy(frame)
    rows, cols = glitch_frame.shape[:2]
    num_glitches = 5 # Number of glitch segments to apply
    for _ in range(num_glitches):
        # Select random start and end rows for the glitch
        start_row = np.random.randint(0, rows // 2)
        end_row = np.random.randint(start_row + 10, rows)
        # Shift the selected rows horizontally by a random amount
        glitch_frame[start_row:end_row, :] = np.roll(glitch_frame[start_row:end_row, :], np.random.randint(-30, 30), axis=1)
    return glitch_frame

def apply_grayscale(frame):
    """
    Converts the frame to grayscale.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def apply_flip_horizontal(frame):
    """
    Flips the frame horizontally.
    """
    return cv2.flip(frame, 1) # 1 means flip horizontally

def apply_gaussian_noise(frame, mean=0, sigma=25):
    """
    Adds Gaussian noise to the frame.
    `mean` is the mean of the Gaussian distribution, `sigma` is its standard deviation.
    """
    row, col, ch = frame.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_frame = frame + gauss
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)

def apply_pixelation(frame, pixel_size=10):
    """
    Applies a pixelation effect to the frame.
    `pixel_size` determines the size of the pixel blocks.
    """
    height, width = frame.shape[:2]
    # Resize to a smaller image
    temp = cv2.resize(frame, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    # Resize back to original size, causing pixelation
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_mirror_effect(frame):
    """
    Applies a mirror effect by combining the original frame with its horizontally flipped version.
    """
    height, width = frame.shape[:2]
    # Split the frame into left and right halves
    left_half = frame[:, :width // 2]
    # Flip the left half to create the right half for the mirror effect
    mirrored_right_half = cv2.flip(left_half, 1)
    # Concatenate the left half and the mirrored right half
    return np.concatenate((left_half, mirrored_right_half), axis=1)

# --- NEWLY ADDED FUNCTIONS ---

def apply_color_tint(frame, color=(0, 0, 255), alpha=0.3):
    """
    Applies a color tint overlay to the frame.
    `color` is a BGR tuple (e.g., (0,0,255) for red).
    `alpha` is the transparency of the tint (0.0 to 1.0).
    """
    overlay = np.full(frame.shape, color, dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

def apply_rotation(frame, angle=45):
    """
    Rotates the frame by a specified angle around its center.
    Positive angle means counter-clockwise rotation.
    """
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the affine transformation
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
    return rotated_frame

def apply_threshold(frame, thresh_val=127):
    """
    Converts the frame to a binary (black and white) image based on a threshold value.
    `thresh_val` is the threshold; pixels above it become white, below become black.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresholded_frame = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR) # Convert back to BGR for consistency

def apply_posterization(frame, k=4):
    """
    Applies a posterization effect by reducing the number of distinct colors in the frame.
    `k` is the number of colors per channel (e.g., k=4 means 4 levels for R, G, B).
    """
    # Divide pixel values by (256 / k), round, and multiply back to quantize colors
    posterized_frame = (frame // (256 // k)) * (256 // k)
    return posterized_frame.astype(np.uint8)

def apply_channel_isolation(frame, channel='red'):
    """
    Isolates and displays only one color channel (Red, Green, or Blue).
    The other channels will be set to zero.
    `channel` can be 'red', 'green', or 'blue'.
    """
    isolated_frame = np.zeros_like(frame)
    if channel.lower() == 'blue':
        isolated_frame[:, :, 0] = frame[:, :, 0] # Blue channel
    elif channel.lower() == 'green':
        isolated_frame[:, :, 1] = frame[:, :, 1] # Green channel
    elif channel.lower() == 'red':
        isolated_frame[:, :, 2] = frame[:, :, 2] # Red channel
    else:
        print("Invalid channel specified. Using original frame.")
        return frame
    return isolated_frame


# Real-time video processing loop
while cap.isOpened():
    ret, frame = cap.read() # Read a frame from the camera
    if not ret or frame is None:
        # Break the loop if frame cannot be read (e.g., camera disconnected)
        print("Failed to grab frame or end of stream.")
        break
    
    # Resize frame to a consistent size for processing and display
    # This also helps standardize output for VideoWriter
    frame = cv2.resize(frame, (720, 480))

    # Start with the original frame for cumulative effects
    processed_frame = frame 
    
    # Apply all defined filters sequentially
    # IMPORTANT NOTE: Applying all filters sequentially can lead to
    # very complex and sometimes undesirable visual results, as effects
    # can interact in unexpected ways or obscure each other.
    # For practical use, you might want to comment out most of these
    # and only enable a few at a time to see their individual effects
    # or specific combinations.

    # Existing filters from previous versions
    processed_frame = apply_filters(processed_frame)
    processed_frame = detect_faces(processed_frame) 
    processed_frame = apply_sepia(processed_frame)
    processed_frame = apply_pencil_sketch(processed_frame)
    processed_frame = apply_cartoon_effect(processed_frame)
    processed_frame = apply_negative(processed_frame)
    processed_frame = apply_emboss(processed_frame)
    processed_frame = apply_sharpen(processed_frame)
    processed_frame = apply_edge_enhancement(processed_frame)
    processed_frame = apply_brightness_contrast(processed_frame)
    processed_frame = apply_blur(processed_frame)
    processed_frame = apply_sobel(processed_frame)
    processed_frame = apply_motion_blur(processed_frame)
    processed_frame = apply_invert(processed_frame) 
    processed_frame = apply_vignette(processed_frame)
    processed_frame = apply_glitch(processed_frame)

    # Filters added in the previous iteration
    processed_frame = apply_grayscale(processed_frame) 
    processed_frame = apply_flip_horizontal(processed_frame)
    processed_frame = apply_gaussian_noise(processed_frame)
    processed_frame = apply_pixelation(processed_frame)
    processed_frame = apply_mirror_effect(processed_frame)
    
    # NEWLY ADDED FILTERS IN THIS ITERATION
    # Experiment with commenting/uncommenting these and changing their order
    processed_frame = apply_color_tint(processed_frame, color=(0, 255, 0), alpha=0.4) # Green tint
    processed_frame = apply_rotation(processed_frame, angle=30) # Rotate by 30 degrees
    processed_frame = apply_threshold(processed_frame, thresh_val=100) # Binary image
    processed_frame = apply_posterization(processed_frame, k=8) # Reduce colors to 8 levels per channel
    processed_frame = apply_channel_isolation(processed_frame, channel='blue') # Show only blue channel

    # Display the processed video in a window
    cv2.imshow("Enhanced Video", processed_frame)
    
    # Write the processed frame to the output video file
    out.write(processed_frame)
    
    # Wait for 1 millisecond and check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and video writer objects
cap.release()
out.release()
# Close all OpenCV display windows
cv2.destroyAllWindows()

 
