# The code was originally written by MariyaSha and later developed further by Phoenix Marie.

import cv2
import numpy as np

# Connecting to the phone camera in real-time
url = 'https://192.168.1.68:8080/video'
cap = cv2.VideoCapture(url)

# Defining the video type and output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (720, 480))

def apply_filters(frame):
    """Applies multiple filters to enhance video quality."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    combined = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
    cv2.rectangle(combined, (50, 50), (670, 430), (255, 255, 255), 2)
    return combined

def detect_faces(frame):
    """Detects faces in the video feed."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return frame

def apply_sepia(frame):
    """Applies a sepia filter to the frame."""
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, sepia_filter)
    return cv2.convertScaleAbs(sepia)

def apply_pencil_sketch(frame):
    """Applies a pencil sketch effect."""
    gray, sketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

def apply_cartoon_effect(frame):
    """Applies a cartoon effect."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_negative(frame):
    """Applies a negative filter effect."""
    return cv2.bitwise_not(frame)

def apply_emboss(frame):
    """Applies an emboss effect."""
    kernel = np.array([[ -2, -1,  0],
                       [ -1,  1,  1],
                       [  0,  1,  2]])
    embossed = cv2.filter2D(frame, -1, kernel)
    return embossed

def apply_sharpen(frame):
    """Applies a sharpening effect."""
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    sharpened = cv2.filter2D(frame, -1, kernel)
    return sharpened

def apply_edge_enhancement(frame):
    """Enhances edges using Laplacian filter."""
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def apply_brightness_contrast(frame, brightness=30, contrast=50):
    """Adjusts brightness and contrast."""
    frame = np.int16(frame)
    frame = frame * (contrast / 127 + 1) - contrast + brightness
    frame = np.clip(frame, 0, 255)
    return np.uint8(frame)

def apply_blur(frame, ksize=(15, 15)):
    """Applies a blur effect."""
    return cv2.GaussianBlur(frame, ksize, 0)

def apply_sobel(frame):
    """Applies a Sobel filter for edge detection."""
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.convertScaleAbs(sobelx + sobely)

def apply_motion_blur(frame):
    """Applies a motion blur effect."""
    kernel = np.zeros((15, 15))
    kernel[int((15 - 1) / 2), :] = np.ones(15)
    kernel /= 15
    return cv2.filter2D(frame, -1, kernel)

def apply_invert(frame):
    """Inverts colors in the frame."""
    return cv2.bitwise_not(frame)

def apply_vignette(frame):
    """Applies a vignette effect to the frame."""
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / 5)
    kernel_y = cv2.getGaussianKernel(rows, rows / 5)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignette = np.copy(frame)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette.astype(np.uint8)

def apply_glitch(frame):
    """Applies a glitch effect to the frame."""
    glitch_frame = np.copy(frame)
    rows, cols = glitch_frame.shape[:2]
    num_glitches = 5
    for _ in range(num_glitches):
        start_row = np.random.randint(0, rows // 2)
        end_row = np.random.randint(start_row + 10, rows)
        glitch_frame[start_row:end_row, :] = np.roll(glitch_frame[start_row:end_row, :], np.random.randint(-30, 30), axis=1)
    return glitch_frame

# Real-time video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    
    # Apply multiple filters
    filtered_frame = apply_filters(frame)
    face_detected_frame = detect_faces(filtered_frame)
    sepia_frame = apply_sepia(face_detected_frame)
    sketch_frame = apply_pencil_sketch(sepia_frame)
    cartoon_frame = apply_cartoon_effect(sketch_frame)
    negative_frame = apply_negative(cartoon_frame)
    embossed_frame = apply_emboss(negative_frame)
    sharpened_frame = apply_sharpen(embossed_frame)
    edge_enhanced_frame = apply_edge_enhancement(sharpened_frame)
    bright_contrast_frame = apply_brightness_contrast(edge_enhanced_frame)
    blurred_frame = apply_blur(bright_contrast_frame)
    sobel_frame = apply_sobel(blurred_frame)
    motion_blur_frame = apply_motion_blur(sobel_frame)
    inverted_frame = apply_invert(motion_blur_frame)
    vignette_frame = apply_vignette(inverted_frame)
    glitch_frame = apply_glitch(vignette_frame)
    
    # Display and save the processed video
    cv2.imshow("Enhanced Video", glitch_frame)
    out.write(glitch_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
