
import cv2
import numpy as np
import matplotlib.image as mpimg

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init LaneLines class."""
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        
        # Load direction images
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')
        
        # Normalize images
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # HYPERPARAMETERS
        self.nwindows = 9        
        self.margin = 100        
        self.minpix = 50         

    def forward(self, img):
        """Take an image and detect lane lines."""
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """Return all pixel that in a specific window."""
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)
        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        """Extract features from a binary image."""
        self.img = img
        self.window_height = int(img.shape[0] // self.nwindows)  # Changed from np.int to int
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image."""
        assert len(img.shape) == 2
        out_img = np.dstack((img, img, img)) * 255

        histogram = hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height // 2

        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))  # Already using np.int32, no change needed
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))  # Already using np.int32, no change needed

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it."""
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        return out_img

    def plot(self, out_img):
        lR, rR, pos = self.measure_curvature()

        value = self.left_fit[0] if abs(self.left_fit[0]) > abs(self.right_fit[0]) else self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')

        if len(self.dir) > 10:
            self.dir.pop(0)

        direction = max(set(self.dir), key=self.dir.count)

        # Display widget for direction
        W, H = 400, 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:], widget[-1,:], widget[:,0], widget[:,-1] = [0, 0, 255], [0, 0, 255], [0, 0, 255], [0, 0, 255]
        out_img[:H, :W] = widget

        msg = "Keep Straight Ahead" if direction == 'F' else "Left Curve Ahead" if direction == 'L' else "Right Curve Ahead"
        curvature_msg = f"Curvature = {min(lR, rR):.0f} m"
        
        if direction == 'L':
            y, x = self.left_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
        elif direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
        else:
            y, x = self.keep_straight_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        
        cv2.putText(out_img, "Good Lane Keeping", org=(10, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=(0, 255, 0), thickness=2)
        cv2.putText(out_img, f"Vehicle is {pos:.2f} m away from center", org=(10, 450), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.66, color=(255, 255, 255), thickness=2)

        return out_img

    def measure_curvature(self):
        """
        Calculate the radius of curvature of the lane and the vehicle's position relative to the lane center.
        """
        ym_per_pix = 30 / 720  
        xm_per_pix = 3.7 / 700  

        ploty = np.linspace(0, self.img.shape[0] - 1, self.img.shape[0])
        

        left_fit_cr = np.polyfit(ploty * ym_per_pix, self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2], 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2], 2)

        y_eval = np.max(ploty)  # Evaluate at the bottom of the image
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Calculate the vehicle position relative to the center
        lane_center = (self.left_fit[0] * y_eval**2 + self.left_fit[1] * y_eval + self.left_fit[2] +self.right_fit[0] * y_eval**2 + self.right_fit[1] * y_eval + self.right_fit[2]) / 2
        car_position = self.img.shape[1] // 2  # The center of the image is the vehicle position
        center_offset_meters = (car_position - lane_center) * xm_per_pix

        return left_curverad, right_curverad, center_offset_meters
