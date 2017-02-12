import cv2
import numpy as np


image = cv2.imread('test.jpg')
mask = np.zeros(image.shape, dtype=np.uint8)
roi_corners = np.array([[(318,449), (318,246), (546,246),(546,449)]], dtype=np.int32)
print roi_corners.shape
channel_count = image.shape[2]
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask, roi_corners, ignore_mask_color)


# apply the mask
masked_image = cv2.bitwise_and(image, mask)
cv2.imshow("roi corners",masked_image)
cv2.waitKey(0)
# save the result
cv2.imwrite('image_masked.png', masked_image)