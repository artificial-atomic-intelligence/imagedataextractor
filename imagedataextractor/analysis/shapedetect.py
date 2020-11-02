import cv2
import numpy as np


class ShapeDetector:

    def __init__(self):
        pass
    
    def detect_circle(self, mask):

        circle = False

        contours = self.get_contours(mask)
        ellipse = cv2.fitEllipse(contours)
        w, h = ellipse[1]
        aspect_ratio = w/h

        ellipse_image = np.zeros_like(mask.copy(), dtype=np.uint8)
        ellipse_image = cv2.ellipse(ellipse_image, ellipse, color=(255,255,255), thickness=-1)
        ellipse_image = (ellipse_image > 0).astype(np.uint8)

        intersection = np.sum(np.logical_and(mask, ellipse_image))
        union = np.sum(np.logical_or(mask, ellipse_image))
        iou = intersection / union

        if aspect_ratio > 0.85 and iou > 0.95:
            circle = True
        
        return circle

    def get_contours(self, x):
        contours = cv2.findContours(x.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 2:
            contours = contours[0][0]
        elif len(contours) == 3:
            contours = contours[1][0]
        return contours

    # def logscale_moments(self, moments):
    #     return -1*np.copysign(1.0, moments)*np.log10(np.abs(moments))

    # def detect(self, mask):
    #     shapes = list(self.shape_dict.keys())
    #     shape_moments = list()

    #     distances = []

    #     for shape in shapes:
    #         shape_image = self.shape_dict[shape] / 255
    #         d = cv2.matchShapes(mask, shape_image, cv2.CONTOURS_MATCH_I1, 0)
    #         distances.append(d)

    # def create_shape_images(self):
    #     import matplotlib.pyplot as plt
    #     # square
    #     square_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.rectangle(square_image, (self.w//4, self.h//4), (self.w-(self.w//4), self.h-(self.h//4)), (255, 255, 255), thickness=-1)
    #     self.shape_dict['square'] = square_image

    #     # rectangle
    #     rect_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.rectangle(rect_image, (self.w//3, self.h//4), (self.w-(self.w//3), self.h-(self.h//4)), (255, 255, 255), thickness=-1)
    #     self.shape_dict['rectangle'] = rect_image

    #     # rod
    #     rod_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.ellipse(rod_image, (self.w//2, self.h//2), (self.w//3, self.h//10), 0, 0, 360, color=(255,255,255), thickness=-1, )
    #     self.shape_dict['rod'] = rod_image

    #     # triangle
    #     tri_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     points = np.array([[self.w//2, self.h//6],
    #                        [self.w//6, self.h-(self.h//6)], 
    #                        [self.w-(self.w//6), self.h-(self.h//6)]]).astype(int)
    #     cv2.drawContours(tri_image, [points], -1, (255,255,255), thickness=-1)
    #     self.shape_dict['triangle'] = tri_image

    #     # circle
    #     circle_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.circle(circle_image, (self.w//2, self.h//2), self.h//4, color=(255,255,255), thickness=-1, )
    #     self.shape_dict['circle'] = circle_image
    #     # pentagon
    #     pent_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     pent_points = np.array([[self.w/2, self.h/4.5], 
    #                        [self.w/3, self.h/3 + self.h/15], 
    #                        [(self.w/3)+(self.w/12), 2*self.h/3], 
    #                        [self.w-(self.w/3)-(self.w/12), 2*self.h/3], 
    #                        [self.w-(self.w/3), self.h/3 + self.h/15]]).astype(int)

    #     cv2.drawContours(pent_image, [pent_points], -1, (255,255,255), thickness=-1)
    #     self.shape_dict['pentagon'] = pent_image

    #     # hexagon
    #     hex_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     hex_points = np.array([[self.w/3, self.h/4], 
    #                            [self.w/5, self.h/2], 
    #                            [self.w/3, self.h-(self.h/4)], 
    #                            [self.w-(self.w/3), self.h-(self.h/4)], 
    #                            [self.w-(self.w/5), self.h/2], 
    #                            [self.w-(self.w/3), self.h/4], ]).astype(int)
    #     cv2.drawContours(hex_image, [hex_points], -1, (255,255,255), thickness=-1)
    #     self.shape_dict['hexagon'] = hex_image
