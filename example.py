from sgbm_parameters_finder import SGBMParameterFinder
import cv2


if __name__ == '__main__':
    image_l = cv2.imread('im_l.png')
    image_r = cv2.imread('im_r.png')
    sgbm_parameters_finder = SGBMParameterFinder(image_l, image_r, out_filename='out.txt')
    sgbm_parameters_finder.play()
