import json
import time
import tkinter as tk
from threading import Thread, Event

import cv2
import numpy as np
from PIL import Image, ImageTk

sliders_ranges = {'downscale rate': (1, 10),
                  'minDisparity': (0, 100),
                  'numDisparities': (16, 32 * 16),  # must be divided by 16
                  'blockSize': (1, 31),  # odd number >= 1
                  'P1': (8, 3000),  # penalty of changing 1 disparity
                  'P2': (32, 6000),  # penalty of changing more than 1 disparity, requires P2 > P1
                  'disp12MaxDiff': (-1, 10),  # maximum pixels difference allowed
                  # of the r->l from the l->r disparity image
                  'preFilterCap': (0, 20),  # derivative clipping size
                  'uniquenessRatio': (0, 10),
                  'speckleWindowSize': (0, 200),
                  'speckleRange': (0, 5),
                  '100sigma': (0, 300),
                  'lambda': (0, 300000),
                  '100 gamma l': (1, 200), '100 gamma r': (1, 200),
                  'opacity': (0, 100)}

default_values = {'numDisparities': 128, 'blockSize': 5, 'P1': 8 * 3 * 5 * 5, 'P2': 32 * 3 * 5 * 5,
                       'disp12MaxDiff': -1, '100sigma': 100, 'lambda': 8000,
                       '100 gamma l': 100, '100 gamma r': 100}

non_sgbm_params = ['100sigma', 'lambda', 'downscale rate', '100 gamma l', '100 gamma r', 'opacity']
gray_params = ['100 gamma l', '100 gamma r']

# column number in the display window
gray_column = 0
disparity_column = 1
sliders_column = 2

max_display_ims = 3


class SGBMParameterFinder:
    def __init__(self, image_l, image_r, out_filename=None, resize_ratio=0.5):
        self.stop_event = Event()
        self.sliders = None
        self.image = None

        self.out_filename = out_filename

        self.image_l_org = image_l
        self.image_r_org = image_r

        self.image_l = image_l
        self.image_r = image_r

        self.image_list_tk = [self.image_l, self.image_r, None]

        self.new_size = tuple([int(x * resize_ratio) for x in self.image_l.shape[:2][::-1]])
        self.gray_new_size = tuple([int(x / max_display_ims) for x in self.new_size])

        self.gray_row_span = len(sliders_ranges) * 2 // max_display_ims
        self.disparity_row_span = len(sliders_ranges) * 2

        self.start_tk_loop()

    def play(self):
        t = Thread(target=self.sample_sliders, args=(self.stop_event,))
        t.start()
        self.root.mainloop()

    def start_tk_loop(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.prepare_window()

    def prepare_window(self):
        # add sliders and ready label
        self.prepare_sliders()
        self.add_ready_label()

        # add disparity images
        self.image, wls_im = self.refresh_image()
        self.add_disparity_image(self.image, self.new_size,
                                 row_span=self.disparity_row_span, column=disparity_column)
        self.image_list_tk[-1] = to_colormap(wls_im)

        # add gray images
        for i, im_display in enumerate(self.image_list_tk):
            self.add_gray_images(im_display, self.gray_new_size,
                                 place=i, row_span=self.gray_row_span, column=gray_column)

    def prepare_sliders(self):
        self.sliders = {title: self.add_slider(self.root, i, slider_range, title, column=sliders_column)
                        for i, (title, slider_range) in enumerate(sliders_ranges.items())}
        for k, val in self.sliders.items():
            if k in default_values.keys():
                val[1].set(default_values[k])

    def add_ready_label(self):
        self.label_ready = tk.StringVar()
        deposit_label = tk.Label(self.root, textvariable=self.label_ready)
        deposit_label.grid(row=len(sliders_ranges) * 2 + 1, column=disparity_column)

    def add_disparity_image(self, image, new_size, row_span, column=0):
        image = to_colormap(image)
        image = self.set_opacity(image)

        self.img_tk = self.convert_to_tkimage(image, new_size)
        im_tk = tk.Label(self.root, image=self.img_tk)
        im_tk.grid(row=0, column=column, columnspan=1, rowspan=row_span)

    def convert_to_tkimage(self, image, new_size):
        im = Image.fromarray(cv2.resize(image, new_size))
        img_tk = ImageTk.PhotoImage(image=im)
        return img_tk

    def set_opacity(self, image):
        opacity = self.sliders['opacity'][1].get()
        image = np.uint8((image / 255 * (1 - opacity / 100) + self.image_l / 255 * (opacity / 100)) * 255)
        return image

    def add_gray_images(self, image, new_size, place, row_span, column=0):
        self.image_list_tk[place] = self.convert_to_tkimage(image=image, new_size=new_size)
        img_tk = tk.Label(self.root, image=self.image_list_tk[place])
        img_tk.grid(row=place * row_span, column=column, columnspan=1, rowspan=row_span)

    @staticmethod
    def add_slider(root, row, slider_range, title, column=1):
        w_text = tk.Label(root, text=title)
        w = tk.Scale(root, from_=slider_range[0], to=slider_range[1], orient=tk.HORIZONTAL)

        w_text.grid(row=row * 2 + 1, column=column)
        w.grid(row=row * 2, column=column)

        return w_text, w

    def sample_sliders(self, stop_threads):
        sliders_values = {k: val[1].get() for k, val in self.sliders.items()}
        while not stop_threads.is_set():
            time.sleep(0.1)
            for k, val in self.sliders.items():
                if sliders_values[k] != val[1].get():
                    sliders_values[k] = val[1].get()

                    self.prepare_gray_images(gamma_l=sliders_values['100 gamma l'], gamma_r=sliders_values['100 gamma r'])
                    wls_img = self.prepare_disparity_images()

                    self.display_images(wls_img)

    def display_images(self, wls_img):
        self.add_disparity_image(self.image, self.new_size,
                                 row_span=self.disparity_row_span, column=disparity_column)
        self.add_gray_images(wls_img, self.gray_new_size, max_display_ims - 1,
                             self.gray_row_span, gray_column)
        self.add_gray_images(self.image_l, self.gray_new_size, 0, self.gray_row_span, gray_column)
        self.add_gray_images(self.image_r, self.gray_new_size, 1, self.gray_row_span, gray_column)

    def prepare_disparity_images(self):
        self.image, wls_img = self.refresh_image()
        wls_img = to_colormap(wls_img)
        wls_img = self.set_opacity(wls_img)
        return wls_img

    def prepare_gray_images(self, gamma_l, gamma_r):
        self.image_l = gamma_correction(self.image_l_org, gamma=gamma_l / 100)
        self.image_r = gamma_correction(self.image_r_org, gamma=gamma_r / 100)

    def refresh_image(self):
        self.label_ready.set('working')
        disparity_image, wls_im = self.compute_disparity()
        self.label_ready.set('ready')

        return disparity_image, wls_im

    def compute_disparity(self):
        params, sgbm_params = self.prepare_sgbm_params()
        stereo = cv2.StereoSGBM_create(**sgbm_params)
        image_l, image_r = self.downscale_images(params['downscale rate'])
        disparity_image = stereo.compute(image_l, image_r).astype(float) / 16.
        wls_im = self.wls_filter(stereo)
        return disparity_image, wls_im

    def adjust_param(self, params, param_name, f):
        params[param_name] = f(params[param_name])
        self.sliders[param_name][1].set(params[param_name])
        return params

    def prepare_sgbm_params(self):
        params = {k: val[1].get() for k, val in self.sliders.items()}

        params = self.adjust_param(params, param_name='numDisparities', f=f_modulo_16)
        params = self.adjust_param(params, param_name='blockSize', f=f_to_odd)

        sgbm_params = params.copy()
        for k in non_sgbm_params:
            sgbm_params.pop(k)

        return params, sgbm_params

    def downscale_images(self, downscale_rate):
        resize_new_shape = tuple([int(x / downscale_rate) for x in self.image_l.shape[:2][::-1]])
        image_l = cv2.resize(self.image_l, resize_new_shape)
        image_r = cv2.resize(self.image_r, resize_new_shape)
        return image_l, image_r

    def wls_filter(self, stereo):
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        left_disp = stereo.compute(self.image_l, self.image_r)
        right_disp = right_matcher.compute(self.image_r, self.image_l)

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        params = {k: val[1].get() for k, val in self.sliders.items() if k in ['100sigma', 'lambda']}
        wls_filter.setLambda(params['lambda'])
        wls_filter.setSigmaColor(params['100sigma'] / 100)
        return wls_filter.filter(left_disp, self.image_l, disparity_map_right=right_disp)

    def on_closing(self):
        if self.out_filename is not None:
            self.save_params(self.out_filename)

        self.stop_event = True
        time.sleep(0.5)
        self.root.destroy()

    def save_params(self, filename):
        sliders_values = {k: val[1].get() for k, val in self.sliders.items()}
        with open(filename, 'w') as outfile:
            json.dump(sliders_values, outfile)


def gamma_correction(image, gamma=1):
    image = image / 255
    return np.uint8(255 * image ** gamma)


def to_colormap(image):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image


def f_to_odd(num):
    return num - num % 2 + 1


def f_modulo_16(num):
    return 16 * int(num / 16)
