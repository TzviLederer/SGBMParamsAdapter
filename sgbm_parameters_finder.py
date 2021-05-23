import json
import time
import tkinter as tk
from threading import Thread, Event

import cv2
from PIL import Image, ImageTk


class SGBMParameterFinder:
    def __init__(self, image_l, image_r, out_filename=None, resize_ratio=0.5):
        sliders_ranges = {'downscale rate': (1, 10),
                          'minDisparity': (0, 100),
                          'numDisparities': (16, 32 * 16),  # must be divided by 16
                          'blockSize': (1, 31),  # odd number >= 1
                          'P1': (8, 300),  # penalty of changing 1 disparity
                          'P2': (32, 600),  # penalty of changing more than 1 disparity, requires P2 > P1
                          'disp12MaxDiff': (-1, 10),  # maximum pixels difference allowed
                          # of the r->l from the l->r disparity image
                          'preFilterCap': (0, 20),  # derivative clipping size
                          'uniquenessRatio': (0, 10),
                          'speckleWindowSize': (0, 200),
                          'speckleRange': (0, 5)}

        self.gray_column = 0
        self.disparity_column = 1
        self.sliders_column = 2

        self.out_filename = out_filename

        self.image_l = image_l
        self.image_r = image_r
        self.image_list_tk = []

        self.max_display_ims = 3

        self.new_size = tuple([int(x * resize_ratio) for x in self.image_l.shape[:2][::-1]])
        self.gray_new_size = tuple([int(x/self.max_display_ims) for x in self.new_size])

        self.gray_row_span = len(sliders_ranges) * 2 // self.max_display_ims
        self.disparity_row_span = len(sliders_ranges) * 2

        self.stop_event = Event()

        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.prepare_window(sliders_ranges)

    def prepare_window(self, sliders_ranges):
        self.sliders = {title: self.add_slider(self.root, i, slider_range, title, column=self.sliders_column)
                        for i, (title, slider_range) in enumerate(sliders_ranges.items())}
        self.add_ready_label(sliders_ranges)
        self.image = self.refresh_image()

        self.add_gray_images(self.image_l, self.gray_new_size,
                             place=0, row_span=self.gray_row_span, column=self.gray_column)
        self.add_gray_images(self.image_r, self.gray_new_size,
                             place=1, row_span=self.gray_row_span, column=self.gray_column)
        self.add_disparity_image(self.image, self.new_size,
                                 row_span=self.disparity_row_span, column=self.disparity_column)

    def add_ready_label(self, sliders_ranges):
        self.label_ready = tk.StringVar()
        deposit_label = tk.Label(self.root, textvariable=self.label_ready)
        deposit_label.grid(row=len(sliders_ranges) * 2 + 1, column=self.disparity_column)

    def add_disparity_image(self, image, new_size, row_span, column=0):
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        im = Image.fromarray(cv2.resize(image, new_size))
        self.img_tk = ImageTk.PhotoImage(image=im)
        im_tk = tk.Label(self.root, image=self.img_tk)
        im_tk.grid(row=0, column=column, columnspan=1, rowspan=row_span)

    def add_gray_images(self, image, new_size, place, row_span, column=0):
        im = Image.fromarray(cv2.resize(image, new_size))
        self.image_list_tk.append(ImageTk.PhotoImage(image=im))
        img_tk = tk.Label(self.root, image=self.image_list_tk[place])
        img_tk.grid(row=place * row_span, column=column, columnspan=1, rowspan=row_span)

    def play(self):
        t = Thread(target=self.sample_sliders, args=(self.stop_event,))
        t.start()
        self.root.mainloop()

    @staticmethod
    def add_slider(root, row, slider_range, title, column=1):
        w_text = tk.Label(root, text=title)
        w = tk.Scale(root, from_=slider_range[0], to=slider_range[1], orient=tk.HORIZONTAL)

        w_text.grid(row=row * 2, column=column)
        w.grid(row=row * 2 + 1, column=column)

        return w_text, w

    def sample_sliders(self, stop_threads):
        sliders_values = {k: val[1].get() for k, val in self.sliders.items()}
        while not stop_threads.is_set():
            time.sleep(0.1)
            for k, val in self.sliders.items():
                if sliders_values[k] != val[1].get():
                    sliders_values[k] = val[1].get()
                    self.image = self.refresh_image()
                    self.add_disparity_image(self.image, self.new_size,
                                             row_span=self.disparity_row_span, column=self.disparity_column)

    def refresh_image(self):
        self.label_ready.set('working')
        params = {k: val[1].get() for k, val in self.sliders.items()}
        params['numDisparities'] = 16 * int(params['numDisparities'] / 16)
        params['blockSize'] = params['blockSize'] - params['blockSize'] % 2 + 1
        sgbm_params = params.copy()
        sgbm_params.pop('downscale rate')

        stereo = cv2.StereoSGBM_create(**sgbm_params)

        image_l, image_r = self.downscale_images(params['downscale rate'])
        out = stereo.compute(image_l, image_r).astype(float) / 16.

        self.label_ready.set('ready')
        return out

    def downscale_images(self, downscale_rate):
        resize_new_shape = tuple([int(x / downscale_rate) for x in self.image_l.shape[:2][::-1]])
        image_l = cv2.resize(self.image_l, resize_new_shape)
        image_r = cv2.resize(self.image_r, resize_new_shape)
        return image_l, image_r

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
