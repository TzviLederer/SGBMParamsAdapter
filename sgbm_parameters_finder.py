import json
import time
import tkinter as tk
from threading import Thread, Event

import cv2
from PIL import Image, ImageTk


class SGBMParameterFinder:
    def __init__(self, image_l, image_r, out_filename=None, resize_ratio=0.5):
        self.out_filename = out_filename

        self.image_l = image_l
        self.image_r = image_r

        self.new_size = tuple([int(x * resize_ratio) for x in self.image_l.shape[:2][::-1]])

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
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.sliders = {title: self.add_slider(self.root, i, slider_range, title)
                        for i, (title, slider_range) in enumerate(sliders_ranges.items())}

        self.label_ready = tk.StringVar()
        depositLabel = tk.Label(self.root, textvariable=self.label_ready)
        depositLabel.grid(row=len(sliders_ranges) * 2 + 1, column=0)
        # self.label_ready = tk.Label(self.root, text='ready')
        # self.label_ready.grid(row=len(sliders_ranges) * 2 + 1, column=0)

        self.image = self.refresh_image()
        self.add_image(self.image, self.new_size, len(self.sliders) * 2)

        self.stop_event = Event()

    def add_image(self, image, new_size, sliders_num):
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        im = Image.fromarray(cv2.resize(image, new_size))
        self.img_tk = ImageTk.PhotoImage(image=im)
        im_tk = tk.Label(self.root, image=self.img_tk)
        im_tk.grid(row=0, column=0, columnspan=1, rowspan=sliders_num)

    def play(self):
        self.t = Thread(target=self.sample_sliders, args=(self.stop_event,))
        self.t.start()
        self.root.mainloop()

    @staticmethod
    def add_slider(root, row, slider_range, title):
        w_text = tk.Label(root, text=title)
        w = tk.Scale(root, from_=slider_range[0], to=slider_range[1], orient=tk.HORIZONTAL)

        w_text.grid(row=row * 2, column=1)
        w.grid(row=row * 2 + 1, column=1)

        return w_text, w

    def sample_sliders(self, stop_threads):
        sliders_values = {k: val[1].get() for k, val in self.sliders.items()}
        while not stop_threads.is_set():
            time.sleep(0.1)
            for k, val in self.sliders.items():
                if sliders_values[k] != val[1].get():
                    sliders_values[k] = val[1].get()
                    self.image = self.refresh_image()
                    self.add_image(self.image, self.new_size, len(self.sliders) * 2)

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
