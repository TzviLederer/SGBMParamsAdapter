import tkinter as tk
from PIL import Image, ImageTk
import cv2
from threading import Thread, Event
import time


class SGBMParameterFinder:
    def __init__(self, image_l, image_r, resize_ratio=0.5):
        self.image_l = image_l
        self.image_r = image_r

        self.new_size = tuple([int(x * resize_ratio) for x in self.image_l.shape[:2][::-1]])

        sliders_ranges = {'minDisparity': (0, 100),
                          'numDisparities': (16, 16 * 16),  # must be divided by 16
                          'blockSize': (1, 15),  # odd number >= 1
                          'P1': (8, 300),  # penalty of changing 1 disparity
                          'P2': (32, 600),  # penalty of changing more than 1 disparity, requires P2 > P1
                          'disp12MaxDiff': (-1, 10),    # maximum pixels difference allowed
                                                        # of the r->l from the l->r disparity image
                          'preFilterCap': (0, 20),  # derivative clipping size
                          'uniquenessRatio': (0, 100),
                          'speckleWindowSize': (0, 200),
                          'speckleRange': (0, 5)}
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.sliders = {title: self.add_slider(self.root, i, slider_range, title)
                        for i, (title, slider_range) in enumerate(sliders_ranges.items())}

        self.image = self.refresh_image()
        self.add_image(self.image, self.new_size, len(self.sliders) * 2)

        self.stop_event = Event()

    def add_image(self, image, new_size, sliders_num):
        im = Image.fromarray(cv2.resize(image, new_size))
        self.img_tk = ImageTk.PhotoImage(image=im)
        im_tk = tk.Label(self.root, image=self.img_tk)
        im_tk.grid(row=0, column=0, columnspan=1, rowspan=sliders_num)

    def play(self):
        self.t = Thread(target=self.sample_sliders, args=(self.stop_event, ))
        self.t.start()
        self.root.mainloop()

    @staticmethod
    def add_slider(root, row, slider_range, title):
        w_text = tk.Label(root, text=title)
        w = tk.Scale(root, from_=slider_range[0], to=slider_range[1], orient=tk.HORIZONTAL)

        w_text.grid(row=row * 2, column=1)
        w.grid(row=row * 2 + 1, column=1)

        return w_text, w

    def sample_sliders(self, stop_event):
        sliders_values = {k: val[1].get() for k, val in self.sliders.items()}
        while not stop_event.is_set():
            time.sleep(0.1)
            for k, val in self.sliders.items():
                if sliders_values[k] != val[1].get():
                    sliders_values[k] = val[1].get()
                    self.image = self.refresh_image()
                    self.add_image(self.image, self.new_size, len(self.sliders) * 2)

    def refresh_image(self):
        params = {k: val[1].get() for k, val in self.sliders.items()}
        stereo = cv2.StereoSGBM_create(**params)
        return stereo.compute(self.image_l, self.image_r).astype(float) / 16.

    def on_closing(self):
        self.stop_event.set()
        time.sleep(0.5)
        self.root.destroy()
