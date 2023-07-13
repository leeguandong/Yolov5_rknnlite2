'''
@Time    : 2023/3/2 16:02
@Author  : leeguandon@gmail.com
'''
import os
import cv2
import time
import glob
import numpy as np
import subprocess as sp


def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
        "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
        "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


class PushStream(object):
    def __init__(self, pushurl="rtsp://127.0.0.1:8554/"):
        self.command = ""
        self.pushurl = pushurl

    def initcmd(self, fps, width, height):
        self.command = ['ffmpeg',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(width, height),
                        '-r', str(fps),
                        '-i', '-',
                        '-pix_fmt', 'yuv420p',
                        '-f', 'rtsp',
                        self.pushurl]
        self.pipe = sp.Popen(self.command, stdin=sp.PIPE)


class Times(object):
    def __init__(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class PipeTimer(Times):
    def __init__(self):
        super(PipeTimer, self).__init__()
        self.total_time = Times()
        self.module_time = {
            'pre': Times(),
            'det': Times(),
            'post': Times(),
            'vis': Times()
        }
        self.img_num = 0

    def get_total_time(self):
        total_time = self.total_time.value()
        total_time = round(total_time, 4)
        average_latency = total_time / max(1, self.img_num)
        fps = 0
        if total_time > 0:
            fps = 1 / average_latency
        return total_time, average_latency, fps

    def info(self):
        total_time, average_latency, fps = self.get_total_time()
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000,
                                                       self.img_num))

        for k, v in self.module_time.items():
            v_time = round(v.value(), 4)
            if v_time > 0 and k in ['det', 'pre', 'post', 'vis']:
                print("{} time(ms): {}; per frame average time(ms): {}".format(
                    k, v_time * 1000, v_time * 1000 / self.img_num))

        print("average latency time(ms): {:.2f}, FPS: {:2f}".format(
            average_latency * 1000, fps))
        return fps

    def report(self, average=False):
        dic = {}
        dic['total'] = round(self.total_time.value() / max(1, self.img_num),
                             4) if average else self.total_time.value()
        dic['det'] = round(self.module_time['det'].value() /
                           max(1, self.img_num),
                           4) if average else self.module_time['det'].value()
        dic['pre'] = round(self.module_time['pre'].value() /
                           max(1, self.img_num),
                           4) if average else self.module_time['pre'].value()
        dic['post'] = round(self.module_time['post'].value() /
                            max(1, self.img_num),
                            4) if average else self.module_time['post'].value()
        dic['vis'] = round(self.module_time['vis'].value() /
                           max(1, self.img_num),
                           4) if average else self.module_time['vis'].value()
        dic['img_num'] = self.img_num
        return dic


class Metrics(object):
    def __init__(self, labels, predict):
        if not isinstance(predict, np.ndarray):
            predict = np.array(predict)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.predict = predict
        self.labels = labels
        self._confuse_metrix()

    def _confuse_metrix(self):
        self.tp = sum((self.labels == 1) & (self.predict == 1))
        self.fp = sum((self.labels == 0) & (self.predict == 1))
        self.fn = sum((self.labels == 1) & (self.predict == 0))
        self.tn = sum((self.labels == 0) & (self.predict == 0))

    def recall(self, rem=3):  # 召回率，命中率
        R = self.tp / (self.tp + self.fn)
        return round(R, rem)

    def precision(self, rem=3):  # 精确率
        P = self.tp / (self.tp + self.fp)
        return round(P, rem)

    def accuracy(self, rem=3):  # 准确率
        ACC = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        return round(ACC, rem)

    def falsealarm(self, rem=3):  # 误报率，虚警率
        FPR = self.fp / (self.fp + self.tn)
        return round(FPR, rem)

    def missrate(self, rem=3):  # 漏报率
        FNR = self.fn / (self.tp + self.fn)
        return round(FNR, rem)
