'''
@Time    : 2023/3/2 16:46
@Author  : leeguandon@gmail.com
'''
import cv2
import platform
import numpy as np
from .rknn_config import RKNNConfigBoard, RKNNConfigPC
from .onnx_config import ONNXConfig
from .utils import sigmoid, nms_boxes


class RKNNYOLOV5(object):
    def __init__(self,
                 RK3588_RKNN_MODEL=None,
                 # RK356X_RKNN_MODEL=None,
                 batch_size=1,
                 img_size=640,
                 obj_thresh=0.25,
                 nms_thresh=0.45,
                 target="rk3588",
                 platform='board',
                 backend='rknn'):
        self.batch_size = batch_size
        self.img_size = img_size
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.target = target
        self.platform = platform
        self.backend = backend

        if self.backend == 'onnxruntime':
            pass
        if self.backend == "rknn":
            if platform == 'board':
                self.model = RKNNConfigBoard(rknn_path=RK3588_RKNN_MODEL, target=self.target)
            if platform == 'pc':
                det_mean = [0.485, 0.456, 0.406]
                det_std = [0.229, 0.224, 0.225]
                rknn_std = [[round(std * 255, 3) for std in det_std]]
                rknn_mean = [[round(mean * 255, 3) for mean in det_mean]]
                self.model = RKNNConfigPC(mean_values=rknn_mean,
                                          std_values=rknn_std,
                                          model_path=RK3588_RKNN_MODEL,
                                          target=self.target,
                                          do_quantization=True)

    def predict_image(self, img):
        outputs = self.model.infer(img)
        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
        return input_data

    def xywh2xyxy(self, x):
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def process(self, input, mask, anchors):
        anchors = [anchors[i] for i in mask]
        grid_h, grid_w = map(int, input.shape[0:2])

        box_confidence = sigmoid(input[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)

        box_class_probs = sigmoid(input[..., 5:])

        box_xy = sigmoid(input[..., :2]) * 2 - 0.5

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)
        box_xy += grid
        box_xy *= int(self.img_size / grid_h)

        box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)
        box_wh = box_wh * anchors

        box = np.concatenate((box_xy, box_wh), axis=-1)

        return box, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        boxes = boxes.reshape(-1, 4)
        box_confidences = box_confidences.reshape(-1)
        box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

        _box_pos = np.where(box_confidences >= self.obj_thresh)
        boxes = boxes[_box_pos]
        box_confidences = box_confidences[_box_pos]
        box_class_probs = box_class_probs[_box_pos]

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= self.obj_thresh)

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        scores = (class_max_score * box_confidences)[_class_pos]

        return boxes, classes, scores

    def post_process(self, input_data):
        masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]

        boxes, classes, scores = [], [], []
        for input, mask in zip(input_data, masks):
            b1, c1, s1 = self.process(input, mask, anchors)
            b1, c1, s1 = self.filter_boxes(b1, c1, s1)
            boxes.append(b1)
            classes.append(c1)
            scores.append(s1)

        boxes = np.concatenate(boxes)
        boxes = self.xywh2xyxy(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            if c != 0:  # just judge person
                continue
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = nms_boxes(b, s, self.nms_thresh)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return {"boxes": None, "classes": None, "scores": None, "status": 0}

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        # return boxes, classes, scores
        return {"boxes": boxes, "classes": classes, "scores": scores, "status": 1}

    def release(self):
        self.model.release()
