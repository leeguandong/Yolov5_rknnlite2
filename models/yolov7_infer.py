'''
@Time    : 2023/3/6 17:45
@Author  : leeguandon@gmail.com
'''
import cv2
import platform
import numpy as np
from .rknn_config import RKNNConfigBoard, RKNNConfigPC
from .onnx_config import ONNXConfig
from .utils import sigmoid, nms_boxes


class RKNNYOLOV7(object):
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
        return outputs

    def box_process(self, position, anchors):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.img_size // grid_h, self.img_size // grid_w]).reshape(1, 2, 1, 1)

        anchors = np.array(anchors)
        anchors = anchors.reshape(*anchors.shape, 1, 1)

        box_xy = position[:, :2, :, :] * 2 - 0.5
        box_wh = pow(position[:, 2:4, :, :] * 2, 2) * anchors

        box_xy += grid
        box_xy *= stride
        box = np.concatenate((box_xy, box_wh), axis=1)

        # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
        xyxy = np.copy(box)
        xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2  # top left x
        xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2  # top left y
        xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2  # bottom right x
        xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2  # bottom right y

        return xyxy

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        if class_num == 1:
            _class_pos = np.where(box_confidences >= self.obj_thresh)
            scores = (box_confidences)[_class_pos]
        else:
            _class_pos = np.where(class_max_score * box_confidences >= self.obj_thresh)
            scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def post_process(self, input_data):
        anchors = [[[12.0, 16.0], [19.0, 36.0], [40.0, 28.0]],
                   [[36.0, 75.0], [76.0, 55.0], [72.0, 146.0]],
                   [[142.0, 110.0], [192.0, 243.0], [459.0, 401.0]]]

        boxes, scores, classes_conf = [], [], []
        # 1*255*h*w -> 3*85*h*w
        input_data = [_in.reshape([len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data]
        for i in range(len(input_data)):
            boxes.append(self.box_process(input_data[i][:, :4, :, :], anchors[i]))
            scores.append(input_data[i][:, 4:5, :, :])
            classes_conf.append(input_data[i][:, 5:, :, :])

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            if c != 0:  # just judge person
                continue
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, self.nms_thresh)

            if len(keep) != 0:
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
