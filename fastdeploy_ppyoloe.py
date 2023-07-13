'''
@Time    : 2023/2/27 14:37
@Author  : leeguandon@gmail.com
'''
import os
import fastdeploy as fd
import cv2
import numpy as np
from ocsort_tracker import OCSORTTracker
from collections import defaultdict
from pathlib import Path
import copy
import time

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
CLASSES = ["player"]
visual = True


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def ppyolo_nms(pred_bboxes, pred_scores):
    boxes = pred_bboxes.reshape(-1, 4)
    box_confidences = np.ones(pred_bboxes.shape[0]).reshape(-1, )
    box_class_probs = pred_scores.reshape(pred_scores.shape[-1], -1)

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score * box_confidences)[_class_pos]

    nboxes, nclasses, nscores = [], [], []

    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def postprocess(boxes, classes, scores):
    try:
        nboxes = []
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                classes_scores = np.append(classes[i], scores[i])
                boxes_ = np.append(classes_scores, box)
                nboxes.append(boxes_.tolist())
        result = {"boxes": np.array(nboxes), 'boxes_num': np.array([len(boxes)])}
        return result
    except:
        result = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
        return result


class SDE_Detector(object):
    def __init__(self):
        use_byte = False
        det_thresh = 0.4
        max_age = 30
        min_hits = 3
        iou_threshold = 0.3
        delta_t = 3
        inertia = 0.2
        min_box_area = 0
        vertical_ratio = 0

        self.tracker = OCSORTTracker(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            inertia=inertia,
            min_box_area=min_box_area,
            vertical_ratio=vertical_ratio,
            use_byte=use_byte)

    def tracking(self, det_results):
        pred_dets = det_results['boxes']
        pred_embs = det_results.get('embeddings', None)

        online_targets = self.tracker.update(pred_dets, pred_embs)
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        for t in online_targets:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]  # top,left,w,h
            tscore = float(t[4])
            tid = int(t[5])
            if tlwh[2] * tlwh[3] <= self.tracker.min_box_area: continue
            if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                3] > self.tracker.vertical_ratio:
                continue
            if tlwh[2] * tlwh[3] > 0:
                online_tlwhs[0].append(tlwh)
                online_ids[0].append(tid)
                online_scores[0].append(tscore)
        tracking_outs = {
            'online_tlwhs': online_tlwhs,  # 坐标
            'online_scores': online_scores,  # >0.4
            'online_ids': online_ids,  # [10,9,8,7,6,5,4,3,2,1]
        }
        return tracking_outs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_tracking_dict(image,
                       num_classes,
                       tlwhs_dict,
                       obj_ids_dict,
                       scores_dict,
                       frame_id=0,
                       fps=0.,
                       ids2names=[]):
    im = np.ascontiguousarray(np.copy(image))  # shape：480,854,3
    im_h, im_w = im.shape[:2]
    text_scale = max(0.5, image.shape[1] / 3000.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    for cls_id in range(num_classes):
        tlwhs = tlwhs_dict[cls_id]
        obj_ids = obj_ids_dict[cls_id]
        scores = scores_dict[cls_id]
        cv2.putText(
            im,
            'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
            (0, int(15 * text_scale) + 5),
            cv2.FONT_ITALIC,
            text_scale, (0, 0, 255),
            thickness=text_thickness)

        record_id = set()
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            center = tuple(map(int, (x1 + w / 2., y1 + h / 2.)))
            obj_id = int(obj_ids[i])

            id_text = '{}'.format(int(obj_id))
            if ids2names != []:
                id_text = '{}_{}'.format(ids2names[cls_id], id_text)
            else:
                id_text = 'class{}_{}'.format(cls_id, id_text)

            _line_thickness = 1 if obj_id <= 0 else line_thickness

            in_region = False
            color = get_color(abs(obj_id)) if in_region == False else (0, 0,
                                                                       255)
            cv2.rectangle(
                im,
                intbox[0:2],
                intbox[2:4],
                color=color,
                thickness=line_thickness)
            cv2.putText(
                im,
                id_text, (intbox[0], intbox[1] - 25),
                cv2.FONT_ITALIC,
                text_scale,
                color,
                thickness=text_thickness)
    return im


option = fd.RuntimeOption()
# option.use_rknpu2() #
# option.use_cpu()
# option.use_openvino_backend() # 一行命令切换使用 OpenVINO部署

# model = fd.vision.detection.PPYOLOE(
#     "/home/sniss/local_disk/rknn-toolkit2-master/examples/onnx/ppyoloe/paddle_ori/mot_ppyoloe_l_36e_pipeline/model.pdmodel",
#     "/home/sniss/local_disk/rknn-toolkit2-master/examples/onnx/ppyoloe/paddle_ori/mot_ppyoloe_l_36e_pipeline/model.pdiparams" ,
# "/home/sniss/local_disk/rknn-toolkit2-master/examples/onnx/ppyoloe/paddle_ori/mot_ppyoloe_l_36e_pipeline/infer_cfg.yml")

model = fd.vision.detection.PPYOLOE(
    "E:/comprehensive_library/Xiaobao/weights/football/mot_ppyoloe_l_36e_pipeline/model.pdmodel",
    "E:/comprehensive_library/Xiaobao/weights/football/mot_ppyoloe_l_36e_pipeline/model.pdiparams",
    "E:/comprehensive_library/Xiaobao/weights/football/mot_ppyoloe_l_36e_pipeline/infer_cfg.yml")

tracker = SDE_Detector()
mot_results = []

video_file = "kitch.mp4"
output_dir = "results"
capture = cv2.VideoCapture(video_file)

# Get Video info : resolution, fps, frame count
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(cv2.CAP_PROP_FPS))
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("video fps: %d, frame_count: %d" % (fps, frame_count))

video_out_name = Path(video_file).stem
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
out_path = os.path.join(output_dir, video_out_name + ".mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

frame_id = 0
start = time.time()
while (1):
    # frame_id = 0
    if frame_id % 10 == 0:
        print('frame id: {}'.format(frame_id))

    ret, frame = capture.read()
    if not ret:
        break
    # img = cv2.imread("test.png")
    img = frame
    result = model.predict(copy.deepcopy(img))

    pred_bboxes = np.array(result.boxes)
    pred_scores = np.array(result.scores)

    boxes, classes, scores = ppyolo_nms(pred_bboxes, pred_scores)

    # boxes
    # array([[ 618.63458252,  172.54750061, 1023.77459717,  781.89233398]])
    # classes
    # array([0])
    #  scores
    # array([0.95259225])

    det_result = postprocess(boxes, classes, scores)
    tracking_outs = tracker.tracking(det_result)
    online_tlwhs = tracking_outs['online_tlwhs']
    online_scores = tracking_outs['online_scores']
    online_ids = tracking_outs['online_ids']
    mot_results.append([online_tlwhs, online_scores, online_ids])

    if visual:
        im = plot_tracking_dict(
            frame,
            1,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            ids2names=CLASSES)
        cv2.imwrite(
            os.path.join(output_dir, '{:05d}.jpg'.format(frame_id)), im)

    frame_id += 1

    writer.write(im)

    # img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # if boxes is not None:
    #     draw(img_1, boxes, scores, classes)
    #
    # img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("fastdeploy_1.png", img_2)

end = time.time()
print("total time:", end - start)
print("fps:", frame_id / (end - start))

writer.release()
