'''
@Time    : 2023/3/2 15:35
@Author  : leeguandon@gmail.com
'''
import os
import sys
import math
import cv2
import copy
import numpy as np

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from pipe_utils import get_test_images, PipeTimer, decode_image, PushStream, Metrics
from cfg_utils import argsparser, print_arguments, merge_cfg
from models.yolov5_infer import RKNNYOLOV5
from models.yolov7_infer import RKNNYOLOV7
from models.yolox_infer import RKNNYOLOX
from models.utils import letterbox


class Pipeline(object):
    def __init__(self, args, cfg):
        self.multi_camera = False
        self.is_video = False
        self.output_dir = args.output_dir
        self.input = self._parse_input(args.image_file, args.image_dir,
                                       args.video_file, args.video_dir,
                                       args.camera_id, args.rtsp)
        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(self.input)

    def _parse_input(self, image_file, image_dir, video_file, video_dir,
                     camera_id, rtsp):
        # parse input as is_video and multi_camera
        if image_file is not None or image_dir is not None:
            input = get_test_images(image_dir, image_file)
            self.is_video = False
            self.multi_camera = False

        elif video_file is not None:
            assert os.path.exists(
                video_file
            ) or 'rtsp' in video_file, "video_file not exists and not an rtsp site."
            self.multi_camera = False
            input = video_file
            self.is_video = True

        elif video_dir is not None:
            videof = [os.path.join(video_dir, x) for x in os.listdir(video_dir)]
            if len(videof) > 1:
                self.multi_camera = True
                videof.sort()
                input = videof
            else:
                input = videof[0]
            self.is_video = True

        elif rtsp is not None:
            if len(rtsp) > 1:
                rtsp = [rtsp_item for rtsp_item in rtsp if 'rtsp' in rtsp_item]
                self.multi_camera = True
                input = rtsp
            else:
                self.multi_camera = False
                input = rtsp[0]
            self.is_video = True

        elif camera_id != -1:
            self.multi_camera = False
            input = camera_id
            self.is_video = True

        else:
            raise ValueError(
                "Illegal Input, please set one of ['video_file', 'camera_id', 'image_file', 'image_dir']"
            )

        return input

    def run_multithreads(self):
        import threading
        if self.multi_camera:
            multi_res = []
            threads = []
            for idx, (predictor, input) in enumerate(zip(self.predictor, self.input)):
                thread = threading.Thread(
                    name=str(idx).zfill(3),
                    target=predictor.run,
                    args=(input, idx))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for predictor, thread in zip(self.predictor, threads):
                thread.join()
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
        else:
            self.predictor.run(self.input)

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
        else:
            self.predictor.run(self.input)


class PipePredictor(object):
    def __init__(self, args, cfg, is_video=True, multi_camera=False):
        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.classes = cfg['classes']
        self.frame_interval = cfg['frame_interval']
        self.moving_avg = cfg['moving_avg']
        self.pushurl = args.pushurl
        self.output_dir = args.output_dir
        self.region_polygon = args.region_polygon
        self.metrics = args.metrics
        self.target = args.target
        self.platform = args.platform
        self.backend = args.backend
        self.visual = args.visual
        self.pipe_timer = PipeTimer()

        # model Initialization
        self.with_yolov5 = cfg.get('YOLOV5', False)['enable'] if cfg.get('YOLOV5', False) else False
        if self.with_yolov5:
            det_cfg = self.cfg['YOLOV5']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            img_size = det_cfg['size']
            obj_thresh = det_cfg['obj_thresh']
            nms_thresh = det_cfg['nms_thresh']
            self.det_predictor = RKNNYOLOV5(
                RK3588_RKNN_MODEL=model_dir,
                batch_size=batch_size,
                img_size=img_size,
                obj_thresh=obj_thresh,
                nms_thresh=nms_thresh,
                target=self.target,
                platform=self.platform,
                backend=self.backend
            )

        self.with_yolov7 = cfg.get("YOLOV7", False)['enable'] if cfg.get('YOLOV7', False) else False
        if self.with_yolov7:
            det_cfg = self.cfg['YOLOV7']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            img_size = det_cfg['size']
            obj_thresh = det_cfg['obj_thresh']
            nms_thresh = det_cfg['nms_thresh']
            self.det_predictor = RKNNYOLOV7(
                RK3588_RKNN_MODEL=model_dir,
                batch_size=batch_size,
                img_size=img_size,
                obj_thresh=obj_thresh,
                nms_thresh=nms_thresh,
                target=self.target,
                platform=self.platform,
                backend=self.backend
            )

        self.with_yolox = cfg.get("YOLOX", False)['enable'] if cfg.get('YOLOX', False) else False
        if self.with_yolox:
            det_cfg = self.cfg['YOLOX']
            model_dir = det_cfg['model_dir']
            batch_size = det_cfg['batch_size']
            img_size = det_cfg['size']
            obj_thresh = det_cfg['obj_thresh']
            nms_thresh = det_cfg['nms_thresh']
            self.det_predictor = RKNNYOLOX(
                RK3588_RKNN_MODEL=model_dir,
                batch_size=batch_size,
                img_size=img_size,
                obj_thresh=obj_thresh,
                nms_thresh=nms_thresh,
                target=self.target,
                platform=self.platform,
                backend=self.backend
            )

        self.in_time = 0  # appear time
        self.out_time = 0  # disappear time
        self.datacollector = {}
        self.gt = []
        self.prd = []

    def set_file_name(self, path):
        if path is not None:
            self.file_name = os.path.split(path)[-1]
            if "." in self.file_name:
                self.file_name = self.file_name.split(".")[-2]
        else:
            # use camera id
            self.file_name = None

    def run(self, input, thread_idx=0):
        if self.is_video:
            self.predict_video(input, thread_idx=thread_idx)
        else:
            self.predict_image(input)
        self.pipe_timer.info()

    def predict_image(self, input):
        batch_loop_cnt = math.ceil(float(len(input)) / self.cfg.batch_size)
        # rknn not support batch mode
        for i in range(batch_loop_cnt):
            start_index = i * self.det_predictor.batch_size
            end_index = min((i + 1) * self.det_predictor.batch_size, len(input))
            batch_file = input[start_index:end_index]

            self.pipe_timer.total_time.start()
            self.pipe_timer.module_time['pre'].start()
            batch_input = [decode_image(f, {})[0] for f in batch_file]
            batch_input_bgr = [
                letterbox(
                    f,
                    img_shape=(self.det_predictor.img_size, self.det_predictor.img_size))[0]
                for f in batch_input]
            batch_input = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_input_bgr]
            self.pipe_timer.module_time['pre'].end()

            self.pipe_timer.module_time['det'].start()
            det_res = self.det_predictor.predict_image(batch_input)
            self.pipe_timer.module_time['det'].end()

            self.pipe_timer.module_time['post'].start()
            res_dict = self.det_predictor.post_process(det_res)
            self.pipe_timer.module_time['post'].end()

            self.pipe_timer.img_num += len(batch_input)
            self.pipe_timer.total_time.end()

            self.pipe_timer.module_time['vis'].start()
            if self.visual:
                self.visualize_image(batch_file, batch_input_bgr, res_dict)
            self.pipe_timer.module_time['vis'].end()
        self.det_predictor.release()

    def predict_video(self, video_file, thread_idx=0):
        capture = cv2.VideoCapture(video_file)

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("video fps: %d, frame_count: %d" % (fps, frame_count))

        if len(self.pushurl) > 0:
            video_out_name = 'output' if self.file_name is None else self.file_name
            pushurl = os.path.join(self.pushurl, video_out_name)
            print("the result will push stream to url:{}".format(pushurl))
            pushstream = PushStream(pushurl)
            pushstream.initcmd(fps, width, height)
        elif self.visual:
            video_out_name = 'output' if self.file_name is None else self.file_name
            if "rtsp" in video_file:
                video_out_name = video_out_name + "_t" + str(thread_idx).zfill(
                    2) + "_rtsp"
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, video_out_name + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_id = 0
        entrance = []
        if len(self.region_polygon) > 0:
            assert len(
                self.region_polygon
            ) % 2 == 0, "region_polygon should be pairs of coords points when det in region-specific dtection."
            assert len(
                self.region_polygon
            ) > 6, 'region_polygon should be at least 3 pairs of point coords.'

            for i in range(0, len(self.region_polygon), 2):
                entrance.append(
                    [self.region_polygon[i], self.region_polygon[i + 1]])
                # entrance.append([width, height])

        while (1):
            ret, frame = capture.read()
            if not ret:
                break

            if frame_id % self.frame_interval == 0:
                print('Thread: {}; frame id: {}'.format(thread_idx, frame_id))
                self.gt += [1]

                self.pipe_timer.total_time.start()
                self.pipe_timer.module_time['pre'].start()
                if self.region_polygon:
                    frame = self.get_region(entrance, frame)
                frame = letterbox(
                    frame,
                    img_shape=(self.det_predictor.img_size, self.det_predictor.img_size))[0]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.pipe_timer.module_time['pre'].end()

                self.pipe_timer.module_time['det'].start()
                det_res = self.det_predictor.predict_image([copy.deepcopy(frame_rgb)])
                self.pipe_timer.module_time['det'].end()

                self.pipe_timer.module_time['post'].start()
                res_dict = self.det_predictor.post_process(det_res)
                if self.moving_avg:
                    status = self.judge_status(res_dict)
                else:
                    status = res_dict.get("status", 0)
                self.pipe_timer.module_time['post'].end()

                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()

                self.pipe_timer.module_time['vis'].start()
                if self.visual:
                    im = self.visualize_video(frame, res_dict, frame_id)
                    if len(self.pushurl) > 0:
                        pushstream.pipe.stdin.write(im.tobytes())
                    else:
                        writer.write(im)
                        if self.file_name is None:  # use camera_id
                            cv2.imshow('RNKK-Pipeline', im)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                self.pipe_timer.module_time['vis'].end()

                # send status
                self.prd += [status]
                print(f"是否有人: {status}")
                # if status != 1:
                #     cv2.imwrite(
                #         os.path.join(self.output_dir, '{:05d}.jpg'.format(frame_id)), im)

            frame_id += 1
        if self.visual and len(self.pushurl) == 0:
            writer.release()
            print('save result to {}'.format(out_path))
        self.det_predictor.release()

        if self.metrics:
            metrics = Metrics(self.gt, self.prd)
            tp, fp, fn, tn = metrics.tp, metrics.fp, metrics.fn, metrics.tn
            accuracy = metrics.accuracy()
            recall = metrics.recall()
            precision = metrics.precision()
            falsealarm = metrics.falsealarm()
            missrate = metrics.missrate()
            print(f'TP:{tp}, FP:{fp}, FN:{fn}, TN:{tn}')
            print(f'The accuracy:{accuracy}, recall:{recall}, precision:{precision},'
                  f'falsealarm:{falsealarm},missrate:{missrate}')

    def visualize_image(self, im_files, images, res_dict):
        boxes = res_dict.get("boxes", [])
        scores = res_dict.get("scores", [])
        classes = res_dict.get("classes", [])

        for i, (im_file, im) in enumerate(zip(im_files, images)):
            for box, score, cl in zip(boxes, scores, classes):
                top, left, right, bottom = box
                print('class: {}, score: {}'.format(self.classes[cl], score))
                print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
                top = int(top)
                left = int(left)
                right = int(right)
                bottom = int(bottom)

                cv2.rectangle(im, (top, left), (right, bottom), (255, 0, 0), 2)
                cv2.putText(im, '{0} {1:.2f}'.format(self.classes[cl], score),
                            (top, left - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            img_name = os.path.split(im_file)[-1]
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            out_path = os.path.join(self.output_dir, img_name)
            cv2.imwrite(out_path, im)
            print("save result to: " + out_path)

    def visualize_video(self, image, res_dict, frame_id):
        boxes = res_dict.get("boxes", [])
        scores = res_dict.get("scores", [])
        classes = res_dict.get("classes", [])
        try:
            for box, score, cl in zip(boxes, scores, classes):
                top, left, right, bottom = box
                print('class: {}, score: {}'.format(self.classes[cl], score))
                print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
                top = int(top)
                left = int(left)
                right = int(right)
                bottom = int(bottom)

                cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
                cv2.putText(image, '{0} {1:.2f}'.format(self.classes[cl], score),
                            (top, left - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
        except:
            pass
        cv2.imwrite(
            os.path.join(self.output_dir, '{:05d}.jpg'.format(frame_id)), image)
        return image

    def get_region(self, entrance, frame):
        # method 1
        entrance = np.array(entrance[:-1])
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.polylines(mask, entrance, 1, 255)
        cv2.fillPoly(mask, entrance, 255)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # method 2 just support 4 point
        xmin = int(entrance[0][0])
        ymin = int(entrance[0][1])
        xmax = int(entrance[1][0])
        ymax = int(entrance[1][1])
        frame = frame[ymin:ymax, xmin:xmax, :]
        return frame

    def judge_status(self, res_dict):
        status = res_dict.get("status", 0)
        if status not in self.datacollector:
            self.datacollector[status] = 1
        else:
            self.datacollector[status] += 1

        if status in self.datacollector and (self.datacollector[status] > max(6, self.in_time)):
            status = 1
            self.in_time = self.datacollector[status]
            self.out_time = 0
        else:
            self.out_time += 1
        if self.out_time > 6:
            status = 0
            self.datacollector.clear()
            self.in_time = 0
            self.out_time = 0
        return status


def main():
    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    # pipeline.run()
    pipeline.run_multithreads()


if __name__ == "__main__":
    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    main()
