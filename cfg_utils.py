import copy
import yaml
from argparse import ArgumentParser


def argsparser():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="infer_person_cfg.yml",
        help=("Path of configure"))
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Dir of image file, `image_file` has a higher priority.")
    parser.add_argument(
        "--video_file",
        type=str,
        default="data/kitch.mp4",
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Dir of video file, `video_file` has a higher priority.")
    parser.add_argument(
        "--rtsp",
        type=str,
        nargs='+',
        default=None,
        help="list of rtsp inputs, for one or multiple rtsp input.")
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="device id of camera to predict.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--pushurl",
        type=str,
        default="",  # 对预测结果视频进行推流的地址，以rtsp://开头，该选项优先级高于视频结果本地存储，
        # 打开时不再另外存储本地预测结果视频, 默认为空，表示没有开启
        help="url of output visualization stream.")
    parser.add_argument(
        '--region_polygon',
        nargs='+',
        type=int,
        default=[],
        help="Clockwise point coords (x0,y0,x1,y1...) of polygon of area when "
             "do_break_in_counting. Note that only support single-class MOT and "
             "the video should be taken by a static camera.")
    parser.add_argument(
        '--visual',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--metrics',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--target',
        type=str,
        default='RK3588'
    )
    parser.add_argument(
        '--platform',
        type=str,
        default='pc',
        help='pc/board,pc just for rknn-toolkit2 test,rk3588 for inference'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='rknn',
        help='rknn/onnxruntime, rknn for npu,onnxruntime for cpu'
    )

    return parser


def merge_cfg(args):
    # load config
    with open(args.config) as f:
        pred_config = yaml.safe_load(f)

    def merge(cfg, arg):
        # update cfg from arg directly
        merge_cfg = copy.deepcopy(cfg)
        for k, v in cfg.items():
            if k in arg:
                merge_cfg[k] = arg[k]
            else:
                if isinstance(v, dict):
                    merge_cfg[k] = merge(v, arg)

        return merge_cfg

    def merge_opt(cfg, arg):
        merge_cfg = copy.deepcopy(cfg)
        # merge opt
        if 'opt' in arg.keys() and arg['opt']:
            for name, value in arg['opt'].items(
            ):  # example: {'MOT': {'batch_size': 3}}
                if name not in merge_cfg.keys():
                    print("No", name, "in config file!")
                    continue
                for sub_k, sub_v in value.items():
                    if sub_k not in merge_cfg[name].keys():
                        print("No", sub_k, "in config file of", name, "!")
                        continue
                    merge_cfg[name][sub_k] = sub_v

        return merge_cfg

    args_dict = vars(args)
    pred_config = merge(pred_config, args_dict)
    pred_config = merge_opt(pred_config, args_dict)

    return pred_config


def print_arguments(cfg):
    print('-----------  Running Arguments -----------')
    buffer = yaml.dump(cfg)
    print(buffer)
    print('------------------------------------------')
