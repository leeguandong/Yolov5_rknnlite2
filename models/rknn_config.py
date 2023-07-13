'''
@Time    : 2023/3/6 9:26
@Author  : leeguandon@gmail.com
'''
import os
from pathlib import Path


class RKNNConfigPC:
    def __init__(self,
                 mean_values=None,
                 std_values=None,
                 model_path=None,
                 target='RK3588',
                 verbose=True,
                 export_path=None,
                 do_quantization=False):
        from rknn.api import RKNN
        self.model_path = model_path
        self.target = target
        if export_path is None:
            self.export_path = Path(model_path)
            self.export_path = str(Path(export_path.parent, export_path.stem + "_pc.rknn"))

        if mean_values is None:
            self.mean_values = [[0, 0, 0]]
        else:
            self.mean_values = mean_values
        if std_values is None:
            self.std_values = [[255, 255, 255]]
        else:
            self.std_values = std_values

        # create rknn
        self.rknn = RKNN(verbose)

        # pre-process config
        self.rknn.config(mean_values=self.mean_values,
                         std_values=self.std_values,
                         target_platform=self.target)
        # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')

        # Load ONNX model
        ret = self.rknn.load_onnx(model=export_path)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)

        # Build model
        ret = self.rknn.build(do_quantization=do_quantization, dataset="./data/dataset.txt")
        if ret != 0:
            print('【RKNNConfig】error :Build model failed!')
            exit(ret)

        print('--> Export rknn model')
        ret = self.rknn.export_rknn(self.export_path)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

        ret = self.rknn.init_runtime()
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)

    def infer(self, data):
        if isinstance(data, list) or isinstance(data, tuple):
            pass
        else:
            data = [data]
        outputs = self.rknn.inference(data)
        return outputs

    def release(self):
        self.rknn.release()


class RKNNConfigBoard:
    def __init__(self,
                 rknn_path=None,
                 target='RK3588',
                 verbose=False):
        from rknnlite.api import RKNNLite
        if rknn_path is None:
            print("【RKNNConfig】error: rknn_path is None")
            exit(0)
        else:
            print("【RKNNConfig】: rknn will load by rknn")
            self.model_path = rknn_path
        self.target = target

        # create rknn
        self.rknn = RKNNLite(verbose=verbose)

        # Load ONNX model
        ret = self.rknn.load_rknn(path=self.model_path)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)
        if self.target == "RK3588":
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        else:
            ret = self.rknn.init_runtime()
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)

    def infer(self, data):
        outputs = self.rknn.inference(data)
        return outputs

    def release(self):
        self.rknn.release()
