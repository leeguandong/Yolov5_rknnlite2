'''
@Time    : 2023/3/6 9:27
@Author  : leeguandon@gmail.com
'''
import onnxruntime as rt


class ONNXConfig:
    def __init__(self, onnx_model_path=None, need_show=False):
        # 判断基本条件是否被满足
        assert (onnx_model_path is not None), "onnx_model_path is empty"

        # 读取模型
        self.onnx_model_path = onnx_model_path

        # 获取模型的输入和输出
        self.sess = rt.InferenceSession(onnx_model_path)
        self.input_name = [input_name.name for input_name in self.sess.get_inputs()]
        self.input_shape = [input_name.shape for input_name in self.sess.get_inputs()]
        self.output_name = [output_name.name for output_name in self.sess.get_outputs()]
        if need_show:
            print("****************************** ONNXConfig ******************************")
            print("-> onnx path is", self.onnx_model_path)
            print("-> onnx input_name is", self.input_name)
            print("-> onnx input_shape is", self.input_shape)
            print("-> onnx output_name is", self.output_name)
            print("****************************** ONNXConfig ******************************")

    def infer(self, data):
        data = data[0]
        outputs = self.sess.run(self.output_name, {self.input_name[0]: data})
        return outputs
