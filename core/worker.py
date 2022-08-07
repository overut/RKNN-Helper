import numpy as np
import time
from rknn.api import RKNN

class RknnCore:
    def __init__(self, args) -> None:
        self.args = args
        self.rknn = RKNN()
        self.is_init = False

    def config(self,
               mean_values=None,
               std_values=None,
               quantized_dtype='asymmetric_quantized-8',
               quantized_algorithm='normal',
               quantized_method='channel',
               target_platform=None,
               quant_img_RGB2BGR=False,
               float_dtype='float16',
               optimization_level=3,
               custom_string=None):
        ret = self.rknn.config(mean_values, std_values, quantized_dtype,
                               quantized_algorithm, quantized_method, 
                               target_platform, quant_img_RGB2BGR,
                               float_dtype, optimization_level, custom_string)   
        self.is_init = True   

    def convert(self):
        if not self.is_init:
            self.config(mean_values=self.args.mean_values,
                        std_values=self.args.std_values,
                        target_platform=self.args.target_platform if self.args.target_platform != '' else None,
                        optimization_level=self.args.optimization_level,
                        float_dtype=self.args.float_dtype)
        print('=============Load Model============')
        if self.args.model_type == 'onnx':
            ret = self.rknn.load_onnx(model=self.args.model, outputs=self.args.output_names)
        elif self.args.model_type == 'pytorch':
            ret = self.rknn.load_pytorch(model=self.args.model, input_size_list=self.args.shape)
        else:
            return False
        if ret != 0:
            print('Load model failed!')
            return False
        print('=============Building Model============')
        ret = self.rknn.build(do_quantization=self.args.quant, dataset=self.args.dataset)
        if ret != 0:
            print('Building model failed!')
            return False
        print('=============Export RKNN model============')
        ret = self.rknn.export_rknn(self.args.rknn_model)
        if ret != 0:
            print('Export RKNN model failed!')
            return False

        print('=============Export RKNN success============')

        return True

    def profile(self):
        print('=============Load Model============')
        ret = self.rknn.load_rknn(self.args.rknn_model)
        if ret != 0:
            print('Load rknn model failed!')
            return False
        print('=============Init Runtime============')
        ret = self.rknn.init_runtime(target=self.args.target_platform if self.args.target_platform != '' else None,perf_debug=False)
        if ret != 0:
            print('Init runtime failed!')
            return False
        print('=============Profile start...============')
        data = np.random.randint(0, 10000, size=self.args.shape)  
        data = data.astype(np.float32) / 10000
        self.rknn.eval_perf(inputs=[data] * self.args.loop_cnt)
        print('=============Profile success============')

        return True
    
    def profile_v2(self):
        print('=============Load Model============')
        ret = self.rknn.load_rknn(self.args.rknn_model)
        if ret != 0:
            print('Load rknn model failed!')
            return False
        print('=============Init Runtime============')
        ret = self.rknn.init_runtime(target=self.args.target_platform if self.args.target_platform != '' else None)
        if ret != 0:
            print('Init runtime failed!')
            return False
        print('=============Profile start...============')
        data = np.random.randint(0, 10000, size=self.args.shape)  
        data = data.astype(np.float32) / 10000
        for i in range(20):
            outputs = self.rknn.inference(inputs=[data])
            
        start = time.time()
        for i in range(self.args.loop_cnt-20):
            self.rknn.inference(inputs=[data])
        end = time.time()
        total_time = end - start
        avg_time = total_time * 1000 / (self.args.loop_cnt-20)
        print('Avg Time(ms): {}'.format(avg_time))
        print('FPS: {}'.format(1000/avg_time))
        print('=============Profile success============')

        return True
    
    def version_info(self):
        return self.rknn.get_sdk_version()
    
    def list_devices(self):
        self.rknn.list_devices()
    
    def release(self):
        self.rknn.release()