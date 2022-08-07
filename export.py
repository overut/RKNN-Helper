import argparse
from core.worker import RknnCore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='', type=str, help='model path')
    parser.add_argument('--model_type', default='onnx', type=str, help='model type:onnx or pytorch')
    parser.add_argument('--rknn_model', default='', type=str, help='save rknn model path')
    parser.add_argument('--shape', default=[[1,3,256,512]], type=list, help='input shape list')
    parser.add_argument('--output_names', default=[], type=list, help='output_names')
    parser.add_argument('--mean_values', default=[[0,0,0]], type=list, help='mean values')
    parser.add_argument('--std_values', default=[[255,255,255]], type=list, help='std values')
    parser.add_argument('--target_platform', default='', type=str, help='target platform,default is None')
    parser.add_argument('--optimization_level', default=3, type=int, help='optimization level')
    parser.add_argument('--float_dtype', default='float16', type=str, help='float data type,only support fp16 now')
    parser.add_argument('--quant',action='store_true',help='quantize')
    parser.add_argument('--dataset', default='dataset.txt', type=str, help='quantize dataset txt path')

    args = parser.parse_args()

    rknn = RknnCore(args)
    rknn.list_devices()
    rknn.convert()
    rknn.release()