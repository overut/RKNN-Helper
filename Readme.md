# RKNN-HELPER
## 基本说明
rknn-toolkit2的一个简单封装，方便大家将原生模型装换成rknn支持的格式，并对其进行性能分析
## 使用说明
1. export.py支持模型导出为rknn模型格式，目前仅支持onnx/pytorch(TorchScript)模型导出
- model指定原生pytorch/onnx模型路径
- model_type指定模型为onnx或pytorch
- rknn_model指定rknn模型导出路径
- 修改shape为指定模型的输入
- output_name为模型输出节点名称
- 修改模型mean,std参数，默认为[0,0,0],[255,255,255]
- target_platform指定导出模型平台，默认为None平台指定为PC上的模拟器，若要模型能够在rk3568上执行，指定为'rk3568'
- target_platform可指定平台为“rk3566”,“rk3568”,“rk3588”,“rv1103”,“rv1106”
- 如需导出量化模型，指定--quant参数，同时需要指定dataset的filelist
- dataset filelist格式如下：
```
1.jpg
2.jpg
...
N.jpg
```
- 多输入情况dataset的filelist格式如下
```
1-1.jpg 1-2.jpg #以空格隔开
2-1.jpg 2-2.jpg
...
N-1.jpg N-2.jpg
```
2. profile.py支持rknn模型在指定平台上进行性能分析
- rknn_model指定加载的rknn模型路径
- 修改shape为指定模型的输入
- target_platform指定运行平台，注意这里要和export中指定的target_platform一致
- loop_cnt指定循环评估次数，性能取平均值
