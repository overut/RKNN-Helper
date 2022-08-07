import argparse
from core.worker import RknnCore

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rknn_model', default='', type=str, help='save rknn model path')
    parser.add_argument('--shape', type=int,nargs='+', default=[1, 3, 256, 512], help='input image size')    
    parser.add_argument('--target_platform', default='', type=str, help='target platform')
    parser.add_argument('--loop_cnt', default=100, type=str, help='loop count,should > 20')

    args = parser.parse_args()

    rknn = RknnCore(args)
    rknn.profile()
    version_info = rknn.version_info()
    print(version_info)
    rknn.release()