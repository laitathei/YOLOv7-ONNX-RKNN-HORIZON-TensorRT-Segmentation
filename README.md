# YOLOv7-ONNX-RKNN-Segmentation
***Remark: This repo only support 1 batch size***
```
git clone --recursive https://github.com/laitathei/YOLOv7-ONNX-RKNN-Segmentation.git
```
## 0. Environment Setting
```
torch: 1.10.1+cu102
torchvision: 0.11.2+cu102
onnx: 1.10.0
onnxruntime: 1.10.0
```

## 1. Yolov7 Prerequisite
```
pip3 install -r requirements.txt
```

## 2. Convert Pytorch model to ONNX
Remember to change the variable to your setting.
```
python3 pytorch2onnx.py --weights ./model/yolov7-seg.pt --include onnx --img-size 480 640 --simplify
```

## 3. RKNN Prerequisite
Install the wheel according to your python version
```
cd rknn-toolkit2/packages
pip3 install rknn_toolkit2-1.5.0+1fa95b5c-cpxx-cpxx-linux_x86_64.whl
```

## 4. Convert ONNX model to RKNN
Remember to change the variable to your setting
To improve perfermance, you can change ```./config/yolov7-seg-xxx-xxx.quantization.cfg``` layer type.
Please follow [official document](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_User_Guide_RKNN_Toolkit2_EN-1.5.0.pdf) hybrid quatization part and reference to [example program](https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/functions/hybrid_quant) to modify your codes.
```
python3 onnx2rknn_step1.py
```
Add following setting into ```./config/yolov7-seg-xxx-xxx.quantization.cfg``` 
```
custom_quantize_layers:
    528_shape4_Slice_315: float16
    638_shape4_Slice_391: float16
    748_shape4_Slice_467: float16
    528_int8: float16
    638_int8: float16
    748_int8: float16
```
```
python3 onnx2rknn_step2.py
```

## 5. RKNN-Lite Inference
```
python3 rknn_lite_inference.py
```

## Reference
```
https://blog.csdn.net/magic_ll/article/details/131944207
https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation
```

