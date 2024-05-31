<a name="CFCrB"></a>
# 1 实验内容
- 了解机器学习基础
- 了解TensorFlow及TensorFlowLite 
- 按照教程完成基于TensorFlowLite Model Maker的花卉模型生成
- 使用实验三的应用验证生成的模型 
- 将上述完成的JupyterNotebook在Github上进行共享  

<a name="a0xmM"></a>
# 2 实验记录
<a name="ev8JI"></a>
## 2.1 准备工作
<a name="BVy6h"></a>
### 2.1.1 创建 codespace
在github提供的codespace环境里选择空白模板进行创建：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717154716971-a52ee3fe-b0b2-44c9-80a9-98974941fda6.png#averageHue=%23e9e1a8&clientId=u40004ca1-2836-4&from=paste&height=194&id=uccacf4aa&originHeight=267&originWidth=682&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=24232&status=done&style=none&taskId=u73771bc2-8e88-484c-93e2-364a0a2ce33&title=&width=496)<br />在模板工程中新建一个Jupyter notebook文件，后缀名为.ipynb。<br />![G53VLRODOKK0LT9(5908I}7.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717154831111-31e7aca0-f6d8-40c0-8e60-998e35ce623d.png#averageHue=%23f4f3f2&clientId=u40004ca1-2836-4&from=paste&height=244&id=ub9f65d20&originHeight=336&originWidth=617&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=32133&status=done&style=none&taskId=uf1668a3d-19ae-4424-9e06-4bd502c61f1&title=&width=448.72727272727275)
<a name="jUP3v"></a>
### 2.1.2 配置环境
1.在终端输入命令，创建一个名为tflite的环境，并指定python版本为3.8。<br />`conda create -n tflite python=3.8`<br />![}{838HG9I9UN@@SXA9E$2B.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717154854229-64c732b2-22d5-4429-b530-f546f195ccdf.png#averageHue=%23f5f4f3&clientId=u40004ca1-2836-4&from=paste&height=132&id=u3488c042&originHeight=182&originWidth=864&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=6508&status=done&style=none&taskId=u204d2d01-e644-4485-b08e-4af0d07ab9c&title=&width=628.3636363636364)<br />![YMF2VIHPCGJT1TR_W7}G.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717154916934-ccbf7f06-0b8f-4cd4-8225-03c28a923864.png#averageHue=%23f4f2f1&clientId=u40004ca1-2836-4&from=paste&height=230&id=u8071cc31&originHeight=316&originWidth=408&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=4170&status=done&style=none&taskId=u9b4bb6f5-1c7e-4504-9151-43fc59afe5d&title=&width=296.72727272727275)<br />2.codespace是基于 Ubuntu Linux 映像创建的，输入`source activate tflite`命令激活刚刚创建的虚拟环境<br />![L~HZE9G%G0W4M%28AN@WW0.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717154937736-24c0d264-85b4-4371-8104-21d06753c7a2.png#averageHue=%23f5f3f2&clientId=u40004ca1-2836-4&from=paste&height=48&id=ueae1cbb0&originHeight=66&originWidth=756&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=3522&status=done&style=none&taskId=u9cd64c05-4237-4273-8b99-e22b5cceb3c&title=&width=549.8181818181819)<br />3.输入命令安装tflite-model-maker库.<br />`pip install tflite-model-maker`<br />![BPO.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717155001579-7ec8fa5a-b19f-4436-8718-12a493231aa6.png#averageHue=%23f1eeeb&clientId=u40004ca1-2836-4&from=paste&height=152&id=uc8898c2c&originHeight=209&originWidth=921&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=11763&status=done&style=none&taskId=uee982c15-80ae-4303-860a-2eaed6f2f38&title=&width=669.8181818181819)<br />安装完毕。<br />![9E1%87G)AEZN47YRBB`QE.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717155019537-00d2d0b7-752a-41cf-895a-597a28f7c4f3.png#averageHue=%23d1a96f&clientId=u40004ca1-2836-4&from=paste&height=63&id=ufaa266c5&originHeight=86&originWidth=403&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=3324&status=done&style=none&taskId=u069121f4-72d1-47d1-ab96-b571a7c6fbb&title=&width=293.09090909090907)<br />4.在notebook中选择在上述步骤中创建的tflite环境作为kernel<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717155136689-70053c6a-c692-4657-b0b8-c2751b96cb52.png#averageHue=%23f1f0ef&clientId=u40004ca1-2836-4&from=paste&height=105&id=uccaf8fb5&originHeight=144&originWidth=918&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=24290&status=done&style=none&taskId=u1c8a14a1-bd2e-4cf9-8680-301f8ff10d9&title=&width=667.6363636363636)
<a name="tpIoo"></a>
### 2.1.3 调试环境
输入实验需要的包，点击运行。
```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
```
产生了如下的错误。
```
ImportError: libusb-1.0.so.0: cannot open shared object file: No such file or directory
```
在终端内依次输入命令：<br />`sudo apt-get update`<br />`sudo apt-get install libusb-1.0-0-dev`<br />等待下载完成后，再次运行程序，成功运行无报错。

---

至此，准备工作全部顺利完成。
<a name="OYnnx"></a>
## 2.2 模型训练
<a name="WGP0h"></a>
### 2.2.1 获取数据
在2.1.3导入实验的依赖包之后，我们首先获取数据集合<br />从storage.googleapis.com中下载所需的数据集：
```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
```

```
Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
228818944/228813984 [==============================] - 12s 0us/step
228827136/228813984 [==============================] - 12s 0us/step
```
<a name="HKq9h"></a>
### 2.2.2 模型训练
1.加载数据集，并将数据集分为训练数据和测试数据。
```python
# 加载数据 划分数据
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

```
2024-05-31 07:02:53.715606: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/envs/tflite/lib/python3.8/site-packages/cv2/../../lib64:
2024-05-31 07:02:53.715641: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2024-05-31 07:02:53.715668: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-4bb514): /proc/driver/nvidia/version does not exist
2024-05-31 07:02:53.750504: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
```
2..训练图像分类模型
```python
# 模型训练
model = image_classifier.create(train_data)
```

```
INFO:tensorflow:Retraining the models...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
 rasLayerV1V2)                                                   
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 5)                 6405      
                                                                 
=================================================================
Total params: 3,419,429
Trainable params: 6,405
Non-trainable params: 3,413,024
_________________________________________________________________
None
Epoch 1/5


2024-05-31 07:03:44.061172: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
2024-05-31 07:03:44.446334: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
2024-05-31 07:03:44.519555: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
2024-05-31 07:03:44.581235: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 154140672 exceeds 10% of free system memory.
2024-05-31 07:03:44.666576: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 38535168 exceeds 10% of free system memory.


103/103 [==============================] - 86s 810ms/step - loss: 0.8497 - accuracy: 0.7852
Epoch 2/5
103/103 [==============================] - 82s 798ms/step - loss: 0.6541 - accuracy: 0.8923
Epoch 3/5
103/103 [==============================] - 83s 802ms/step - loss: 0.6190 - accuracy: 0.9181
Epoch 4/5
103/103 [==============================] - 83s 807ms/step - loss: 0.6036 - accuracy: 0.9238
Epoch 5/5
103/103 [==============================] - 82s 798ms/step - loss: 0.5876 - accuracy: 0.9345
```
3.评估模型
```python
# 评估模型
loss, accuracy = model.evaluate(test_data)
```

```
12/12 [==============================] - 12s 772ms/step - loss: 0.6011 - accuracy: 0.9292
```
4.在当前目录下，导出为TensorFlow lite模型
```python
# 导出模型
model.export(export_dir='.')
```

```
2024-05-31 07:16:43.402955: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.


INFO:tensorflow:Assets written to: /tmp/tmptc6htbog/assets


INFO:tensorflow:Assets written to: /tmp/tmptc6htbog/assets
2024-05-31 07:16:47.651998: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2024-05-31 07:16:47.652134: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2024-05-31 07:16:47.682406: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
  function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 17.318ms.
  function_optimizer: function_optimizer did nothing. time = 0.009ms.

/opt/conda/envs/tflite/lib/python3.8/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
  warnings.warn("Statistics for quantized inputs were expected, but not "
2024-05-31 07:16:48.244548: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
2024-05-31 07:16:48.244595: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.


INFO:tensorflow:Label file is inside the TFLite model with metadata.


fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
INFO:tensorflow:Label file is inside the TFLite model with metadata.


INFO:tensorflow:Saving labels in /tmp/tmp305imm_7/labels.txt


INFO:tensorflow:Saving labels in /tmp/tmp305imm_7/labels.txt


INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite


INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
```
![F[UFQM%BSK%I{5%BAH`@EEC.png](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717155520216-7cf2d221-20cb-4e14-bbba-2dfda8f1783c.png#averageHue=%23f0eeed&clientId=u40004ca1-2836-4&from=paste&height=108&id=ud29f72fa&originHeight=115&originWidth=293&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=2443&status=done&style=none&taskId=u6542caf6-872e-4338-844c-e6b4146fcdf&title=&width=275.0909118652344)
<a name="GCsqn"></a>
# 3 实验结果
将本次训练得到的模型导入到实验四的程序中，替换之前的成品模型。<br />![替换前](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717155569777-faff4e6c-d09c-4bf5-a54f-c3d713cd1ada.png#averageHue=%23fcfbfa&clientId=u40004ca1-2836-4&from=paste&height=109&id=ufe7c1f10&originHeight=150&originWidth=433&originalType=binary&ratio=1.375&rotation=0&showTitle=true&size=7147&status=done&style=none&taskId=u13423c74-2b13-4af9-9572-2104b9b1bbc&title=%E6%9B%BF%E6%8D%A2%E5%89%8D&width=314.90909090909093 "替换前")<br />为了不改变程序中的变量引用，我们先修改实验四的模型的名称，带上后缀“1”，然后重命名本次训练的模型名称，与实验四引用的一致。<br />![替换后（xxx1的是先前的）](https://cdn.nlark.com/yuque/0/2024/png/38674938/1717155586539-f586431b-92e7-4665-83eb-4e4d02ac05f4.png#averageHue=%23faf9f9&clientId=u40004ca1-2836-4&from=paste&height=135&id=u6510fb3c&originHeight=186&originWidth=504&originalType=binary&ratio=1.375&rotation=0&showTitle=true&size=3462&status=done&style=none&taskId=u3ed99598-5a31-4157-9692-d52f5a3ff29&title=%E6%9B%BF%E6%8D%A2%E5%90%8E%EF%BC%88xxx1%E7%9A%84%E6%98%AF%E5%85%88%E5%89%8D%E7%9A%84%EF%BC%89&width=366.54545454545456 "替换后（xxx1的是先前的）")<br />重新编译程序，然后进行调试，可以正常识别花卉。<br />
![532a72c20c36265173789ad6aeea54c4.jpg](https://cdn.nlark.com/yuque/0/2024/jpeg/38674938/1717155854026-e4335975-a6dc-4197-9305-517cb18197bd.jpeg#averageHue=%233f443d&clientId=u40004ca1-2836-4&from=paste&height=465&id=u43b4d9c0&originHeight=1920&originWidth=886&originalType=binary&ratio=1.375&rotation=0&showTitle=false&size=50000&status=done&style=none&taskId=u011b352c-672e-490c-a75b-193be2a107d&title=&width=100)
