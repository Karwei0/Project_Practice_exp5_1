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


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228818944/228813984 [==============================] - 12s 0us/step
    228827136/228813984 [==============================] - 12s 0us/step
    


```python
# 加载数据 划分数据
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

    2024-05-31 07:02:53.715606: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/envs/tflite/lib/python3.8/site-packages/cv2/../../lib64:
    2024-05-31 07:02:53.715641: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2024-05-31 07:02:53.715668: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-4bb514): /proc/driver/nvidia/version does not exist
    2024-05-31 07:02:53.750504: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    


```python
# 模型训练
model = image_classifier.create(train_data)
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
    


```python
# 评估模型
loss, accuracy = model.evaluate(test_data)
```

    12/12 [==============================] - 12s 772ms/step - loss: 0.6011 - accuracy: 0.9292
    


```python
# 导出模型
model.export(export_dir='.')
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
    
