# yolov5算法和深度摄像机

## yolov5算法框架

![](C:\Users\15710\Pictures\Screenshots\屏幕截图 2025-07-22 231134.png)

## 代码逻辑

### yolo_v5.py

**主要功能：基于YOLOv5的ROS节点，用于从ROS图像话题中接收图像数据，使用YOLOv5模型进行目标检测，并将检测结果（边界框，类别名称）通过ROS话题发布出去。同时，还显示带有检测框的图像。**

**init函数**（初始化YOLO检测器）

1. 加载各种参数，使用rospy.get_param获取ROS参数服务器的参数；

   yolov5_path：YOLOv5模型的路径；

   weight_path：YOLOv5模型的权重文件路径；

   image_topic：订阅的图像话题；

   pub_topic：发布检测结果的话题；

   self.camera_frame：相机的坐标系名称；

   conf：置信度阈值。

2. 使用torch.hub.load从本地加载YOLOv5模型；

   ```python
   self.model = torch.hub.load(yolov5_path, 'custom',path=weight_path, source='local')
   ```

3. 根据参数/use_cpu选择使用CPU还是GPU运行模型，如果/use_cpu为true，则使用CPU，否则使用GPU；

   ```python
   if (rospy.get_param('/use_cpu', 'false')):
       self.model.cpu()
   else:
       self.model.cuda()
   ```

   

4. 设置置信度，初始化图像和状态，加载类别颜色，订阅图像话题；

5. 创建三个ROS发布者，self.position_pub：发布检测到的目标框信息；self.image_pub：发布带有检测框的图像；self.class_pub：发布检测到的类别信息；

   ```python
   self.position_pub = rospy.Publisher(pub_topic,  BoundingBoxes, queue_size=1)
   self.image_pub = rospy.Publisher('/yolov5/detection_image',  Image, queue_size=1)
   self.class_pub = rospy.Publisher('/yolov5/detected_classes',  String, queue_size=1) 
   ```

   

6. 等待图像消息，使用rospy.loginfo输出日志信息，提示正在等待图像。

**image_callback函数**(接收图像消息时被调用)

1. 初始化检测结果消息，设置消息头，设置获取图像状态；

2. 使用np.frombuffer将图像消息的二进制数据image.data转换为Numpy数组，数据类型为八位无符号整数，reshape将一维的二进制数据重新调整为图像的高度、宽度和通道数；

   ```python
   self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height,image.width, -1)
   ```

3. 调整颜色间，从BGR转换为RGB，运用模型进行检测，results为包含模型的检测结果，boxes提取检测结果（坐标、置信度和类别消息）；

   ```python
   self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
   results = self.model(self.color_image)
   ```

4. 调用self.dectshow函数，传入原始图像、检测结果以及图像的高度和宽度，在图像上绘制检测框，并发布带有检测框的图像。

```python
boxs = results.pandas().xyxy[0].values
self.dectshow(self.color_image, boxs, image.height, image.width)
```

**dectshow函数**（处理检测结果并在图像上绘制检测框）

1. 创建原始图像副本img，初始化检测类别列表；

2. 遍历每个目标框，创建边界框消息，选择或生成颜色，如果类别在self.classes_colors字典中，则直接使用此颜色，否则随机生成一个颜色，并存储到字典中；

3. 绘制边界框，并调整文本位置，运用 cv2.putText绘制类别名称；

   ```python
   # 在图像上绘制边界框
   cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])),(int(color[0]),int(color[1]), int(color[2])), 2) 
   # 根据边界框的y坐标位置，调整类别名称文本的显示位置
   	if box[1] < 20: 
       	text_pos_y = box[1] + 30
       else:
       	text_pos_y = box[1] - 10
   # 在图像上绘制类别名称文档
   cv2.putText(img, box[-1]，(int(box[0]),int(text_pos_y)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
   
   ```

   

4. 发布边界框消息、检测到的类别列表和处理后的图像。

**publish_image函数**(发布图像)

- 创建一个实例image_temp，生成Header实例，将相机坐标系设置为self.camera_frame；

- 将图像数据转换为字节形式，并赋值给image_temp.data，将生成的header赋值给image_temp.header，使用self.image_pub.publish(image_temp)将图像消息发布出去。

**main函数**

初始化一个名为yolov5_ros的ROS节点，每次启动节点时，ROS会自动为节点分配一个唯一的名称；创建实例yolo_dact；调用rospy.spin()，进入ROS事件循环，等待接收图像消息并进行处理。

### common.py

**主要功能：是一个用于目标检测的强大深度学习模型，主要涉及模型的构建、不同后端的推理（如pytorch、torchscript、onnx、openvino等）以及推理结果的处理（如非极大值抑制、坐标调整、显示和保存检测结果等）。**

1. **autopad函数**：用于计算卷积层的填充大小，使得输出的尺寸与输入的尺寸相同。如果提供了填充大小p，则直接使用；否则，根据卷积核大小k自动计算填充大小。

2. **Conv类（卷积层）**：定义了一个标准的卷积层结构，包括卷积操作、批归一化操作和激活函数。其中，forward实现了前向传播过程，forward_fuse则是在融合了BN和卷积层后的前向传播过程。

3. **DWConv类（深度可分离卷积）**：这是深度可分离卷积类，首先进行深度方向上的卷积，然后再进行点卷积。关键特点：每个输入通道只与一个输出通道进行卷积操作，从而减少计算量。

4. **TransformerLayer类**：这个类定义了一个Transformer层，但是不包含LayerNorm层，以提高性能。Transformer层的核心是多头自注意力机制，通过查询（q）、键（k）和值（v）来计算特征之间的相关性，并对原始特征进行加和和线性变换。其中，self.ma: 多头自注意力层。self.fc1, self.fc2: 定义两个全连接（线性变换）层。

5. **TransformerBlock类**：这个类定义了一个Vision Transformer模块，包含一个可选的卷积层(self.conv)、一个线性层(self.linear)以及多个TransformerLayer(self.tr)。首先通过卷积层调整通道数，然后通过线性层学习位置嵌入，接着通过多个TransformerLayer处理特征图。

6. **Bottleneck类**：标准残差单元，包含两个卷积层（cv1和cv2），其中一个使用1x1卷积进行降维，另一个使用3x3卷积进行升维。如果shortcut为True且输入输出通道数相同，则残差连接会将原始输入与经过两个卷积层处理的输出相加。

7. **BottleneckCSP类**：对残差单元进行了改进，通过两条路径处理特征图。一条路径通过多个残差单元，另一条路径直接传递特征图。最后将两条路径的特征图进行连接并应用BatchNorm和激活函数(self.act)。

8. **C3类**：包含三个卷积层的CSP残差单元，结构与BottleneckCSP类似，区别在于内部残差单元的个数和输出特征图的处理方式不同。

9. **C3TR类**：在C3的基础上增加了TransformerBlock层，用于引入注意力机制，增强模型对特征图中长距离依赖的捕捉能力。

   C3SPP类：在C3的基础上增加了SPP层，用于不同尺度上的特征池化，提高模型的多尺度特征表达能力。

10. **C3Ghost类**：在C3的基础上使用了GhostBottleneck层，这种层可以在不增加计算量的前提下增加特征图的通道数。

11. **SPP和SPPF类**：定义了两种不同版本的空间金字塔池化层，在YOLOv5模型中用于增强特征图的多尺度特征表达能力。SPP层通过不同大小的池化核对特征图进行池化，然后将结果拼接起来；SPPF层则是通过多次池化和拼接来实现类似的效果，但计算效率更高。

12. **Focus类**：用于将宽高信息转化为通道信息，实现特征图分辨率降低的同时增加通道数。具体做法是将输入特征图分成四个部分，然后将这四个部分在通道维度上拼接起来。

13. **GhostConv和GhostBottleneck类**：分别定义了Ghost卷积层和Ghost残差单元。Ghost卷积通过引入一个额外的1x1卷积层来生成隐藏通道，然后通过一个5x5卷积层生成输出通道。Ghost残差单元则是在残差单元中引入了Ghost卷积。

14. **Contract和Expand类**：用于在通道维度上进行压缩或扩展。Contract类压缩宽高信息到通道中，而Expand类则进行相反的操作，将通道信息扩展到宽高中。

15. **DetectMultiBackend类**：这个类用于在不同后端进行YOLOv5模型的推理。它支持多种推理后端，包括PyTorch、TorchScript、ONNX、OpenVINO、TensorRT和TensorFlow等。不同后端的加载和推理方式都有所不同，代码中根据权重的文件后缀来判断使用哪种后端，并进行相应的初始化和推理设置。这个类的设计使得YOLOv5模型可以轻松地在不同的推理环境中运行。

16. **AutoShape 类**：用于处理不同格式的输入图像，并自动进行预处理、推理和非极大值抑制（NMS）。

17. **Detections 类**：用于封装推理结果，提供了一个方便的方式来管理和访问检测结果。

### experimental.py

**主要功能：定义了一些用于YOLOv5目标检测模型的实验性模块，这些模块主要用于模型的不同层，以实现特定的功能，如跨卷积、加权求和、混合深度卷积和模型集成。**

1. **CrossConv**：这是一个**交叉卷积下采样**的模块。它有两个卷积操作：cv1 和 cv2。cv1 使用一个1×k的卷积核，cv2 使用一个k×1的卷积核，并且可以进行组卷积（groups=g）。如果输入和输出通道数相同（shortcut=True），则在两个卷积操作的结果上加上输入本身，否则直接返回两个卷积操作的结果。
2. **Sum**：这是一个**加权求和**的模块，可以对两个或更多的输入进行加权求和。构造函数接受两个参数：n（输入的数量）和weight（是否使用权重）。如果weight=True，则会为每个输入分配一个权重，并且这些权重会在前向传播时被应用。如果weight=False，则简单的对所有输入求和。
3. **MixConv2d**：这是一个**混合深度卷积**的模块。它可以接受多个不同大小的卷积核进行并行操作，然后将结果在通道维度上进行拼接。k是一个包含多个卷积核大小的列表，equal_ch参数决定每个卷积核的中间通道数是否相等。根据equal_ch的设置，模块会计算每个卷积核的中间通道数。在前向传播时，模块会将每个卷积核的结果在通道维度上进行拼接，然后通过批归一化和激活函数进行处理。
4. **Ensemble**：这是一个**模型集成**的模块。它可以加载多个模型的权重，并将这些模型集成在一起。forward方法接受输入x，并且可以启用额外的功能如数据增强（augment）、性能分析（profile）和可视化（visualize）。集成的方式是将每个模型的输出在通道维度进行拼接（nms ensemble），然后返回拼接后的输出和None。
5. **attempt_load**：这是一个**尝试加载模型权重**的函数。它接受一个或多个权重文件路径（weights），并且可以选择性地进行层融合（fuse）。对于每个权重文件，函数都会加载模型，并根据fuse参数决定是否进行层融合。加载的模型会被添加到Ensemble对象中，然后对整个集成模型进行兼容性更新。如果集成模型只有一个子模型，则直接返回该模型；否则返回集成模型。在集成模型的情况下，还会设置一些共享的属性如names和stride。

### yolo.py

**主要功能：定义了YOLOv5模型的特定模块和解析函数，用于模型的构建、前向传播、参数初始化、模型融合等操作，是YOLOv5模型的核心实现之一。**

1. Detect类： 这是一个检测层类，负责处理模型的最终输出。它通过卷积层对特征图进行处理，生成检测结果。在推理阶段，它还负责将特征图中的预测结果（边界框坐标、置信度等）转换为实际的预测值。

2. Model类： 这是YOLOv5模型的主要类，包含了模型的构建、前向传播、参数初始化等功能。它首先解析配置文件，然后根据配置文件中的定义构建模型的各层。在模型构建完成后，它还会计算并设置模型各层的步长（stride）和锚点（anchors），并初始化模型的权重和偏置。

   前向传播：

   - forward 是模型的前向传播方法，根据不同的模式（训练或推理）调用不同的辅助方法。

   - _forward_once 用于处理单尺度输入，包括从前面的层获取输入、计算当前层的输出、在必要时保存输出等。

   - _forward_augment 用于处理多尺度输入，通过不同尺度的输入来增强模型的稳健性。

   辅助函数：

   - _initialize_biases 用于初始化检测层的偏置。

   - _print_biases 用于打印检测层的偏置信息。

   - fuse 用于融合模型中的卷积层和批量归一化层（BatchNorm2d），以提高模型的推理效率。

   - info 用于打印模型的信息，包括层数、类型、参数数量等。

3. parse_model函数： 这个函数用于解析模型配置文件，并根据配置文件构建模型的各层。它会处理每一层的参数，并生成一个包含所有层的nn.Sequential对象。

4. 主程序： 文件末尾的主程序部分用于创建并测试YOLOv5模型。通过命令行参数指定模型的配置文件和设备，然后创建模型实例并根据参数选择是否进行模型速度的评估或测试所有预定义的模型配置文件。

### hubconf.py

**主要功能：包含了YOLOv5的使用说明，展示了如何通过 PyTorch Hub 加载不同的YOLOv5模型，包括预训练模型和自定义模型。**

1. _create函数：内部函数，用于创建指定的YOLOv5模型。

   ```python
   def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
       # 导入必要的库
       from pathlib import Path
       from models.common import AutoShape, DetectMultiBackend
       from models.yolo import Model
       from utils.downloads import attempt_download
       from utils.general import check_requirements, intersect_dicts, set_logging
       from utils.torch_utils import select_device
       check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))# 检查并安装所需的库
       set_logging(verbose=verbose)# 设置日志输出
       
       name = Path(name)
       path = name.with_suffix('.pt') if name.suffix == '' else name  # checkpoint path
       try:# 尝试选择设备，优先选择GPU，否则使用CPU
           device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)
   	# 如果加载预训练模型且通道数和类别数符合默认值，使用DetectMultiBackend加载模型。
   	# 否则，创建一个新的YOLOv5模型实例，并根据需要加载预训练权重。
   	# 如果加载了预训练权重，则将模型的类别名称设置为预训练模型的类别名称。
           if pretrained and channels == 3 and classes == 80:
               model = DetectMultiBackend(path, device=device)  # download/load FP32 model
               # model = models.experimental.attempt_load(path, map_location=device)  # download/load FP32 model
           else:
               cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
               model = Model(cfg, channels, classes)  # create model
               if pretrained:
                   ckpt = torch.load(attempt_download(path), map_location=device)  # load
                   csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                   csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                   model.load_state_dict(csd, strict=False)  # load
                   if len(ckpt['model'].names) == classes:
                       model.names = ckpt['model'].names  # set class names attribute
           if autoshape:
               model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
           return model.to(device)
   
       except Exception as e:
           help_url = 'https://github.com/ultralytics/yolov5/issues/36'
           s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
           raise Exception(s) from e
   ```

2. 定义各个加载函数功能

   custom 函数:：加载自定义或本地的YOLOv5模型。

   yolov5n 函数：加载YOLOv5-nano模型。

   yolov5s 函数：加载YOLOv5-small模型。

   yolov5m 函数：加载YOLOv5-medium模型。

   yolov5l 函数：加载YOLOv5-large模型。

   yolov5x 函数：加载YOLOv5-xlarge模型。

   yolov5n6 函数：加载YOLOv5-nano-P6模型。

   yolov5s6 函数：加载YOLOv5-small-P6模型。

   yolov5m6 函数：加载YOLOv5-medium-P6模型。

   yolov5l6 函数：加载YOLOv5-large-P6模型。

   yolov5x6 函数：加载YOLOv5-xlarge-P6模型。

3. 主程序（用于验证模型的推理能力）

   ```python
   if __name__ == '__main__':
       # 加载预训练的yolov5s模型
       model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True) 
       # 导入必要的库
   	from pathlib import Path
       import cv2
       import numpy as np
       from PIL import Image
   	# 创建一个包含不同输入类型的图像列表imgs
       imgs = ['data/images/zidane.jpg',  # filename
               Path('data/images/zidane.jpg'),  # Path
               'https://ultralytics.com/images/zidane.jpg',  # URI
               cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
               Image.open('data/images/bus.jpg'),  # PIL
               np.zeros((320, 640, 3))]  # numpy
   	# 用模型对图像列表进行批量推理，并打印和保存结果
       results = model(imgs, size=320)  # batched inference
       results.print()
       results.save()
   ```

   

### val.py

**主要功能：验证YOLOv5模型准确性的代码，支持多种模型格式（如PyTorch, TorchScript, ONNX, OpenVINO等）。通过计算P (Precision), R (Recall), mAP@0.5, mAP@0.5:0.95等指标来评估模型性能。**

1. **save_one_txt 函数**：该函数用于将单个图像的预测结果保存为txt文件。

   函数将每个预测结果的类别、归一化后的边界框坐标（xywh格式）和置信度（save_conf 为真）写入指定的txt文件。

2. **save_one_json 函数**：该函数用于将单个图像的预测结果保存为JSON格式。

   函数将每个预测结果转换为COCO数据集格式的JSON对象，并添加到 jdict 列表中。

3. **process_batch 函数**：该函数用于评估一批图像的预测结果与真实标签的正确性。

   函数计算预测边界框与真实标签之间的IoU，并返回一个布尔矩阵 correct，其中每个元素表示预测边界框是否在不同IoU阈值下的正确性。该函数通过将预测结果与真实标签进行匹配，并根据IoU阈值进行排序和去重来评估预测的准确性。

4. **run 函数**：YOLOv5模型验证的核心函数，用于处理数据、进行模型推理、评估预测结果，并保存结果。
   （1）初始化/加载模型和设置设备，将模型设置为评估模式，检查是否为COCO数据集，获取类别数量、IoU阈值向量等，创建数据加载器，进行数据预处理；
   （2）遍历数据加载器中的每个批次，将图像和标签移动到指定设备，进行模型推理和NMS，计算损失（如果需要），评估预测结果与真实标签的匹配情况；
   （3）生成混淆矩阵和其他统计信息，计算各类别的精度（P）、召回率（R）、F1分数、平均精度（mAP）等，打印整体和各类别的评估结果；
   （4）生成并保存验证图像的预测结果图，打印推理速度等信息，如果需要，将预测结果保存为JSON文件，并使用 pycocotools 计算mAP。

5. **parse_opt 函数**：用于解析命令行参数，配置验证任务所需的参数。
   （1）使用 argparse.ArgumentParser 创建一个参数解析器对象；
   （2）添加各种参数，解析命令行参数并存储在 opt 对象中，检查数据集配置文件是否为有效的YAML文件，根据数据集配置文件和命令行参数设置 save_json 和 save_txt；
   （3）打印解析后的参数信息，返回解析后的参数对象 opt。

6. **main 函数**：是验证任务的主函数，负责根据命令行参数执行不同的任务。

   （1）使用 check_requirements 函数检查并安装所需的依赖项；

   （2）根据 opt.task 的值执行不同的操作。

   正常任务（‘train’, ‘val’, ‘test’）：如果置信度阈值大于0.0001，打印警告信息，调用run函数进行验证，并传递解析后的参数。

   速度基准测试（‘speed’）：设置置信度阈值为 0.25，IoU阈值为0.45，并禁用保存JSON文件。遍历每个模型权重文件，调用 run 函数进行速度基准测试，并禁用绘图。

   速度与mAP基准测试（‘study’）：设置置信度阈值为 0.25，IoU阈值为 0.7。遍历每个模型权重文件，遍历不同的图像大小，调用 run 函数进行验证，并将结果和时间信息添加到列表 y 中，将结果和时间信息保存到指定的txt文件中。使用 os.system 命令将所有生成的txt文件压缩成一个zip文件，调用 plot_val_study 函数绘制速度与mAP的关系图。

### tf.py

**主要功能：将Yolov5模型从PyTorch框架转换为TensorFlow、Keras和TFLite版本，支持模型的推理和导出。通过定义多个类，实现了Yolov5模型中各个层在TensorFlow框架中的等效功能。**

1. **自定义TensorFlow层**

   **TFBN**：BatchNormalization层的封装，用于处理批归一化操作。
   **TFPad**：自定义的填充层，用于对输入张量进行常量填充。
   **TFConv**：标准卷积层的封装，支持不同的激活函数，并且处理了PyTorch和TensorFlow之间不同的填充方式。
   **TFFocus**：一个用于将空间信息（宽高）压缩到通道中的层。
   **TFBottleneck**：实现了YOLOv5中的标准瓶颈层，用于减少参数量和计算量。
   **TFConv2d**：这是对PyTorch的nn.Conv2d层的TensorFlow实现。
   **TFBottleneckCSP**：实现了YOLOv5中的CSP  Bottleneck层，结合了瓶颈层和CSP结构，可以更好地利用不同尺度的信息。
   **TFC3**：实现了具有3个卷积的CSP瓶颈结构。
   **TFSPP**：实现空间金字塔池化层，用于YOLOv3-SPP模型。
   **TFSPPF**：实现空间金字塔池化—快速层。
   **TFDetect**：实现YOLOv5的检测层，用于生成检测框、置信度和类别预测，并在推理时应用非极大值抑制。
   **TFUpsample**：实现了上采样层，用于调整图像的大小。
   **TFConcat**：实现了张量的连接操作。

2. **parse_model函数**：用于解析YOLOv5模型结构并构建相应TensorFlow模型。

   - 日志初始化：使用LOGGER.info打印表头，包含每个层的索引、重复次数、参数数量、模块类型和参数。

   ```python
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
   ```

   - 参数提取：从配置字典中提取锚点、类别数、深度倍增因子和宽度倍增因子。

     ```python
     anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors # 计算每个检测层的锚点数量
     no = na * (nc + 5)  #  计算每个检测层的输出数量
     ```

   - 模型构建

     ```python
     layers, save, c2 = [], [], ch[-1]  
     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):# 遍历模型的骨干网络和头部网络部分
         m_str = m
         m = eval(m) if isinstance(m, str) else m  # eval strings
         # 如果模块类型是字符串，使用eval函数将其转换为对应的类对象
         for j, a in enumerate(args):
             try:
                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
             except NameError:
                 pass
             
     	n = max(round(n * gd), 1) if n > 1 else n  # depth gain
         # 对于卷积层模块，调整通道数并确保与配置一致
         if m in [nn.Conv2d, Conv, Bottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
             c1, c2 = ch[f], args[0]
             c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
     	    args = [c1, c2, *args[1:]]
             # 对于BottleneckCSP和C3模块，将重复次数插入参数列表中
             if m in [BottleneckCSP, C3]:
                 args.insert(2, n)
                 n = 1
             # 对于nn.BatchNorm2d模块，参数为输入通道数
             elif m is nn.BatchNorm2d:
                 args = [ch[f]]
             # 对于Concat模块，计算输出通道数为所有输入通道数之和
             elif m is Concat:
                 c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
             # 对于Detect模块，处理锚点参数和类别数，并将图像尺寸添加到参数列表中
             elif m is Detect:
                 args.append([ch[x + 1] for x in f])
                 if isinstance(args[1], int):  # number of anchors
                     args[1] = [list(range(args[1] * 2))] * len(f)
                 args.append(imgsz)
             # 对于其他模块，输出通道数与输入通道数相同
             else:
                 c2 = ch[f]
     ```

   - 对于其他模块，输出通道数与输入通道数相同

     ```python
     tf_m = eval('TF' + m_str.replace('nn.', ''))
     # 如果重复次数大于1，使用keras.Sequential创建一个包含多次重复的层。否则，直接创建单个层。
     m_ = keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)]) if n > 1  
     	else tf_m(*args, w=model.model[i])  
     ```

     - 记录每个层的类型、参数数量、索引和输入层索引;将需要保存的层索引添加到savelist中;将构建的层添加到layers列表中;更新输出通道数c2并将其添加到ch列表中。

3. **TFModel类**：将YOLOv5模型从PyTorch框架转换为TensorFlow或Keras框架，并提供预测功能。

   - **init函数**：初始化self.yaml属性；如果传入的参数nc与配置文件中的nc不同，则会记录并覆盖配置文件中的类别数；使用parse_model函数解析模型配置字典，生成TensorFlow或Keras模型及其保存列表，并保存到self.model和self.savelist属性中。

   - **predict函数**：用于执行模型预测。

     ```python
     def predict(self, inputs, tf_nms=False, agnostic_nms=False, topk_per_class=100,topk_all=100, iou_thres=0.45,conf_thres=0.25):
         y = []  # outputs
         x = inputs
         # 遍历模型的每一层：如果当前层m的f属性不为-1，说明当前层的输入不是来自前一层，而是来自savelist中指定的层。根据f的值从savelist中获取输入。
         for i, m in enumerate(self.model.layers):
             if m.f != -1:  # if not from previous layer
                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  
             x = m(x)  # run
             y.append(x if m.i in self.savelist else None)  
             if tf_nms:
                 boxes = self._xywh2xyxy(x[0][..., :4])
                 probs = x[0][:, :, 4:5]
                 classes = x[0][:, :, 5:]
                 scores = probs * classes
                 if agnostic_nms:
                     nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
                     return nms, x[1]
                 else:
                     boxes = tf.expand_dims(boxes, 2)
                     nms = tf.image.combined_non_max_suppression(
                         boxes, scores, topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes=False)
                     return nms, x[1]
             return x[0]
     ```

4. **AgnosticNMS类**：用于在对象检测任务中进行非极大值抑制（ NMS）操作，实现非最大抑制层，去除重复检测框，提高检测结果的质量。

   - **call函数**：Keras层中用于定义前向传播的方法。这里使用tf.map_fn对输入的每个样本执行nms方法。tf.map_fn可以看作是张量的map函数，它会将输入的每一个元素（在这个例子中，每一个样本的预测结果）传递给提供的函数（这里是self.nms），并返回结果。

     ```python
     def call(self, input, topk_all, iou_thres, conf_thres):
         return tf.map_fn(lambda x: self._nms(x, topk_all, iou_thres, conf_thres), 					input,fn_output_signature=(tf.float32, tf.float32, tf.float32, 							tf.int32),name='agnostic_nms')
     ```

     - **_nms函数**：一个静态方法，这意味着它不依赖于类的实例状态，可以直接通过类名调用。该方法接收预测的边界框（boxes）、类别概率（classes）和类别分数（scores），并根据置信度和IOU阈值执行非极大值抑制操作。

       ```python
       @staticmethod
       def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):  
           boxes, classes, scores = x
           class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32) # 类别索引提取
           scores_inp = tf.reduce_max(scores, -1) # 置信度提取
           # 非极大值抑制
           selected_inds = tf.image.non_max_suppression(boxes, scores_inp, 							max_output_size=topk_all, iou_threshold=iou_thres, score_threshold=conf_thres)
           # 根据索引提取(selected_)和填充（padded_)边界框、置信度和类别
           selected_boxes = tf.gather(boxes, selected_inds) 
           padded_boxes = tf.pad(selected_boxes,paddings=[[0, topk_all - tf.shape(selected_boxes)			[0]], [0, 0]], mode="CONSTANT", constant_values=0.0)
           selected_scores = tf.gather(scores_inp, selected_inds)
           padded_scores = tf.pad(selected_scores,paddings=[[0, 
                                  topk_all -tf.shape(selected_boxes)[0]]],
                                  mode="CONSTANT", constant_values=-1.0)
           selected_classes = tf.gather(class_inds, selected_inds)
           padded_classes = tf.pad(selected_classes,
                            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                            mode="CONSTANT", constant_values=-1.0)
           valid_detections = tf.shape(selected_inds)[0] # 有效检测数量
           return padded_boxes, padded_scores, padded_classes, valid_detections
       ```

## 运行问题解答

1. **启动ROS RealSense节点时遇到节点加载失败。**

   原因：

   ①realsense2_camera包未正确编译或安装，导致nodelet插件未生成或未被ROS识别；

   ②ROS环境未正确source，导致插件路径未加载；

   ③依赖库缺失或版本不匹配；

   ④工作空间构建方式不正确，未包含realsense2_camera的nodelet插件。

2. **缺少Python模块catkin_pkg，导致importError。**

   解决：在conda环境myenv中安装catkin_pkg模块。

3. **empy版本不兼容。**

   报错信息：AttributeError: module "em' has no attribute"RAW_OPT'。

   解决：降级empy版本到一个兼容的版本，比如3.3.4。

4. **yolov5_ros启动时，torch.load加载权重文件失败。**

   原因：torch.load没有传入weights_only参数，默认在PyTorch 2.6中为True，导致加载失败。

   解决：修改torch.load调用，显式传入weights_only=False参数，绕过默认限制。

5. **节点在构造函数中启动订阅后，一直打印 "waiting for image."，没收到图像消息。**

   原因：

   ①话题名不匹配，节点订阅的话题和摄像头发布的话题不一致；

   ②摄像头节点没有正确发布图像；

   ③ROS网络配置问题，节点无法接收到图像消息。

   解决：

   ①查看 yolo_v5.launch 文件，确认传入的参数 "~image_topic" 是否正确；

   ②确认摄像头节点发布的图像话题名称；

   ③确认 rosnode 和 rostopic 状态，确保图像话题存在且有消息发布。

6. **终端无法连接到ROS Master，导致yolov5_ros节点一直等待图像消息，节点无法正常通信。**

   报错信息：ERROR: Unable to communicate with master！

   解决：

   ①确认ROS Master是否启动；

   ②可能执行rostopic命令的终端没有正确source ROS环境，或者ROS Master没有启动。

7. **执行roslaunch yolov5_ros yolo_v5.launch后，/usb_cam/image_raw话题并未出现，说明摄像头节点没有启动或没有发布图像。**

原因：

①摄像头驱动节点未启动；

②摄像头设备未正确连接或驱动异常；

③启动文件中未包含摄像头节点启动。