# Summary

* [前言](README.md)
* [一、TensorFlow的建模流程](一、TensorFlow的建模流程.md)
* [1-1,结构化数据建模流程范例](./1-1,结构化数据建模流程范例.md)
- - 数据探索： `pandas` 技巧，如 `query(), plot(), get_dummies()` ；
- - Keras 基本使用：训练、使用、保存、加载模型；
- - Keras 中使用Sequential按层顺序构建模型。
* [1-2,图片数据建模流程范例](./1-2,图片数据建模流程范例.md)
- - 使用tf.data.Dataset搭配tf.image中的一些图片处理方法构建数据管道；
- - Keras 中使用函数式API构建任意结构模型。
* [1-3,文本数据建模流程范例](./1-3,文本数据建模流程范例.md)
- - 使用tf.data.Dataset搭配.keras.layers.experimental.preprocessing.TextVectorization预处理层；
- - Keras 中继承Model基类构建自定义模型。
* [1-4,时间序列数据建模流程范例](./1-4,时间序列数据建模流程范例.md)
- - Keras 中使用函数式API构建任意结构模型。

第一章总的来说，是做了一些展示，里面涉及到的知识将在以后课程中展开，而数据预处理，相信要具体问题具体学习。

* [二、TensorFlow的核心概念](二、TensorFlow的核心概念.md)
* [2-1,张量数据结构](./2-1,张量数据结构.md)
- - 常量与变量；
- - 常量的重新赋值相当于创造新的内存空间，变量的值可以改变。
* [2-2,三种计算图](./2-2,三种计算图.md)
- - 静态图（tf 1.x）速度快，但是写起来复杂；
- - 动态图（tf 2.x & PyTorch）语法像 `python` 一样自然，但速度慢；
- - tf 2.x 的 Autograph 试图结合上述两者优点。
* [2-3,自动微分机制](./2-3,自动微分机制.md)
- - 磁带 tape 与 autograph ；
- - 举了使用梯度下降求函数最小值的例子。

第二章的知识点其实不多，但是很大：作者用简单的语言解释了“动态图”等等机制，这其实是很核心的概念，相信其背后的实现难度、考虑其实是很复杂的。但是对于初学者/应用者来讲，无需了解很多。

基础的几个概念了解了，应该投身于高质量的自主实验上。这是我总结的 CS 类知识学习经验。机理、核心概念、语法有所了解，接下来的知识点无法再靠阅读来获取，必须投身实践，自己探索、试错，并形成自己的代码风格。

* [三、TensorFlow的层次结构](三、TensorFlow的层次结构.md)
* [3-1,低阶API示范](./3-1,低阶API示范.md)
- - 以正向传播求损失举例；
- - 仅仅在函数声明前加上 @tf.function ，“使用autograph机制转换成静态图加速”，速度就提升了很多倍。
* [3-2,中阶API示范](./3-2,中阶API示范.md)
- - 用数据管道 `tf.data.Dataset.from_tensor_slices((X,Y))` 举了例子。
* [3-3,高阶API示范](./3-3,高阶API示范.md)
- - 展示了套用 keras 的操作；
- - 如 MyModel(models.Model) 、 tf.keras.metrics 等。

第三章也是示范篇章，了解一个方案的程序构造，具体内容将在之后三章展开。

* [四、TensorFlow的低阶API](四、TensorFlow的低阶API.md)
* [4-1,张量的结构操作](./4-1,张量的结构操作.md)
- - tf.slice(t,[1,0],[3,5])) #tf.slice(input,begin_vector,size_vector)
- - tf.print(t[1:4,:4:2]) #第1行至最后一行，第0列到最后一列每隔两列取一列
- - 对于变量，assign 修改元素
- - p = tf.gather(scores,[0,5,9],axis=1) #抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
- - tf.print(a[...,1]) #省略号可以表示多个冒号
- - s = tf.gather_nd(scores,indices = [(0,0),(2,4),(3,6)]) #抽取第0个班级第0个学生，第2个班级的第4个学生，第3个班级的第6个学生的全部成绩
- - p = tf.boolean_mask(scores,\[True,False,False,False,False,True,False,False,False,True\],axis=1) #抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
- - tf.print(tf.boolean_mask(c,c<0),"\n") 
- - tf.print(c[c<0]) #布尔索引，为boolean_mask的语法糖形式
- - d = tf.where(c<0,tf.fill(c.shape,np.nan),c)
- - 维度变换相关函数主要有 tf.reshape, tf.squeeze, tf.expand_dims, tf.transpose.
- - 和numpy类似，可以用tf.concat和tf.stack方法对多个张量进行合并，可以用tf.split方法把一个张量分割成多个张量。
* [4-2,张量的数学运算](./4-2,张量的数学运算.md)
- - 向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。 许多向量运算符都以reduce开头。
- - values,indices = tf.math.top_k(a,3,sorted=True) #tf.math.top_k可以用于对张量排序。
- - 除了一些常用的运算外，大部分和矩阵有关的运算都在tf.linalg子包中；
- - a@b  #等价于tf.matmul(a,b)；
- - TensorFlow的广播规则和numpy是一样的。
* [4-3,AutoGraph的使用规范](./4-3,AutoGraph的使用规范.md)
- - 被@tf.function修饰的函数应尽量使用TensorFlow中的函数而不是Python中的其他函数；
- - 避免在@tf.function修饰的函数内部定义tf.Variable；
- - 被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。
* [4-4,AutoGraph的机制原理](./4-4,AutoGraph的机制原理.md)
- - 先建立静态图，然后运行。
* [4-5,AutoGraph和tf.Module](./4-5,AutoGraph和tf.Module.md)
- - 将相关的tf.Variable创建放在类的初始化方法中。而将函数的逻辑放在其他方法中。
- - 利用tf.Module提供的封装，再结合TensoFlow丰富的低阶API，实际上我们能够基于TensorFlow开发任意机器学习模型(而非仅仅是神经网络模型)，并实现跨平台部署使用。

本章内容充实，开始时，讲了张量运算的相关内容，举了很多典型的例子，我记录在这里了。实践时，应多使用已有方法。

AutoGraph 的规范、原理相互呼应，很好理解“为什么如此定义规范”。

基于 tf.Module 构造方法计算类，很巧妙。可以在 `__init__(self)` 中声明变量。

* [五、TensorFlow的中阶API](五、TensorFlow的中阶API.md)
* [5-1,数据管道Dataset](./5-1,数据管道Dataset.md)
- - 为了处理大量数据，提出数据管道；
- - 可以从 Numpy array, Pandas DataFrame, Python generator, csv文件, 文本文件, 文件路径, tfrecords文件等方式构建数据管道。
- - Dataset数据结构应用非常灵活，因为它本质上是一个Sequece序列，其每个元素可以是各种类型，例如可以是张量，列表，字典，也可以是Dataset;
- - Dataset包含了非常丰富的数据转换功能。
- - 此外，本节还介绍了构建高效数据管道的建议，如使用prefetch、cache及各种方法组合等。
* [5-2,特征列feature_column](./5-2,特征列feature_column.md)
- - 要创建特征列，请调用 tf.feature_column 模块的函数；
- - 举了泰坦尼克号的例子，有很好的数据预处理规范（应用`pandas`）;
- - 定义特征列并不是处理数据，而是作为 tf.keras.layers.DenseFeatures 的第一层。
* [5-3,激活函数activation](./5-3,激活函数activation.md)
- - 除了通过activation参数指定激活函数以外；
- - 还可以显式添加layers.Activation激活层。
* [5-4,模型层layers](./5-4,模型层layers.md)
- - tf.keras.layers内置了非常丰富的各种功能的模型层；
- - 讲了一个自定义模型层的简单例子。
* [5-5,损失函数losses](./5-5,损失函数losses.md)
- - 监督学习的目标函数由损失函数和正则化项组成（Objective = Loss + Regularization）;
- - 对于keras模型，目标函数中的正则化项一般在各层中指定.
* [5-6,评估指标metrics](./5-6,评估指标metrics.md)
- - 通常损失函数都可以作为评估指标，如MAE,MSE,CategoricalCrossentropy等也是常用的评估指标;
- - 但评估指标不一定可以作为损失函数，例如AUC,Accuracy,Precision。因为评估指标不要求连续可导，而损失函数通常要求连续可导。
* [5-7,优化器optimizers](./5-7,优化器optimizers.md)
- - 深度学习优化算法大概经历了 SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam 这样的发展历程;
- - 对于一般新手炼丹师，优化器直接使用Adam，并使用其默认参数就OK了;
- - 一些爱写论文的炼丹师由于追求评估指标效果，可能会偏爱前期使用Adam优化器快速下降，后期使用SGD并精调优化器参数得到更好的结果。
* [5-8,回调函数callbacks](./5-8,回调函数callbacks.md)
- - tf.keras的回调函数实际上是一个类，一般是在model.fit时作为参数指定；
- - 用于控制在训练过程开始或者在训练过程结束，在每个epoch训练开始或者训练结束，在每个batch训练开始或者训练结束时执行一些操作，例如收集一些日志信息，改变学习率等超参数，提前终止训练过程等等；
- - 如果需要深入学习tf.Keras中的回调函数，不要犹豫阅读内置回调函数的源代码。

本章内容一下就比前面的内容丰满许多。

模型的训练，其步骤、数学原理应该了然于胸，才能更好地了解本章所言。

* [六、TensorFlow的高阶API](六、TensorFlow的高阶API.md)
* [6-1,构建模型的3种方法](./6-1,构建模型的3种方法.md)
- - 使用Sequential按层顺序构建模型（包含`callback`函数调用）；
- - 使用函数式API构建任意结构模型（可以很灵活的网络形式，如三个分支）；
- - 继承Model基类构建自定义模型。
* [6-2,训练模型的3种方法](./6-2,训练模型的3种方法.md)
- - 模型的训练主要有内置fit方法（该方法功能非常强大, 支持对numpy array, tf.data.Dataset以及 Python generator数据进行训练；并且可以通过设置回调函数实现对训练过程的复杂控制逻辑）；
- - 内置tran_on_batch方法（该内置方法相比较fit方法更加灵活，可以不通过回调函数而直接在批次层次上更加精细地控制训练的过程）；
- - 自定义训练循环（自定义训练循环无需编译模型，直接利用优化器根据损失函数反向传播迭代参数，拥有最高的灵活性）。
* [6-3,使用单GPU训练模型](./6-3,使用单GPU训练模型.md)
- - 目测 cuda 10.2 不支持 tf 2.1 ，没法做这节课的内容；
- - PyTorch 就没问题；
- - 本节开头有如何使用 Colab 的链接，知乎专栏里还有关于国内租用服务器的帖子。
* [6-4,使用多GPU训练模型](./6-4,使用多GPU训练模型.md)
- - “本节代码只能在 Colab 上才能正确执行”，但其实在本机上也能用 CPU 跑；
- - 介绍了 MirroredStrategy ，涉及到 `strategy = tf.distribute.MirroredStrategy()` 。
* [6-5,使用TPU训练模型](./6-5,使用TPU训练模型.md)
- - 如果想尝试使用Google Colab上的TPU来训练模型，也是非常方便，仅需添加6行代码。
* [6-6,使用tensorflow-serving部署模型](./6-6,使用tensorflow-serving部署模型.md)
- - TensorFlow训练好的模型以tensorflow原生方式保存成protobuf文件后可以用许多方式部署运行；
- - 例如通过 tensorflow-js 可以用javascrip脚本加载模型并在浏览器中运行模型；
- - 通过 tensorflow-lite 可以在移动和嵌入式设备上加载并运行TensorFlow模型；
- - 通过 tensorflow-serving 可以加载模型后提供网络接口API服务，通过任意编程语言发送网络请求都可以获取模型预测结果；
- - 通过 tensorFlow for Java接口，可以在Java或者spark(scala)中调用tensorflow模型进行预测。
- - 安装 tensorflow serving 有2种主要方法：通过Docker镜像安装，通过apt安装；
- - 通过Docker镜像安装是最简单，最直接的方法，推荐采用；
- - Docker可以理解成一种容器，其上面可以给各种不同的程序提供独立的运行环境。
* [6-7,使用spark-scala调用tensorflow模型](./6-7,使用spark-scala调用tensorflow模型.md)
- - 本篇文章介绍在spark中调用训练好的tensorflow模型进行预测的方法；
- - 本文内容的学习需要一定的spark和scala基础；
- - 如果使用pyspark的话会比较简单，只需要在每个excutor上用Python加载模型分别预测就可以了；
- - 但工程上为了性能考虑，通常使用的是scala版本的spark；
- - 本篇文章我们通过TensorFlow for Java 在spark中调用训练好的tensorflow模型；
- - 利用spark的分布式计算能力，从而可以让训练好的tensorflow模型在成百上千的机器上分布式并行执行模型推断。

至此，tf2 全部内容学毕。作者出身数据行业，开始几节范例与最后几节部署见其功底。

应速速实践。

* [后记：一个吃货和一道菜的故事](./后记：一个吃货和一道菜的故事.md)



```python
import torch
torch.cuda.get_device_capability()
torch.cuda.get_device_name()
```
