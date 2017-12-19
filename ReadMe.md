### 1. 项目说明

研一上学期选的“互联网创新与服务”课程布置的作业，其中具体的作业要求是：

利用提供的数据集，构造一个高效实用的图像分类器模型。根据已提供的12500张猫和狗的图片，搭建神经网络训练一个图像分类模型，为每一个测试数据预测出最佳的类别标签，最终以正确率评估模型的优劣。

### 2. 数据说明

Dogs vs. Cats是一个传统的二分类问题。其训练集包含12500张图片，放置在同一文件夹（train）下，命名格式为*<category>.<num>.jpg*, 如*cat.10000.jpg*、*dog.100.jpg*，测试集包含12500张图片，放置在同一文件夹（test）下，命名为*<num>.jpg*，如*1000.jpg*。需根据训练集的图片训练模型，并在测试集上进行预测，提交图片的分类结果。最后提交的csv文件如下，第一列是图片的*<num>*，第二列是图片的分类结果。（0代表狗，1代表猫）

```
10001,0
10002,1
```

### 3. 文件组织架构

​	本项目参考了[PyTorch](https://github.com/chenyuntc/pytorch-best-practice)的官方代码，在其上进行相应修改。

​	文件组织架构如下：

```
├── checkpoints/
├── data/
│   ├── __init__.py
│   ├── dataset.py
├── models/
│   ├── __init__.py
│   ├── BasicModule.py
│   └── ResNet34.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
├── submission.csv
```

其中：

- checkpoints/： 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练

- data/：数据相关操作，包括数据预处理、dataset实现等

- models/：模型定义，可以有多个模型，在这里

- ResNet34，一个模型对应一个文件

- config.py：配置文件，所有可配置的变量都集中在此，并提供默认值

- main.py：主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数

- requirements.txt：程序依赖的第三方库

- readme.docx：提供程序的必要说明

- submission.csv：最终的分类结果

  **程序注释清晰，具体细节可看注释**

### 4. 运行环境

系统： Ubuntu 14.04

语言： Anaconda2.7 +  Python

框架： Pytorch

安装PyTorch：参照[PyTorch官网](http://pytorch.org/) ，`conda install pytorch torchvision -c pytorch`

### 5.**关于__init__.py**

可以看到，几乎每个文件夹下都有*__init__.py*，一个目录如果包含了*__init__.py* 文件，那么它就变成了一个包（package）。*__init__.py*可以为空，也可以定义包的属性和方法，但其必须存在，其它程序才能从这个目录中导入相应的模块或函数。例如在data/文件夹下有*__init__.py*，则在main.py 中就可以

```
from data.dataset import DogCat

```

而如果*在data/__init__.py*中写入

```
from .dataset import DogCat

```

则在main.py中就可以直接写为：

```
from data import DogCat

```

或者

```
import data;
dataset = data.DogCat

```

相比于*from data.dataset import DogCat*更加便捷。可以看到，几乎每个文件夹下都有*__init__.py*，一个目录如果包含了*__init__.py* 文件，那么它就变成了一个包（package）。*__init__.py*可以为空，也可以定义包的属性和方法，但其必须存在，其它程序才能从这个目录中导入相应的模块或函数。例如在data/文件夹下有*__init__.py*，则在main.py 中就可以

```
from data.dataset import DogCat

```

而如果*在data/__init__.py*中写入

```
from .dataset import DogCat

```

则在main.py中就可以直接写为：

```
from data import DogCat

```

或者

```
import data;
dataset = data.DogCat

```

相比于*from data.dataset import DogCat*更加便捷。

### 6. **配置文件**

在模型定义、数据处理和训练等过程都有很多变量，这些变量应提供默认值，并统一放置在配置文件中，这样在后期调试、修改代码或迁移程序时会比较方便，在这里我们将所有可配置项放在*config.py*中。

这些都只是默认参数，在这里还提供了更新函数，根据字典更新配置参数。

这样我们在实际使用时，并不需要每次都修改config.py，只需要通过命令行传入所需参数，覆盖默认配置即可。

例如：

```
opt = DefaultConfig()
new_config = {'lr':0.1,'use_gpu':False}
opt.parse(new_config)
opt.lr == 0.1
```

### 7.帮助函数

为了方便他人使用, 程序中还应当提供一个帮助函数，用于说明函数是如何使用。程序的命令行接口中有众多参数，如果手动用字符串表示不仅复杂，而且后期修改config文件时，还需要修改对应的帮助信息，十分不便。这里使用了Python标准库中的inspect方法，可以自动获取config的源代码。help的代码如下:

```
def help():
  '''
  打印帮助的信息： python file.py help
   '''
   
   print('''
   usage : python {0} <function> [--args=value,]
   <function> := train | test | help
   example: 
           python {0} train --env='env0701' --lr=0.01
           python {0} test --dataset='path/to/dataset/root/'
           python {0} help
   avaiable args:'''.format(__file__))

   from inspect import getsource
   source = (getsource(opt.__class__))
   print(source)
```

### 8. 使用

正如help函数的打印信息所述，可以通过命令行参数指定变量名.下面是三个使用例子，fire会将包含-的命令行参数自动转层下划线_，也会将非数值的值转成字符串。所以--train-data-root=data/train和--train_data_root='data/train'是等价的

```
# 首先安装指定依赖：
pip install -r requirements.txt

# 训练模型
python main.py train 
        --train-data-root=../data/train/ 
        --load-model-path=None
        --use-gpu=True
# 测试模型
python main.py test
       --test-data-root=data/test1 
       --load-model-path='checkpoints/resnet34.pth' 
      
# 打印帮助信息
python main.py help
```

