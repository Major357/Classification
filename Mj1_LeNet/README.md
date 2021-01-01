---

 - [LeNet-Model(pytorch版本)](https://blog.csdn.net/qq_41375318/article/details/110039768)

---

* [1.一 论文导读](#1)
* [2.二 论文精读](#2)
* [3.三 代码实现](#3)
* [4.四 问题思索](#4)

> 《LeNet》-1994
> ---LeNet
> 作者：
> 单位：
> 发表会议及时间：1994

<h2 id=1>一 论文导读<h2>

[转载于https://www.jiqizhixin.com/graph/technologies/6c9baf12-1a32-4c53-8217-8c9f69bd011b](https://www.jiqizhixin.com/graph/technologies/6c9baf12-1a32-4c53-8217-8c9f69bd011b) 

LeNet5诞生于1994年，是最早的卷积神经网络之一，并且推动了深度学习领域的发展。自从1988年开始，在多年的研究和许多次成功的迭代后，这项由Yann LeCun完成的开拓性成果被命名为LeNet5。


1989年，Yann LeCun等人在贝尔实验室的研究首次将反向传播算法进行了实际应用，并且认为学习网络泛化的能力可以通过提供来自任务域的约束来大大增强。

他将使用反向传播算法训练的卷积神经网络结合到读取“手写”数字上，并成功应用于识别美国邮政服务提供的手写邮政编码数字。这即是后来被称为LeNet的卷积神经网络的雏形。

同年，Yann LeCun在发表的另一篇论文中描述了一个小的手写数字识别问题，并且表明即使该问题是线性可分的，单层网络也表现出较差的泛化能力。而当在多层的、有约束的网络上使用有位移不变性的特征检测器（shift invariant feature detectors）时，该模型可以在此任务上表现得非常好。

他认为这些结果证明了将神经网络中的自由参数数量最小化可以增强神经网络的泛化能力。

1990年他们发表的论文再次描述了反向传播网络在手写数字识别中的应用，他们仅对数据进行了最小限度的预处理，而模型则是针对这项任务精心设计的，并且对其进行了高度约束。

输入数据由图像组成，每张图像上包含一个数字，在美国邮政服务提供的邮政编码数字数据上的测试结果显示该模型的错误率仅有1%，拒绝率约为9%。

其后8年他们的研究一直继续，直到1998年，Yann LeCun，Leon Bottou，Yoshua Bengio和Patrick Haffner在发表的论文中回顾了应用于手写字符识别的各种方法，并用标准手写数字识别基准任务对这些模型进行了比较，结果显示卷积神经网络的表现超过了其他所有模型。

他们同时还提供了许多神经网络实际应用的例子，如两种用于在线识别手写字符的系统和能每天读取数百万张支票的模型。

他们的研究取得了巨大的成功，并且激起了大量学者对神经网络的研究的兴趣。

**在今天向过去回首，目前性能最好的神经网络的架构已与LeNet不尽相同，但这个网络是大量神经网络架构的起点，并且也给这个领域带来了许多灵感。**

<h2 id=2>二 论文精读<h2>

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200618155040728.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMzc1MzE4,size_16,color_FFFFFF,t_70)

<h2 id=3>三 代码实现<h2>




```python
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


```


<h2 id=4>四 问题思索<h2>