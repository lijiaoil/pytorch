# Pytorch容器 #
## nn.Sequential(*layers) ##
类似于torch7中的Sequential，将每一个模块按照他们的顺序送入到nn.Sequential中 ,输入可以是一些列有顺序的模块

	```python
	conv1=nn.FractionalMaxPool2d(2,output_ratio=(scaled,scaled))
	conv2=nn.Conv2d(D,D,kernel_size= 3,stride=1,padding=1,bias=True)
	conv3=nn.Upsample(size=(inputRes,inputRes),mode='bilinear')
	return nn.Sequential(conv1,conv2,conv3)
	```

输入也可以是一个orderDict

	```python
	# Example of using Sequential with OrderedDict
	# orderdict按照建造时候的顺序进行存储
	model = nn.Sequential(OrderedDict([    
	  ('conv1', nn.Conv2d(1,20,5)),
	  ('relu1', nn.ReLU()),
	  ('conv2', nn.Conv2d(20,64,5)),
	  ('relu2', nn.ReLU())
	]))
	```

orderdict示例

	```python
	#orderdict部分源代码来自https://www.cnblogs.com/gide/p/6370082.html，稍有改动
	import collections
	print "Regular dictionary"
	d={}
	d['a']='A'
	d['b']='B'
	d['c']='C'
	for k,v in d.items():
	    print k,v
	
	print "\nOrder dictionary"
	d1 = collections.OrderedDict()
	d1['a'] = 'A'
	d1['b'] = 'B'
	d1['c'] = 'C'
	d1['2'] = '2'
	d1['1'] = '1'
	for k,v in d1.items():
	    print k,v
	```

![](https://img-blog.csdnimg.cn/20210630100149334.jpg#pic_center)


输入也可以是list,然后输入的时候用*来引用

    ```python
	 layers = []
	 layers.append(block(inplanes, outplanes,inputRes,baseWidth=9,cardinality=4,stride=1,preact=preact))
	 return nn.Sequential(*layers)  # 不加*号，会报错 TypeError: list is not a Module subclass
	```
因为从nn.Sequential的定义来看

![](https://img-blog.csdnimg.cn/2021063010042731.png)

输入要么事orderdict,要么事一系列的模型，遇到上述的list，必须用*号进行转化

Sequential好处是啥呢？

    ```python
	def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )
	```
在前传forward的时候，一步完成，而不需要conv,bn，relu在forward函数里都写一遍
## torch.nn.ModuleList  ##
存储一系列模型的module list，操作类似于标准的python list

	```python
	class MyModule(nn.Module):
	    def __init__(self):
	        super(MyModule, self).__init__()
	        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])
	
	    def forward(self, x):
	        # ModuleList can act as an iterable, or be indexed using ints
	        for i, l in enumerate(self.linears):
	            x = self.linears[i // 2](x) + l(x)
	        return x
	```

代码中的`[nn.Linear(10, 10) for i in range(10)]`是一个python列表，必须要把它转换成一个module list列表才可以被pytorch使用，否则在运行的时候会报错，什么错呢？`RuntimeError: Input type (CUDAFloatTensor) and weight type (CPUFloatTensor) should be the same`,如果这部分不转成modulelist（gpu）,那么只是pythonlist(cpu) 来承载这些模型，定造成cpu和gpu的冲突。

## python *号 ##
单个星号代表这个位置接收任意多个非关键字参数，并转化成元表。也就是b 会接受除了a之外的剩下的非关键字参数，需要注意的是加在形参面前代表的是收集参数，如果*号加在了是实参上(例如第十四行)，代表的是将输入迭代器拆成一个个元素

	```python
	d1 = collections.OrderedDict()
	d1['a'] = 'A'
	d1['b'] = 'B'
	d1['c'] = 'C'
	d1['2'] = '2'
	d1['1'] = '1'
	
	def one(a,*b):
	    print(b)
	def two(*b):
	    print(b)
	c = [6,7,8,9]
	one(1,2,3,4,5,6)
	one(*c)    #传入实参的时候，加上*号，可以将列表中的元素拆成一个个的元素
	one(*d1)   #传入实参的时候，加上*号，可以将字典中的元素拆成一个个的元素
	one(c)
	one(d1)
	two(c)
	two(d1)
	
	```



![](https://img-blog.csdnimg.cn/20210630100921361.png)

** 双星号代表这个位置接收任意多个关键字参数，并按照关键字转化成字典
用双星号传入实参的时候，一定是所有的实参必须带有关键字

	```python
	def three1(**b):
	    print(b)
	three(a=1,b=2,c=3,d=4,e=5,f=6)
	```
    
# pytorch中BatchNorm1d、BatchNorm2d、BatchNorm3d #
## nn.BatchNorm1d(num_features) ##
	```python
	    1.对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作
	    2.num_features：
	            来自期望输入的特征数，该期望输入的大小为'batch_size x num_features [x width]'
	            意思即输入大小的形状可以是'batch_size x num_features' 和 'batch_size x num_features x width' 都可以。
	            （输入输出相同）
	            输入Shape：（N, C）或者(N, C, L)
	            输出Shape：（N, C）或者（N，C，L）
	
	      eps：为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
	      momentum：动态均值和动态方差所使用的动量。默认为0.1。
	      affine：一个布尔值，当设为true，给该层添加可学习的仿射变换参数。
	    3.在每一个小批量（mini-batch）数据中，计算输入各个维度的均值和标准差。gamma与beta是可学习的大小为C的参数向量（C为输入大小）
	      在训练时，该层计算每次输入的均值与方差，并进行移动平均。移动平均默认的动量值为0.1。
	      在验证时，训练求得的均值/方差将用于标准化验证数据。 
	    4.例子
	            >>> # With Learnable Parameters
	            >>> m = nn.BatchNorm1d(100) #num_features指的是randn(20, 100)中（N, C）的第二维C
	            >>> # Without Learnable Parameters
	            >>> m = nn.BatchNorm1d(100, affine=False)
	            >>> input = autograd.Variable(torch.randn(20, 100)) #输入Shape：（N, C）
	            >>> output = m(input)  #输出Shape：（N, C）
	```
## nn.BatchNorm2d(num_features) ##
	```python
        1.对小批量(mini-batch)3d数据组成的4d输入进行批标准化(Batch Normalization)操作
        2.num_features： 
                来自期望输入的特征数，该期望输入的大小为'batch_size x num_features x height x width'
                （输入输出相同）
                    输入Shape：（N, C，H, W)
                    输出Shape：（N, C, H, W）
          eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
          momentum： 动态均值和动态方差所使用的动量。默认为0.1。
          affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数。
        3.在每一个小批量（mini-batch）数据中，计算输入各个维度的均值和标准差。gamma与beta是可学习的大小为C的参数向量（C为输入大小）
          在训练时，该层计算每次输入的均值与方差，并进行移动平均。移动平均默认的动量值为0.1。
          在验证时，训练求得的均值/方差将用于标准化验证数据。
        4.例子
            >>> # With Learnable Parameters
            >>> m = nn.BatchNorm2d(100) #num_features指的是randn(20, 100, 35, 45)中（N, C，H, W)的第二维C
            >>> # Without Learnable Parameters
            >>> m = nn.BatchNorm2d(100, affine=False)
            >>> input = autograd.Variable(torch.randn(20, 100, 35, 45))  #输入Shape：（N, C，H, W)
            >>> output = m(input)
	```
## nn.BatchNorm3d(num_features) ##
    ```python
		1.对小批量(mini-batch)4d数据组成的5d输入进行批标准化(Batch Normalization)操作
        2.num_features： 
                来自期望输入的特征数，该期望输入的大小为'batch_size x num_features depth x height x width'
                （输入输出相同）
                 输入Shape：（N, C，D, H, W)
                 输出Shape：（N, C, D, H, W）

          eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
          momentum： 动态均值和动态方差所使用的动量。默认为0.1。
          affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数。

        3.在每一个小批量（mini-batch）数据中，计算输入各个维度的均值和标准差。gamma与beta是可学习的大小为C的参数向量（C为输入大小）
          在训练时，该层计算每次输入的均值与方差，并进行移动平均。移动平均默认的动量值为0.1。
          在验证时，训练求得的均值/方差将用于标准化验证数据。
        4.例子
            >>> # With Learnable Parameters
            >>> m = nn.BatchNorm3d(100)  #num_features指的是randn(20, 100, 35, 45, 10)中（N, C, D, H, W）的第二维C
            >>> # Without Learnable Parameters
            >>> m = nn.BatchNorm3d(100, affine=False)  #num_features指的是randn(20, 100, 35, 45, 10)中（N, C, D, H, W）的第二维C
            >>> input = autograd.Variable(torch.randn(20, 100, 35, 45, 10)) #输入Shape：（N, C, D, H, W） 
            >>> output = m(input)
	```
# pytorch Adam #
	```python
	torch.optim.Adam(params,
	                lr=0.001,
	                betas=(0.9, 0.999),
	                eps=1e-08,
	                weight_decay=0,
	                amsgrad=False)
	```
params
模型里需要被更新的可学习参数

lr:学习率

eps:加在分母上防止除0

weight_decay:作用是用当前可学习参数p的值修改偏导数,weight_decay的作用是L2正则化，和Adam并无直接关系。

amsgrad:如果amsgrad为True，则在上述伪代码中的基础上，保留历史最大的v，记为vmax，每次计算都是用最大的vmax，否则是用当前v。amsgrad和Adam并无直接关系。
![](https://img-blog.csdnimg.cn/20210630112137704.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTUyMDM0,size_16,color_FFFFFF,t_70)

# Variable #
[https://www.pytorchtutorial.com/2-2-variable/](https://www.pytorchtutorial.com/2-2-variable/)

## 什么是 Variable？ ##
在 Torch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动. 那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor . 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.

我们定义一个 Variable:

    ```python
	import torch
	from torch.autograd import Variable # torch 中 Variable 模块
	 
	# 先生鸡蛋
	tensor = torch.FloatTensor([[1,2],[3,4]])
	# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
	variable = Variable(tensor, requires_grad=True)
	 
	print(tensor)
	"""
	 1  2
	 3  4
	[torch.FloatTensor of size 2x2]
	"""
	 
	print(variable)
	"""
	Variable containing:
	 1  2
	 3  4
	[torch.FloatTensor of size 2x2]
	"""
	```
## Variable 计算, 梯度 ##
我们再对比一下 tensor 的计算和 variable 的计算

    ```python
	t_out = torch.mean(tensor*tensor)       # x^2
	v_out = torch.mean(variable*variable)   # x^2
	print(t_out)
	print(v_out)    # 7.5
	```
到目前为止, 我们看不出什么不同, 但是时刻记住, Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph. 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 而 tensor 就没有这个能力。

`v_out = torch.mean(variable*variable)`就是在计算图中添加的一个计算步骤, 计算误差反向传递的时候有他一份功劳, 举个例子:

    ```python
	v_out.backward()    # 模拟 v_out 的误差反向传递
	 
	# 下面两步看不懂没关系, 只要知道 Variable 是计算图的一部分, 可以用来传递误差就好.
	# v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
	# 针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2
	 
	print(variable.grad)    # 初始 Variable 的梯度
	"""
	 0.5000  1.0000
	 1.5000  2.0000
	"""
	```
## 获取 Variable 里面的数据 ##
直接`print(variable)`只会输出 Variable 形式的数据, 在很多时候是用不了的(比如想要用 plt 画图), 所以我们要转换一下, 将它变成 tensor 形式.

    ```python
	print(variable)     #  Variable 形式
	"""
	Variable containing:
	 1  2
	 3  4
	[torch.FloatTensor of size 2x2]
	"""
	 
	print(variable.data)    # tensor 形式
	"""
	 1  2
	 3  4
	[torch.FloatTensor of size 2x2]
	"""
	 
	print(variable.data.numpy())    # numpy 形式
	"""
	[[ 1.  2.]
	 [ 3.  4.]]
	"""
	```
# detach 与 detach_ #
## 计算图 ##
在pytorch中,autograd是由计算图实现的.Variable是autograd的核心数据结构,其构成分为三部分: data(tensor), grad(也是Variable), grad_fn(得到这一节点的直接操作).对于requires_grad为false的节点,是不具有grad的.
![](https://img-blog.csdnimg.cn/202106301706179.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTUyMDM0,size_16,color_FFFFFF,t_70)

用户自己创建的节点是`leaf_node`(如图中的abc三个节点),不依赖于其他变量,对于`leaf_node`不能进行`in_place`操作.根节点是计算图的最终目标(如图y),通过链式法则可以计算出所有节点相对于根节点的梯度值.这一过程通过调用root.backward()就可以实现.
因此,detach所做的就是,重新声明一个变量,指向原变量的存放位置,但是`requires_grad`为false.更深入一点的理解是,计算图从detach过的变量这里就断了, 它变成了一个`leaf_node`.即使之后重新将它的`requires_node`置为true,它也不会具有梯度.

![](https://img-blog.csdnimg.cn/20210630171712158.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTUyMDM0,size_16,color_FFFFFF,t_70)

另一方面,在调用完backward函数之后,非`leaf_node`的梯度计算完会立刻被清空.这也是为什么在执行backward之前显存占用很大,执行完之后显存占用立刻下降很多的原因.当然,这其中也包含了一些中间结果被存在buffer中,调用结束后也会被释放.至于另一个参数volatile,如果一个变量的`volatile=true`,它可以将所有依赖于它的节点全部设为`volatile=true`,优先级高于`requires_grad=true`.这样的节点不会进行求导,即使`requires_grad`为真,也无法进行反向传播.在inference中如果采用这种设置,可以实现一定程度的速度提升,并且节约大概一半显存。
## tensor.detach() ##
返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是`requires_grad=false`，得到的这个tensor永远不需要计算其梯度，不具有grad。即使之后重新将它的requires_grad置为true,它也不会具有梯度grad。这样我们就会继续使用这个新的tensor进行计算，后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播。

注意：使用detach返回的tensor和原始的tensor共同一个内存，即一个修改另一个也会跟着改变。

    ```python
	import torch
	 
	a = torch.tensor([1, 2, 3.], requires_grad=True)
	print(a.grad)
	out = a.sigmoid()
	 
	out.sum().backward()
	print(a.grad)
	"""
	None
	tensor([0.1966, 0.1050, 0.0452])
	"""
	```
### 当使用detach()分离tensor但是没有更改这个tensor时，并不会影响backward(): ###

	```
	import torch
	 
	a = torch.tensor([1, 2, 3.], requires_grad=True)
	print(a.grad)
	out = a.sigmoid()
	print(out)
	 
	#添加detach(),c的requires_grad为False
	c = out.detach()
	print(c)
	 
	#这时候没有对c进行更改，所以并不会影响backward()
	out.sum().backward()
	print(a.grad)
	 
	"""
	None
	tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
	tensor([0.7311, 0.8808, 0.9526])
	tensor([0.1966, 0.1050, 0.0452])
	"""
	```
从上可见tensor  c是由out分离得到的，但是我也没有去改变这个c，这个时候依然对原来的out求导是不会有错误的，即c,out之间的区别是c是没有梯度的，out是有梯度的,但是需要注意的是下面两种情况是汇报错的.

### 当使用detach()分离tensor，然后用这个分离出来的tensor去求导数，会影响backward()，会出现错误 ###

    ```python
	import torch
	 
	a = torch.tensor([1, 2, 3.], requires_grad=True)
	print(a.grad)
	out = a.sigmoid()
	print(out)
	 
	#添加detach(),c的requires_grad为False
	c = out.detach()
	print(c)
	 
	#使用新生成的Variable进行反向传播
	c.sum().backward()
	print(a.grad)
	 
	"""
	None
	tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
	tensor([0.7311, 0.8808, 0.9526])
	Traceback (most recent call last):
	  File "test.py", line 13, in <module>
	    c.sum().backward()
	  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/tensor.py", line 102, in backward
	    torch.autograd.backward(self, gradient, retain_graph, create_graph)
	  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/autograd/__init__.py", line 90, in backward
	    allow_unreachable=True)  # allow_unreachable flag
	RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
	"""
	```
### 当使用detach()分离tensor并且更改这个tensor时，即使再对原来的out求导数，会影响backward()，会出现错误 ###

如果此时对c进行了更改，这个更改会被autograd追踪，在对out.sum()进行backward()时也会报错，因为此时的值进行backward()得到的梯度是错误的：

	```python
	import torch
	 
	a = torch.tensor([1, 2, 3.], requires_grad=True)
	print(a.grad)
	out = a.sigmoid()
	print(out)
	 
	#添加detach(),c的requires_grad为False
	c = out.detach()
	print(c)
	c.zero_() #使用in place函数对其进行修改
	 
	#会发现c的修改同时会影响out的值
	print(c)
	print(out)
	 
	#这时候对c进行更改，所以会影响backward()，这时候就不能进行backward()，会报错
	out.sum().backward()
	print(a.grad)
	 
	"""
	None
	tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
	tensor([0.7311, 0.8808, 0.9526])
	tensor([0., 0., 0.])
	tensor([0., 0., 0.], grad_fn=<SigmoidBackward>)
	Traceback (most recent call last):
	  File "test.py", line 16, in <module>
	    out.sum().backward()
	  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/tensor.py", line 102, in backward
	    torch.autograd.backward(self, gradient, retain_graph, create_graph)
	  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/autograd/__init__.py", line 90, in backward
	    allow_unreachable=True)  # allow_unreachable flag
	RuntimeError: one of the variables needed for gradient computation has been modified 
	by an inplace operation
	"""
	```
## tensor.detach_() ##
将一个tensor从创建它的图中分离，并把它设置成叶子tensor

其实就相当于变量之间的关系本来是`x -> m -> y`,这里的叶子tensor是x，但是这个时候对m进行了`m.detach_()`操作,其实就是进行了两个操作：


- 将m的`grad_fn`的值设置为None,这样m就不会再与前一个节点x关联，这里的关系就会变成`x, m -> y`,此时的m就变成了叶子结点


- 然后会将m的`requires_grad`设置为False，这样对y进行backward()时就不会求m的梯度

**总结：**其实`detach()`和`detach_()`很像，两个的区别就是`detach_()`是对本身的更改，`detach()`则是生成了一个新的tensor,比如`x -> m -> y`中如果对m进行`detach()`，后面如果反悔想还是对原来的计算图进行操作还是可以的,但是如果是进行了`detach_()`，那么原来的计算图也发生了变化，就不能反悔了.
 

## 用处 ##

    ```
	# y=A(x), z=B(y) 求B中参数的梯度，不求A中参数的梯度
	# 第一种方法
	y = A(x)
	z = B(y.detach())
	z.backward()
	 
	# 第二种方法
	y = A(x)
	y.detach_()
	z = B(y)
	z.backward()
	```
## GAN网络中detach的使用 ##
`backward_D`时，如果输入是真实图，那么产生loss，输入假图，也产生loss。
这两个梯度进行更新D。如果是真实图(`real_B`)，由于real_B是初始结点，所以没什么可担心的。但是对于生成图`fake_B`，由于`fake_B`是由 `netG.forward(real_A)`产生的。我们只希望该loss更新D不要影响到G. 因此这里需要“截断反传的梯度流”，用 `fake_AB = fake_AB.detach()`从而让梯度不要通过`fake_AB`反传到netG中！

`backward_G`时，由于在调用 `backward_G`已经调用了`zero_grad`，所以没什么好担心的。
更新G时，来自D的GAN损失是，`netD.forward(fake_AB)`，得到 `pred_fake`，然后得到损失，反传播即可。注意，这里反向传播时，会先将梯度传到`fake_AB`结点，然而我们知道 `fake_AB`即 `fake_B`结点，而`fake_B`正是由`netG(real_A)`产生的，所以还会顺着继续往前传播，从而得到G的对应的梯度。

