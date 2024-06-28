# d2l-zh-pytorch-part II

[l1正则与l2正则的特点是什么，各有什么优势？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/26485586)

## normalization和norm

## Transformer，sam-track的代码实践

2024-2-4

TRANSFORMER

<img src="C:\Users\23828\AppData\Roaming\Typora\typora-user-images\image-20240204152252684.png" alt="image-20240204152252684" style="zoom:50%;" />

### 定义一个输入和隐藏层=4，输出=8的前馈网络层，命令行结果输出如下：



<img src="C:\Users\23828\AppData\Roaming\Typora\typora-user-images\image-20240204151643552.png" alt="image-20240204151643552" style="zoom:50%;" />



### eval方法解释： 与self.train(false)等价，和dropout，batchnorm等关系密切

### 返回模型对象自身

![image-20240204152452658](C:\Users\23828\AppData\Roaming\Typora\typora-user-images\image-20240204152452658.png)



```python
#transformer decoder块定义
def __init__(self, vocab_size, key_size, query_size, value_size,
                    num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, num_layers, dropout, use_bias=False, **kwargs):
```



```cmd
(d2l) D:\python-AI-ML-learning\d2l-zh\transformer>python -i model.py
>>> encoder = TransformerEncoder(
... 200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
>>> encoder.eval()
TransformerEncoder(
  (embedding): Embedding(200, 24)
  (pos_encoding): PositionalEncoding(
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (blks): Sequential(
    (block0): EncoderBlock(
      (attention): MultiHeadAttention(
        (attention): DotProductAttention(
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (W_q): Linear(in_features=24, out_features=24, bias=False)
        (W_k): Linear(in_features=24, out_features=24, bias=False)
        (W_v): Linear(in_features=24, out_features=24, bias=False)
        (W_o): Linear(in_features=24, out_features=24, bias=False)
      )
      (addnorm1): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      (ffn): PositionWiseFFN(
        (dense1): Linear(in_features=24, out_features=48, bias=True)
        (relu): ReLU()
        (dense2): Linear(in_features=48, out_features=24, bias=True)
      )
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
    )
    (block1): EncoderBlock(
      (attention): MultiHeadAttention(
        (attention): DotProductAttention(
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (W_q): Linear(in_features=24, out_features=24, bias=False)
        (W_k): Linear(in_features=24, out_features=24, bias=False)
        (W_v): Linear(in_features=24, out_features=24, bias=False)
        (W_o): Linear(in_features=24, out_features=24, bias=False)
      )
      (addnorm1): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      (ffn): PositionWiseFFN(
        (dense1): Linear(in_features=24, out_features=48, bias=True)
        (relu): ReLU()
        (dense2): Linear(in_features=48, out_features=24, bias=True)
      )
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
    )
  )
)
>>> encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape
torch.Size([2, 100, 24])
>>> torch.Size([2, 100, 24])
torch.Size([2, 100, 24])
>>> X = torch.ones((2, 100, 24))
>>> valid_lens = torch.tensor([3, 2])
>>> X
tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]],

        [[1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         ...,
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.],
         [1., 1., 1.,  ..., 1., 1., 1.]]])
>>> valid_lens                       
tensor([3, 2])
>>> encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
>>> encoder_blk.eval()
EncoderBlock(
  (attention): MultiHeadAttention(
    (attention): DotProductAttention(
      (dropout): Dropout(p=0.5, inplace=False)
    )
    (W_q): Linear(in_features=24, out_features=24, bias=False)
    (W_k): Linear(in_features=24, out_features=24, bias=False)
    (W_v): Linear(in_features=24, out_features=24, bias=False)
    (W_o): Linear(in_features=24, out_features=24, bias=False)
  )
  (addnorm1): AddNorm(
    (dropout): Dropout(p=0.5, inplace=False)
    (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
  )
  (ffn): PositionWiseFFN(
    (dense1): Linear(in_features=24, out_features=48, bias=True)
    (relu): ReLU()
    (dense2): Linear(in_features=48, out_features=24, bias=True)
  )
  (addnorm2): AddNorm(
    (dropout): Dropout(p=0.5, inplace=False)
    (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
  )
)
>>> encoder_blk(X, valid_lens).shape
torch.Size([2, 100, 24])
>>>
```

#### ` (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)`的解释：

<img src="C:\Users\23828\AppData\Roaming\Typora\typora-user-images\image-20240204153622855.png" alt="image-20240204153622855" style="zoom:50%;" />


在神经网络中，每个线性层（例如全连接层或卷积层）都包含权重（weights）和偏置（bias）。`bias` 是一个可学习的参数，其目的是在层的输出上添加一个固定的偏移量。

在 PyTorch 或其他深度学习框架中，创建线性层时，你可以选择是否包含偏置项。这是通过设置 `bias` 参数来控制的。下面是两种情况的区别：

1. **bias=True：** 如果设置 `bias=True`，则创建的线性层将包含一个可学习的偏置项。这意味着层的输出会受到一个偏置的影响，使得模型可以通过学习调整这个偏置项来更好地拟合训练数据。

   ```
   # 示例：创建一个包含偏置项的线性层
   linear_layer_with_bias = nn.Linear(in_features=10, out_features=5, bias=True)
   ```

2. **bias=False：** 如果设置 `bias=False`，则创建的线性层将没有偏置项。这样，层的输出不再受到一个额外的偏置的影响，相当于对每个输出节点都应用了零偏移。

   ```
   # 示例：创建一个没有偏置项的线性层
   linear_layer_no_bias = nn.Linear(in_features=10, out_features=5, bias=False)
   ```

选择是否使用偏置通常取决于具体的任务和网络结构。在某些情况下，使用偏置可以帮助模型更灵活地适应数据。但在其他情况下，尤其是在一些规范化层（如 Batch Normalization 或 Layer Normalization）的使用中，有时可以选择不使用偏置项。

总体而言，这是一个在网络设计中的超参数选择，可以根据具体的实验和问题进行调整。



![image-20240205092237014](C:\Users\23828\AppData\Roaming\Typora\typora-user-images\image-20240205092237014.png)



## 2-5

AOT和De-AOT

associating objects with transformers

decoupling AOT

![image-20240205113203992](C:\Users\23828\AppData\Roaming\Typora\typora-user-images\image-20240205113203992.png)







# attention

x_train和y_train设置为实际分布+一定噪声的测试用数据

在这里解释attention机制时，x_train由范围内随机数生成，y_train = f(x_train)+噪声，实际应用时得到的数据就是围绕真实分布有噪声的，training data和 test data都是从同分布的数据集中获取的，只不过在这里设置x_test为等间距的，便于plt画图

x_test为测试时给出的数据

y_hat是模型预测的数据

y_truth是根据实际的分布函数算出的真实数据

X_repeat矩阵形状为

0.0 0.0 0.0 0.0 0.0

0.2 0.2 0.2 0.2 0.2

0.4 0.4 0.4 0.4 0.4

0.6 0.6 0.6 0.6 0.6 

0.8 0.8 0.8 0.8 0.8 

计算X_repeat和每个x_train之间的关联程度，然后

<img src="C:\Users\23828\AppData\Roaming\Typora\typora-user-images\image-20240206121931294.png" alt="image-20240206121931294" style="zoom: 33%;" />

latent 潜在向量特征表示

GPT等模型中，输入的文本数据通过编码器 (encoder)处理后得到的隐藏表示

"Latent vector" 是指**GPT等模型中，输入的文本数据通过编码器 (encoder)处理后得到的隐藏表示** (hidden representation)，通常是一个固定大小的向量。 这个向量被称为"latent vector"，因为它包含了输入文本的潜在信息，即模型通过学习得到的文本特征或语义信息。
