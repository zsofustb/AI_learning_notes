### 安装方法与einops介绍

```
pip install einops
```

einops可以将tensor维度相关变化的代码变得更加清晰，写法容易、直观

### 使用方法

#### 导入第三方库

```
import torch
from einops import rearrange, reduce, repeat
```

#### rearrange

```
# transpose
x = torch.rand((2, 3, 4, 5))
out1 = rearrange(x, 'bs ic h w -> bs h ic w')  # <==> x.transpose(1, 2)
print(out1.shape)
print(torch.equal(out1, x.transpose(1, 2)))

# reshape减少维度
x = torch.rand((2, 3, 4, 5))
# 相当于x.reshape(6, 4, 5)
out2 = rearrange(x, 'bs ic h w -> (bs ic) h w')
print(out2.shape)
print(torch.equal(out2, x.reshape(6, 4, 5)))

# reshape增加维度
x = torch.rand((6, 4, 5))
out3 = rearrange(x, '(bs ic) h w -> bs ic h w', bs=2)
print(out3.shape)
print(torch.equal(out3, x.reshape(2, 3, 4, 5)))


# unsqueeze
x = torch.rand((2, 3, 4, 5))
out6 = rearrange(x, 'bs ic h w -> bs ic h w 1')
print(torch.equal(out6, torch.unsqueeze(x, dim=-1)))
```



#### reduce

```
# 平均池化
x = torch.rand((2, 3, 4, 5))
# 最后一维平均池化
out1 = reduce(x, 'bs ic h w -> bs ic h', 'mean')
print(out1.shape)
print(torch.equal(out1, torch.mean(x, dim=-1, keepdim=False)))
# 最后一维求和 & 维度保持不变
out2 = reduce(x, 'bs ic h w -> bs ic h 1', 'sum')
print(out2.shape)
print(torch.equal(out2, torch.sum(x, dim=-1, keepdim=True)))
# 最后两维进行最大池化
out3 = reduce(x, 'bs ic h w -> bs ic', 'max')
print(out3.shape)
```



#### repeat

```
x = torch.rand((2, 3, 4, 5))
out1 = rearrange(x, 'bs ic h w -> bs ic h w 1')
out2 = repeat(out1, 'bs ic h w t -> bs ic h (2 w) (2 t)')
print(out1.shape)
print(torch.equal(out2, torch.tile(out1, (1, 1, 1, 2, 2))))

x = torch.arange(4)
out1 = rearrange(x, '(h w) -> h w', h=2)
out2 = repeat(out1, 'h w -> (2 h) (3 w)')
print(out2)
```



#### einsum

三个准则（einsum 索引规则）

| 说明                                                         | 示例        | 含义                                                         |
| ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| **在不同输入之间重复出现的索引**<br>表示沿着该维度做乘法     | `ik,kj->ij` | 索引 `k` 同时出现在第一个矩阵的列和第二个矩阵的行 → 矩阵乘法 |
| **只出现在输入、未出现在输出的索引**<br>表示在该维度上求和（reduce） | `ik,kj->ij` | 索引 `k` 最终未出现在输出 → 对 `k` 维度求和                  |
| **输出维度的顺序可任意指定**<br>可方便地进行 permute         | `ik,kj->ji` | 把矩阵乘法结果再转置一次                                     |

```
# 内积
x = torch.arange(3)
print(torch.equal(einsum(x, x, 'i, i -> '), torch.inner(x, x)))
# 外积
x = torch.arange(3)
print(torch.equal(einsum(x, x, 'i, j -> i j'), torch.outer(x, x)))

# 矩阵对应位置相乘
x = torch.Tensor([[0, 1], [2, 3]])
y = torch.Tensor([[0, 1], [2, 3]])
print(einsum(x, y, 'i j, i j -> i j'))
# 矩阵对应位置相乘再相加
print(einsum(x, y, 'i j, i j ->'))
# 观察下面输出
print(einsum(x, y, 'i j, j k -> i j k'))
print(einsum(x, y, 'i j, j k -> i k j'))


# 矩阵乘法
x = torch.rand((2, 3))
y = torch.rand((3, 4))
print(torch.equal(torch.mm(x, y), einsum(x, y, 'i j, j k -> i k')))

x = torch.arange(16)
print(x.shape)
mat = rearrange(x, '(j k) -> j k', j=4)
print(mat)
# 对角线元素
diag = einsum(mat, 'i i -> i')
print(diag)
# trace
tr = einsum(mat, 'i i ->')
print(tr)
```

