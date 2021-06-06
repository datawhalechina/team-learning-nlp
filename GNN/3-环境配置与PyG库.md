# 环境配置与PyG中图与图数据集的表示和使用

## 一、引言

PyTorch Geometric (PyG)是PyTorch的一个几何深度学习扩展库。PyG囊括了在图和其他不规则结构上进行深度学习的各种方法。在图和其他不规则结构上进行深度学习也被称为几何深度学习。所囊括的方法来自各种已发表的论文。PyG内置了一个易于使用的mini-batch加载器，用于大量的通用基准数据集的加载。图数据集可以由节点数量非常多的单个图组成，也可以由数量非常多的节点数量较少的图组成。此外，PyG还集成了一些有用的数据转换工具。PyG既可以用于图的学习任务，也可以用于三维网格或点云的学习任务。

通过此实践内容，我们将

1. 首先学习程序运行环境的配置。
2. 接着学习PyG中图的表示及其使用，即PyG中`Data`类的学习。
3. 最后学习PyG中图数据集的表示及其使用，即PyG中`Dataset`类的学习。

## 二、环境配置

内容来源：[Installation — pytorch_geometric 1.7.0 documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

1. 使用`nvidia-smi`命令查询显卡驱动是否正确安装

![image-20210515204452045](images/image-20210515204452045.png)

2. 安装正确版本的pytorch和cudatoolkit，此处安装1.8.1版本的pytorch和11.1版本的cudatoolkit
   1. `conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`
   2. 确认是否正确安装，正确的安装应出现下方的结果
   ```txt
   $ python -c "import torch; print(torch.__version__)"
   # 1.8.1
   $ python -c "import torch; print(torch.version.cuda)"
   # 11.1
   ```
   
3. 安装正确版本的PyG

   ```txt
   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
   pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
   pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
   pip install torch-geometric
   ```

其他版本的安装方法以及安装过程中出现的大部分问题的解决方案可以在[内容来源](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)页面找到。

## 三、`Data`类——PyG中图的表示及其使用

### `Data`对象的构造

`Data`类的构造函数：

```python
class Data(object):

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
    r"""
    Args:
        x (Tensor, optional): 节点属性矩阵，大小为`[num_nodes, num_node_features]`
        edge_index (LongTensor, optional): 边索引矩阵，大小为`[2, num_edges]`，第0行为尾节点，第1行为头节点，头指向尾
        edge_attr (Tensor, optional): 边属性矩阵，大小为`[num_edges, num_edge_features]`
        y (Tensor, optional): 节点或图的标签，任意大小（，其实也可以是边的标签）
	
    """
    self.x = x
    self.edge_index = edge_index
    self.edge_attr = edge_attr
    self.y = y

    for key, item in kwargs.items():
        if key == 'num_nodes':
            self.__num_nodes__ = item
        else:
            self[key] = item

```

`edge_index`的每一列定义一条边，其中第一行为边的起点，第二行为边的终点。这种表示方法被称为**COO格式（coordinate format）**，通常用于表示稀疏矩阵。PyG不是用稠密矩阵$\mathbf{A} \in \{ 0, 1 \}^{|\mathcal{V}| \times |\mathcal{V}|}$来持有邻接矩阵的信息，而是用仅存储邻接矩阵$\mathbf{A}$中非$0$元素的稀疏矩阵来表示图。

通常，一个图至少包含`x, edge_index, edge_attr, y, num_nodes`5个属性，当图包含其他属性时，通过指定额外的参数，我们可以让`Data`对象包含其他的属性：

```python
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes, other_attr=other_attr)
```

可以将一个`dict`对象转换为一个`Data`对象：

```python
graph_dict = {
    'x': x,
    'edge_index': edge_index,
    'edge_attr': edge_attr,
    'y': y,
    'num_nodes': num_nodes,
    'other_attr': other_attr
}
graph_data = Data.from_dict(graph_dict)
```

`from_dict`是一个类方法：

```python
@classmethod
def from_dict(cls, dictionary):
    r"""Creates a data object from a python dictionary."""
    data = cls()
    for key, item in dictionary.items():
        data[key] = item

    return data
```

注意：`graph_dict`中值的类型与大小的要求与`Data`类的构造函数的要求相同。

### `Data`对象转换成其他类型数据

也可以将`Data`对象转换为`dict`对象：

```python
def to_dict(self):
    return {key: item for key, item in self}
```

或转换为`namedtuple`

```python
def to_namedtuple(self):
    keys = self.keys
    DataTuple = collections.namedtuple('DataTuple', keys)
    return DataTuple(*[self[key] for key in keys])
```

### 获取`Data`对象属性

```python
x = graph_data['x']
```

### 设置`Data`对象属性

```python
graph_data['x'] = x
```

### 获取`Data`对象包含属性的关键字

```python
graph_data.keys()
```

### 对边排序并移除重复的边

```python
graph_data.coalesce()
```

### `Data`对象的其他性质

我们通过观察PyG中内置的一个图来查看`Data`对象的性质：

```python
from torch_geometric.datasets import KarateClub

dataset = KarateClub()
data = dataset[0]  # Get the first graph object.
print(data)
print('==============================================================')

# 获取图的一些信息
print(f'Number of nodes: {data.num_nodes}') # 节点数量
print(f'Number of edges: {data.num_edges}') # 边数量
print(f'Number of node features: {data.num_node_features}') # 节点属性的维度
print(f'Number of node features: {data.num_features}') # 同样是节点属性的维度
print(f'Number of edge features: {data.num_edge_features}') # 边属性的维度
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # 平均节点度
print(f'if edge indices are ordered and do not contain duplicate entries.: {data.is_coalesced()}') # 是否边是有序的同时不含有重复的边
print(f'Number of training nodes: {data.train_mask.sum()}') # 用作训练集的节点
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}') # 用作训练集的节点的数量
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}') # 此图是否包含孤立的节点
print(f'Contains self-loops: {data.contains_self_loops()}')  # 此图是否包含自环的边
print(f'Is undirected: {data.is_undirected()}')  # 此图是否是无向图
```

## 三、`Dataset`类——PyG中图数据集的表示及其使用

PyG内置了大量常用的基准数据集，接下来我们用PyG中包含的`Planetoid`数据集来学习PyG中图数据集的表示及其使用。

在PyG中，初始化一个数据集是简单直接的。初始化一个PyG内置的数据集，将自动下载其原始文件，并将其处理成包含`Data`对象的`Dataset`对象。

```python
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/dataset/Cora', name='Cora')
# Cora()

len(dataset)
# 1

dataset.num_classes
# 7

dataset.num_node_features
# 1433
```

可以看到该数据集只有一个图，包含7个分类任务，节点的属性为1433维度。

```python
data = dataset[0]
# Data(edge_index=[2, 10556], test_mask=[2708],
#         train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])

data.is_undirected()
# True

data.train_mask.sum().item()
# 140

data.val_mask.sum().item()
# 500

data.test_mask.sum().item()
# 1000
```

现在我们看到该数据集包含的唯一的图，有2708个节点，节点特征为1433维，有10556条边，有140个用作训练集的节点，有500个用作验证集的节点，有1000个用作测试集的节点。PyG内置的其他数据集，请小伙伴一一试验，以观察不同数据集的不同。

## 结语

通过此实践环节，我们学习了基于`Data`类的简单图的表示和使用，以及基于`Dataset`类的PyG内置数据集的表示和使用。在后面的内容中，我们将学习如何基于`Data`类表示复杂图，以及如何构造自己的数据集类并使用。
