import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from gin_node import GINNodeEmbedding


class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="sum"):
        """GIN Graph Pooling Module

        此模块首先采用GINNodeEmbedding模块对图上每一个节点做嵌入，然后对节点嵌入做池化得到图的嵌入，最后用一层线性变换得到图的最终的表示（graph representation）。

        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表示的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(output, min=0, max=50)
