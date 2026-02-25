import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add


class GraphLayer(torch.nn.Module):
    def __init__(self, num_edge_types, in_channels, out_channels, dropout=0):
        super(GraphLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.edge_type_weights = Parameter(torch.Tensor(num_edge_types, in_channels, out_channels))
        self.self_loop_weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_channels)
        self.edge_type_weights.data.uniform_(-bound, bound)
        self.self_loop_weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() > 0:
            target_idx, source_idx = edge_index
            edge_type_weights = torch.einsum('ij,jkl->ikl', edge_attr, self.edge_type_weights)
            output = torch.matmul(x[source_idx].unsqueeze(1), edge_type_weights).squeeze(1)
            output = F.dropout(output, self.dropout, training=self.training)
            output = scatter_add(output, target_idx, dim=0, dim_size=x.size(0))
            deg = scatter_add(x.new_ones((target_idx.size())), target_idx, dim=0, dim_size=x.size(0))
            output = output / deg.unsqueeze(-1).clamp(min=1)
            output = output + torch.mm(x, self.self_loop_weight)
        else:
            output = torch.mm(x, self.self_loop_weight)


        output = output  + self.bias

        return output
