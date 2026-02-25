from layer import GraphLayer
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

class MHCN(torch.nn.Module):
    def __init__(self, args, dataset):
        super(MHCN, self).__init__()

        self.drop = args.drop

        self.horizontal_11 = GraphLayer(dataset.num_edge_features, dataset.num_features, args.num_dim)
        self.horizontal_12 = GraphLayer(dataset.num_edge_features, args.num_dim, args.num_dim)
        self.horizontal_13 = GraphLayer(dataset.num_edge_features, args.num_dim, args.num_dim)

        self.vertical_1 = torch.nn.Linear(args.num_dim + dataset.num_edge_features, args.num_dim)

        self.horizontal_21 = GraphLayer(dataset.num_node_features, args.num_dim, args.num_dim)
        self.horizontal_22 = GraphLayer(dataset.num_node_features, args.num_dim, args.num_dim)

        self.vertical_2 = torch.nn.Linear(args.num_dim + dataset.num_node_features, args.num_dim)

        self.horizontal_31 = GraphLayer(dataset.num_edge_features * dataset.num_edge_features, args.num_dim, args.num_dim)
        self.horizontal_32 = GraphLayer(dataset.num_edge_features * dataset.num_edge_features, args.num_dim, args.num_dim)

        self.fc1 = torch.nn.Linear(3 * args.num_dim, args.num_dim)
        self.fc2 = torch.nn.Linear(args.num_dim, args.num_dim // 2)
        self.fc3 = torch.nn.Linear(args.num_dim // 2, dataset.num_classes)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data, args):

        data.x = F.dropout(data.x, p=self.drop, training=self.training)
        data.x = F.elu(self.horizontal_11(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.horizontal_12(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.horizontal_13(data.x, data.edge_index, data.edge_attr))

        x_1 = scatter_add(data.x, data.batch, dim=0, dim_size=data.num_graphs)

        data.x = torch.cat([avg_pool(data.x, data.assi_12), data.assi_edge_2], dim=1)
        data.x = self.vertical_1(data.x)


        data.x = F.dropout(data.x, p=self.drop, training=self.training)
        data.x = F.elu(self.horizontal_21(data.x, data.edge_index_2, data.edge_attr_2))
        data.x = F.elu(self.horizontal_22(data.x, data.edge_index_2, data.edge_attr_2))

        x_2 = scatter_mean(data.x, data.batch_2, dim=0, dim_size=data.num_graphs)


        data.x = torch.cat([avg_pool(data.x, data.assi_23), data.assi_edge_3], dim=1)
        data.x = self.vertical_2(data.x)
        data.x = avg_pool(data.x, data.assi_23)


        data.x = F.dropout(data.x, p=self.drop, training=self.training)
        data.x = F.elu(self.horizontal_31(data.x, data.edge_index_3, data.edge_attr_3))
        data.x = F.elu(self.horizontal_32(data.x, data.edge_index_3, data.edge_attr_3))

        x_3 = scatter_mean(data.x, data.batch_3, dim=0, dim_size=data.num_graphs)

        x = torch.cat([x_1, x_2, x_3], dim=1)


        if args.no_train:
            x = x.detach()

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



def avg_pool(x, assignment):
    row, col = assignment
    return scatter_mean(x[row], col, dim=0)