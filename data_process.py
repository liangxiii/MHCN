import torch
import torch.utils.data
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform


def combine_attr_vectors(onehot1, onehot2):
    index1 = torch.argmax(onehot1).item()
    index2 = torch.argmax(onehot2).item()
    combined_index = index1 * len(onehot1) + index2
    combined_onehot = torch.zeros(len(onehot1) * len(onehot2))
    combined_onehot[combined_index] = 1
    return combined_onehot


def find_edge_indices_vectorized(edges, edge_index):
    edges_t = edges.t()
    edges_expanded = edges_t.unsqueeze(2)
    edge_index_expanded = edge_index.unsqueeze(1)
    matches = (edges_expanded == edge_index_expanded).all(dim=0)
    result = torch.full((edges.shape[0],), -1, dtype=torch.long)
    col_indices = matches.int().argmax(dim=1)
    mask = matches.any(dim=1)
    result[mask] = col_indices[mask]
    return result

@functional_transform('tohighorder')
class ToHighOrder(BaseTransform):
    def __init__(self):
        pass

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        tuples_list = map(tuple, data.edge_index.t().tolist())
        tuples_set = {frozenset(item) for item in tuples_list}
        tuples_list = [set(item) for item in tuples_set]

        #生成二阶边
        triples_set = set()
        edge_index_2 = []
        edge_attr_index_2 = []
        for i in range(len(tuples_list)):
            for j in range(len(tuples_list)):
                union_set = tuples_list[i] | tuples_list[j]
                if len(union_set) == 3:
                    edge_index_2.append([i, j])
                    edge_attr_index_2.append(list(tuples_list[i] & tuples_list[j]))
                    triples_set.add(frozenset(union_set))

        edge_attr_index_2 = [item[0] for item in edge_attr_index_2]
        data.edge_attr_2 = data.x[edge_attr_index_2]


        triples_list = [set(item) for item in triples_set]

        edge_index_3 = []
        edge_attr_index_3 = []
        for i in range(len(triples_list)):
            for j in range(len(triples_list)):
                common_nodes = triples_list[i] & triples_list[j]
                if len(common_nodes) == 2:
                    edge_index_3.append([i, j])
                    edge_attr_index_3.append(list(common_nodes))


        if len(edge_attr_index_3) != 0:
            edge_attr_index_3 = torch.tensor(edge_attr_index_3)
            head = edge_attr_index_3[:, 0]
            tail = edge_attr_index_3[:, 1]
            posi_ht = (data.edge_index[0, None, :] == head[:, None]) & (data.edge_index[1, None, :] == tail[:, None])
            posi_th = (data.edge_index[0, None, :] == tail[:, None]) & (data.edge_index[1, None, :] == head[:, None])
            indices_ht = torch.argmax(posi_ht.int(), dim=1)
            indices_th = torch.argmax(posi_th.int(), dim=1)
            edge_attr_3 = [combine_attr_vectors(data['edge_attr'][indices_ht[i]], data['edge_attr'][indices_th[i]]) for i in range(len(indices_ht))]
            data.edge_attr_3 = torch.stack(edge_attr_3, dim=0)
        else:
            data.edge_attr_3 = torch.tensor([], dtype=torch.float)

        tuples = torch.tensor([list(item) for item in tuples_list], dtype=torch.long)
        data.assi_12 = torch.stack((tuples.flatten(), torch.arange(len(tuples_list) * 2, dtype=torch.long) // 2), dim=1)
        data.edge_index_2 = torch.tensor(edge_index_2, dtype=torch.long).T
        data.assi_edge_2 = find_edge_indices_vectorized(tuples, data.edge_index)
        data.assi_edge_2 = data.edge_attr[data.assi_edge_2]

        triples = [list(item) for item in triples_list]
        assi_23 = []
        assi_edge_3 = []
        for i in range(len(triples)):
            posi_01 = [index for index, item in enumerate(tuples_list) if item == {triples[i][0], triples[i][1]}]
            posi_12 = [index for index, item in enumerate(tuples_list) if item == {triples[i][1], triples[i][2]}]
            posi_02 = [index for index, item in enumerate(tuples_list) if item == {triples[i][0], triples[i][2]}]
            assi_23.append([posi_01[0], i]) if posi_01 else None
            assi_23.append([posi_12[0], i]) if posi_12 else None
            assi_23.append([posi_02[0], i]) if posi_02 else None
            n_1 = tuples_list[posi_01[0]] & tuples_list[posi_12[0]] if posi_01 and posi_12 else set()
            n_2 = tuples_list[posi_12[0]] & tuples_list[posi_02[0]] if posi_12 and posi_02 else set()
            n_3 = tuples_list[posi_01[0]] & tuples_list[posi_02[0]] if posi_01 and posi_02 else set()
            temp = 0
            temp = temp + data.x[n_1.pop()] if n_1 else temp
            temp = temp + data.x[n_2.pop()] if n_2 else temp
            temp = temp + data.x[n_3.pop()] if n_3 else temp
            assi_edge_3.append(temp)
        data.assi_edge_3 = torch.stack(assi_edge_3) if assi_edge_3 else torch.tensor([], dtype=torch.float)
        data.assi_23 = torch.tensor(assi_23, dtype=torch.long)


        triples = torch.tensor(triples, dtype=torch.long)
        data.assi_13 = torch.stack((triples.flatten(), torch.arange(len(triples_list) * 3, dtype=torch.long) // 3), dim=1)
        data.edge_index_3 = torch.tensor(edge_index_3, dtype=torch.long).T

        return data




def collate(data_list):
    keys = data_list[0].keys()
    assert 'batch' not in keys

    batch = Batch()
    for key in keys:
        batch[key] = []
    batch.batch = []
    batch.batch_2 = []
    batch.batch_3 = []

    keys.remove('edge_index')
    props = ['edge_index_2', 'edge_index_3', 'assi_12', 'assi_13', 'assi_23', 'assi_edge_3']
    keys = [x for x in keys if x not in props]

    cumsum_1 = N_1 = cumsum_2 = N_2 = cumsum_3 = N_3 = 0
    for i, data in enumerate(data_list):
        for key in keys:
            batch[key].append(data[key])

        N_1 = data.num_nodes
        batch.edge_index.append(data.edge_index + cumsum_1)
        batch.batch.append(torch.full((N_1, ), i, dtype=torch.long))

        data.assi_12 = data.assi_12.T
        N_2 = data.assi_12[1].max().item() + 1
        batch.edge_index_2.append(data.edge_index_2 + cumsum_2)
        batch.assi_12.append(data.assi_12 + torch.tensor([[cumsum_1], [cumsum_2]]))
        batch.batch_2.append(torch.full((N_2, ), i, dtype=torch.long))

        if data.edge_index_3.numel() > 0:
            data.assi_23 = data.assi_23.T
            data.assi_13 = data.assi_13.T
            N_3 = data.assi_23[1].max().item() + 1
            batch.edge_index_3.append(data.edge_index_3 + cumsum_3)
            batch.assi_13.append(data.assi_13 + torch.tensor([[cumsum_1], [cumsum_3]]))
            batch.assi_23.append(data.assi_23 + torch.tensor([[cumsum_2], [cumsum_3]]))
            batch.batch_3.append(torch.full((N_3, ), i, dtype=torch.long))
            batch.assi_edge_3.append(data.assi_edge_3)
            cumsum_3 += N_3

        cumsum_1 += N_1
        cumsum_2 += N_2


    batch['x'] = torch.cat(batch['x'], dim=0)
    batch['y'] = torch.cat(batch['y'], dim=0)
    batch['edge_index'] = torch.cat(batch['edge_index'], dim=-1)
    batch['edge_attr'] = torch.cat(batch['edge_attr'], dim=0)

    batch['edge_index_2'] = torch.cat(batch['edge_index_2'], dim=-1)
    batch['edge_attr_2'] = torch.cat(batch['edge_attr_2'], dim=0)
    batch['assi_12'] = torch.cat(batch['assi_12'], dim=-1)
    batch['assi_edge_2'] = torch.cat(batch['assi_edge_2'], dim=0)

    batch['edge_index_3'] = torch.cat(batch['edge_index_3'], dim=-1)
    batch['edge_attr_3'] = torch.cat(batch['edge_attr_3'], dim=0)
    batch['assi_13'] = torch.cat(batch['assi_13'], dim=-1)
    batch['assi_23'] = torch.cat(batch['assi_23'], dim=-1)
    batch['assi_edge_3'] = torch.cat(batch['assi_edge_3'], dim=0)


    batch.batch = torch.cat(batch.batch, dim=-1)
    batch.batch_2 = torch.cat(batch.batch_2, dim=-1)
    batch.batch_3 = torch.cat(batch.batch_3, dim=-1)



    return batch.contiguous()


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super(DataLoader, self).__init__(dataset, collate_fn=collate, **kwargs)

class MyFilter(object):
    def __call__(self, data):
        return True

class MyPreTransform(object):
    def __call__(self, data):
        return data