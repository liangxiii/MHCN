import os.path as osp
import argparse
import random
import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
from model import MHCN
from data_process import DataLoader, ToHighOrder, MyPreTransform, MyFilter




def train(args, loader, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output= model(data, args)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def test(args, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data, args)
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def k_fold(dataset, folds=10):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12306)
    train_indices, test_indices = [], []
    ys = [graph.y.item() for graph in dataset]
    for train, test in skf.split(torch.zeros(len(dataset)), ys):
        train_indices.append(torch.tensor(train, dtype=torch.long))
        test_indices.append(torch.tensor(test, dtype=torch.long))
    return train_indices, test_indices

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PTC_MR', help="MUTAG or PTC_MR or PROTEINS")
    parser.add_argument('--no-train', default=False)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--num_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--drop", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    seed_it(5799)

    BATCH = args.batch_size
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
    dataset = TUDataset(path, name=args.dataset, pre_transform=T.Compose([MyPreTransform(), ToHighOrder()]),
        pre_filter=MyFilter(), force_reload=True)

    perm = torch.randperm(len(dataset), dtype=torch.long)
    dataset = dataset[perm]

    model = MHCN(args, dataset).to(device)
    k_fold_indices = k_fold(dataset, args.folds)

    final_acc = []
    for fold, (train_idx, test_idx) in enumerate(zip(*k_fold_indices)):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=15, min_lr=0.00001)

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]

        test_loader = DataLoader(test_dataset, batch_size=BATCH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

        print('---------------- Split {} ----------------'.format(fold))

        test_acc, acc = 0, 0
        for epoch in range(1, args.epoch + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train(args, train_loader, optimizer)
            acc = test(args, test_loader)
            if acc >= test_acc:
                test_acc = acc
            if epoch % 5 == 0:
                print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f},'
                      'Test Acc: {:.7f}'.format(epoch, lr, train_loss, test_acc))
        final_acc.append(test_acc)
    final_acc = torch.tensor(final_acc)
    print('---------------- Final Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(final_acc.mean(), final_acc.std()))