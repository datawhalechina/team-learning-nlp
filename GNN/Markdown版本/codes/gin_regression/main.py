import os
import torch
import argparse
from tqdm import tqdm
from ogb.lsc import PCQM4MEvaluator
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pcqm4m_data import MyPCQM4MDataset
from gin_graph import GINGraphPooling

from torch.utils.tensorboard import SummaryWriter

def parse_args():

    parser = argparse.ArgumentParser(description='Graph data miming with GNN')
    parser.add_argument('--task_name', type=str, default='GINGraphPooling',
                        help='task name')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--drop_ratio', type=float, default=0.,
                        help='dropout ratio (default: 0.)')
    parser.add_argument('--save_test', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--dataset_root', type=str, default="dataset",
                        help='dataset root')
    args = parser.parse_args()

    return args


def prepartion(args):
    save_dir = os.path.join('saves', args.task_name)
    if os.path.exists(save_dir):
        for idx in range(1000):
            if not os.path.exists(save_dir + '=' + str(idx)):
                save_dir = save_dir + '=' + str(idx)
                break

    args.save_dir = save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.output_file = open(os.path.join(args.save_dir, 'output'), 'a')
    print(args, file=args.output_file, flush=True)


def train(model, device, loader, optimizer, criterion_fn):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = criterion_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            pred = model(batch).view(-1,)
            y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)
    return y_pred


def main(args):
    prepartion(args)
    nn_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    # automatic dataloading and splitting
    dataset = MyPCQM4MDataset(root=args.dataset_root)
    split_idx = dataset.get_idx_split()
    train_data = dataset[split_idx['train']]
    valid_data = dataset[split_idx['valid']]
    test_data = dataset[split_idx['test']]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()
    criterion_fn = torch.nn.MSELoss()

    device = args.device

    model = GINGraphPooling(**nn_params).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}', file=args.output_file, flush=True)
    print(model, file=args.output_file, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    writer = SummaryWriter(log_dir=args.save_dir)
    not_improved = 0
    best_valid_mae = 9999
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch), file=args.output_file, flush=True)
        print('Training...', file=args.output_file, flush=True)
        train_mae = train(model, device, train_loader, optimizer, criterion_fn)

        print('Evaluating...', file=args.output_file, flush=True)
        valid_mae = eval(model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae}, file=args.output_file, flush=True)

        writer.add_scalar('valid/mae', valid_mae, epoch)
        writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.save_test:
                print('Saving checkpoint...', file=args.output_file, flush=True)
                checkpoint = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae, 'num_params': num_params
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt'))
                print('Predicting on test data...', file=args.output_file, flush=True)
                y_pred = test(model, device, test_loader)
                print('Saving test submission file...', file=args.output_file, flush=True)
                evaluator.save_test_submission({'y_pred': y_pred}, args.save_dir)

            not_improved = 0
        else:
            not_improved += 1
            if not_improved == args.early_stop:
                print(f"Have not improved for {not_improved} epoches.", file=args.output_file, flush=True)
                break

        scheduler.step()
        print(f'Best validation MAE so far: {best_valid_mae}', file=args.output_file, flush=True)

    writer.close()
    args.output_file.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
 