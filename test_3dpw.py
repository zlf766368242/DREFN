import argparse
import torch
from dataset.dataloader import Pose3dPW3D
from model.MAGCNmodelGRU_3dpw_short import MultiScaleModel
from model.MAGCNmodelGRU_3dpw_long import MultiScaleModel2
from torch import nn, optim
import numpy as np
import random
import yaml

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def lr_decay(optimizer, lr_now, gamma):
    lr_new = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

def main():
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0,1000)
        setup_seed(seed)
    print('The seed is :',seed)

    past_length = args.past_length
    future_length = args.future_length

    if args.debug:
        dataset_train = Pose3dPW3D(input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale, debug=True)
    else:
        dataset_train = Pose3dPW3D(input_n=args.past_length, output_n=args.future_length, split=0, scale=args.scale)
   
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    dataset_test = Pose3dPW3D(input_n=args.past_length, output_n=args.future_length, split=1, scale=args.scale)
    loaders_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                num_workers=8)

    dim_used = dataset_train.dim_used
    if args.task == 'short':
        model = MultiScaleModel(
        in_dim=3,
        h_dim=args.nf,
        past_timestep=args.past_length,
        future_timestep=args.future_length,
        kernel_size=args.kernel_size,
        graph_args_j=args.graph_args_j,
        graph_args_b=args.graph_args_b)
        weights_path_j = '/root/dataDisk/DREFN/pretrain/3dpw_short_model3dpw_ckpt_1.pth'
    else:
        model = MultiScaleModel2(
        in_dim=3,
        h_dim=args.nf,
        past_timestep=args.past_length,
        future_timestep=args.future_length,
        kernel_size=args.kernel_size,
        graph_args_j=args.graph_args_j,
        graph_args_b=args.graph_args_b)
        weights_path_j = '/root/dataDisk/DREFN/pretrain/3dpw_long_model3dpw_ckpt_1.pth'
    model = model.cuda()
    checkpoint = torch.load(weights_path_j)
    if 'state_dict' in checkpoint:
        pretrained_weights_j = checkpoint['state_dict']
    else:
        pretrained_weights_j = checkpoint
    pretrained_weights_j = {k: v for k, v in pretrained_weights_j.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_weights_j)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.eval()
    mpjpe = test(model, optimizer, 0, ('all', loaders_test), dim_used, backprop=False)
    avg_mpjpe = mpjpe
    print('avg mpjpe:',avg_mpjpe)

    return


def train(model, optimizer, epoch, loader, dim_used=[], backprop=True, x=0.4):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, length, _ = data[0].size()
        data = [d.to(device) for d in data]
        loc, vel, loc_end, _, item = data
        optimizer.zero_grad()
        all_traj = torch.cat([loc, loc_end], dim=2)
        all_traj = all_traj.float()

        loc_pred, denoised_pred = model(all_traj)
        loss1 = torch.mean(torch.norm(loc_pred - loc_end, dim=-1))

        if args.denoise_mode == 'all':
            loss2 = torch.mean(torch.norm(denoised_pred - all_traj, dim=-1))
        elif args.denoise_mode == 'past':
            loss2 = torch.mean(torch.norm(denoised_pred - loc, dim=-1))
        elif args.denoise_mode == 'future':
            loss2 = torch.mean(torch.norm(denoised_pred - loc_end, dim=-1))
        else:
            raise ValueError("args.denoise_mode")
        loss = x * loss1 + (2 - x) * loss2

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size

    print('%s epoch %d avg loss: %.5f' % ('train', epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']


def test(model, optimizer, epoch, act_loader, dim_used=[], backprop=False):
    act, loader = act_loader[0], act_loader[1]

    model.eval()

    validate_reasoning = False
    if validate_reasoning:
        acc_list = [0] * args.n_layers
    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'ade': 0}

    output_n = args.future_length
    if output_n == 12:
        eval_frame = [2, 5, 8, 11]
    elif output_n == 30:
        eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    elif output_n == 10:
        eval_frame = [5, 7, 9]

    t_3d = np.zeros(len(eval_frame))

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            batch_size, n_nodes, length, _ = data[0].size()
            data = [d.to(device) for d in data]
            loc, vel, loc_end, loc_end_ori, _ = data
            pred_length = loc_end.shape[2]
            optimizer.zero_grad()
            loc_end_fake = torch.zeros_like(loc_end)
            all_traj = torch.cat([loc, loc_end_fake], dim=2)
            all_traj = all_traj.float()

            loc_pred, _ = model(all_traj)

            loc_end_ori = loc_end_ori.float()
            pred_3d = loc_end_ori.clone()
            loc_pred = loc_pred.transpose(1, 2)
            loc_pred = loc_pred.contiguous().view(batch_size, loc_end.shape[2], n_nodes * 3)

            pred_3d[:, :, dim_used] = loc_pred

            pred_p3d = pred_3d.contiguous().view(batch_size, pred_length, -1, 3)  # [:, input_n:, :, :]
            targ_p3d = loc_end_ori.contiguous().view(batch_size, pred_length, -1, 3)  # [:, input_n:, :, :]

            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += torch.mean(torch.norm(
                    targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                    1)).item() * batch_size

            res['counter'] += batch_size

    t_3d *= args.scale
    N = res['counter']
    actname = "{0: <14} |".format(act)

    return t_3d / N


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=-1, metavar='S', help='random seed (default: -1)')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument('--model_save_dir', type=str, default='ckpt', help='dir to save model')
    parser.add_argument("--model_save_name", type=str, default="default")
    parser.add_argument("--task", type=str, default="short")
    args = parser.parse_args()

    if args.task == 'short':
        with open('cfg/3dpw_short.yml', 'r') as f:
            yml_arg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open('cfg/3dpw_long.yml', 'r') as f:
            yml_arg = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")
    main()
