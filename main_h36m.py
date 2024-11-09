import argparse
import torch
from dataset.dataloader import H36motion3D
from model.MAGCNmodelGRU_h36m_short import MultiScaleModel
from model.MAGCNmodelGRU_h36m_long import MultiScaleModel2
from torch import nn, optim
import json
import time
import numpy as np
import math
import random
import yaml
import os

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
        seed = random.randint(0, 1000)
        setup_seed(seed)

    past_length = args.past_length
    future_length = args.future_length

    if args.debug:
        dataset_train = H36motion3D(actions='walking', input_n=args.past_length, output_n=args.future_length, split=0,
                                    scale=args.scale)
    else:
        dataset_train = H36motion3D(actions='all', input_n=args.past_length, output_n=args.future_length, split=0,
                                    scale=args.scale)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
    loaders_test = {}
    for act in acts:
        dataset_test = H36motion3D(actions=act, input_n=args.past_length, output_n=args.future_length, split=2,
                                   scale=args.scale)
        loaders_test[act] = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                        drop_last=False,
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
    else:
        model = MultiScaleModel2(
            in_dim=3,
            h_dim=args.nf,
            past_timestep=args.past_length,
            future_timestep=args.future_length,
            kernel_size=args.kernel_size,
            graph_args_j=args.graph_args_j,
            graph_args_b=args.graph_args_b)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'losess': []}
    best_test_loss = 1e8
    best_ade = 1e8
    best_epoch = 0
    lr_now = args.lr

    for epoch in range(0, args.epochs):
        if args.apply_decay:
            if epoch % args.epoch_decay == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)
        train(model, optimizer, epoch, loader_train, dim_used)
        if epoch % args.test_interval == 0:
            if args.future_length == 25 or args.future_length == 13:
                avg_mpjpe = np.zeros((2))
            elif args.future_length == 15:
                avg_mpjpe = np.zeros((2))
            else:
                avg_mpjpe = np.zeros((4))
            for act in acts:
                mpjpe = test(model, optimizer, epoch, (act, loaders_test[act]), dim_used, backprop=False)
                avg_mpjpe += mpjpe

            avg_mpjpe = avg_mpjpe / len(acts)
            print('avg mpjpe:', avg_mpjpe)
            avg_avg_mpjpe = np.mean(avg_mpjpe)

            if avg_avg_mpjpe < best_test_loss:
                best_test_loss = avg_avg_mpjpe
                best_all_test_loss = avg_mpjpe
                best_epoch = epoch
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                if args.model_save_name == 'default':
                    if args.future_length == 25 or args.future_length == 13:
                        file_path = os.path.join(args.model_save_dir, 'h36m_ckpt_long_best.pth')
                    else:
                        file_path = os.path.join(args.model_save_dir, 'h36m_ckpt_best.pth')
                else:
                    if args.future_length == 25 or args.future_length == 13:
                        file_path = os.path.join(args.model_save_dir,
                                                 args.model_save_name + '_h36m_ckpt_long_best.pth')
                    else:
                        file_path = os.path.join(args.model_save_dir, args.model_save_name + '_h36m_ckpt_best.pth')
                torch.save(state, file_path)

            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            if args.model_save_name == 'default':
                if args.future_length == 25 or args.future_length == 13:
                    file_path = os.path.join(args.model_save_dir, 'h36m_ckpt_long_' + str(epoch) +str(avg_avg_mpjpe)+ '.pth')
                else:
                    file_path = os.path.join(args.model_save_dir, 'h36m_ckpt_' + str(epoch) +str(avg_avg_mpjpe)+ '.pth')
            else:
                if args.future_length == 25 or args.future_length == 13:
                    file_path = os.path.join(args.model_save_dir,
                                             args.model_save_name + '_h36m_ckpt_long_' + str(epoch) +str(avg_avg_mpjpe)+ '.pth')
                else:
                    file_path = os.path.join(args.model_save_dir,
                                             args.model_save_name + '_h36m_ckpt_' + str(epoch) +str(avg_avg_mpjpe)+ '.pth')
            torch.save(state, file_path)

            print("Best Test Loss: %.5f \t Best epoch %d" % (best_test_loss, best_epoch))
            print("Best AVG loss:", best_all_test_loss)

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
        loc_pred,denoised_pred = model(all_traj)

        loss1 = torch.mean(torch.norm(loc_pred-loc_end,dim=-1))

        if args.denoise_mode == 'all':
            loss2 = torch.mean(torch.norm(denoised_pred-all_traj,dim=-1))
        elif args.denoise_mode == 'past':
            loss2 = torch.mean(torch.norm(denoised_pred-loc,dim=-1))
        elif args.denoise_mode == 'future':
            loss2 = torch.mean(torch.norm(denoised_pred-loc_end,dim=-1))
        else:
            raise ValueError("args.denoise_mode")
        loss = x*loss1 + (2-x)*loss2

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
    res = {'epoch': epoch, 'loss': 0, 'counter': 0}

    output_n = args.future_length
    if output_n == 25:
        eval_frame = [13, 24]
    elif output_n == 15:
        eval_frame = [3, 14]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    elif output_n == 13:
        eval_frame = [6, 12]
    t_3d = np.zeros(len(eval_frame))

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            batch_size, n_nodes, length, _ = data[0].size()
            data = [d.to(device) for d in data]
            loc, vel, loc_end, loc_end_ori, _ = data
            loc_start = loc[:, :, -1:]
            pred_length = loc_end.shape[2]

            optimizer.zero_grad()

            loc_end_fake = torch.zeros_like(loc_end)
            all_traj = torch.cat([loc, loc_end_fake], dim=2)
            loc_pred = model.predict(all_traj)  # (B,N,T,3)

            pred_3d = loc_end_ori.clone()
            loc_pred = loc_pred.transpose(1, 2)
            loc_pred = loc_pred.contiguous().view(batch_size, loc_end.shape[2], n_nodes * 3)
            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

            pred_3d[:, :, dim_used] = loc_pred
            pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
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
    if args.future_length == 25 or args.future_length == 13:
        print('Act: {},  ErrT: {:.3f} {:.3f}, TestError {:.4f}' \
              .format(actname,
                      float(t_3d[0]) / N, float(t_3d[1]) / N, 
                      float(t_3d.mean()) / N))
    elif args.future_length == 15:
        print('Act: {},  ErrT: {:.3f} {:.3f}, TestError {:.4f}' \
              .format(actname,
                      float(t_3d[0]) / N, float(t_3d[1]) / N,
                      float(t_3d.mean()) / N))
    else:
        print('Act: {},  ErrT: {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}' \
              .format(actname,
                      float(t_3d[0]) / N, float(t_3d[1]) / N, float(t_3d[2]) / N, float(t_3d[3]) / N,
                      float(t_3d.mean()) / N))

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
        with open('cfg/h36m_short.yml', 'r') as f:
            yml_arg = yaml.load(f,Loader=yaml.FullLoader)
    else:
        with open('cfg/h36m_long.yml', 'r') as f:
            yml_arg = yaml.load(f,Loader=yaml.FullLoader)

    parser.set_defaults(**yml_arg)
    args = parser.parse_args()
    args.cuda = True

    device = torch.device("cuda" if args.cuda else "cpu")
    main()




