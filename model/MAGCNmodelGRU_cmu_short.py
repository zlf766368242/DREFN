import torch
from torch import nn
from torch.nn import functional as F
from model.graph_cmu import *
class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_num,
                 t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True):
        super().__init__()
        self.k_num = k_num
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*(k_num),
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A_skl):
        x = self.conv(x)
        b, kc, t, v = x.size()
        x = x.view(b, self.k_num, kc // self.k_num, t, v)
        A_all = A_skl
        x = torch.einsum('bkctv,kvw->bctw', (x, A_all))
        return x.contiguous()
class St_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, stride=1, dropout=0.,residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0]-1) // 2, 0)
        self.gcn = SpatialConv(in_channels,128, kernel_size[1], t_kernel_size)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A_skl):
        res = self.residual(x)
        x = self.gcn(x, A_skl)
        x = self.tcn(x) + res
        return self.relu(x)

class Encoderj(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, stride=1, dropout=0.,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        self.gcn = SpatialConv(in_channels, out_channels, kernel_size[1], t_kernel_size)
        self.gru = nn.GRU(input_size= out_channels, hidden_size=out_channels, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, A_j,mask):
        B, V, T, C = x.shape
        x = x * mask
        x = x.permute(0, 3, 2, 1)
        res = self.residual(x)
        res = res.permute(0, 3, 2, 1).contiguous()
        x = self.gcn(x, A_j)
        x = self.bn1(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(B * V, T, -1)
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = x.view(B, V, T, -1)
        return self.relu(x + res)

class Encoderb(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, stride=1, dropout=0.,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        self.gcn = SpatialConv(in_channels, out_channels, kernel_size[1], t_kernel_size)
        self.gru = nn.GRU(input_size= out_channels, hidden_size=out_channels, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, A_b, mask):
        B, V, T, C = x.shape
        x = x * mask
        x = x.permute(0, 3, 2, 1)
        res = self.residual(x)
        res = res.permute(0, 3, 2, 1).contiguous()  
        x = self.gcn(x, A_b)
        x = self.bn1(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(B * V, T, -1)
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = x.view(B, V, T, -1)
        return self.relu(x + res)
class STblockj(nn.Module):
    def __init__(self, 
                in_dim,
                h_dim, 
                past_timestep, 
                future_timestep,
                kernel_size,
                graph_args_j,
                decoder_dim=64, 
                num_heads=8, 
                encoder_depth=5,
                dim_per_head=64,
                mlp_dim=64,
                noise_dev=0.6,
                part_noise=True,
                denoise_mode='past',
                part_noise_ratio=0.36,
                add_joint_token=True,
                n_agent=25,
                dropout=0.,
                range_noise_dev=False):
        super(STblockj, self).__init__()
        self.all_timesteps = past_timestep + future_timestep
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))
        self.past_timestep = past_timestep
        self.future_timestep = future_timestep
        self.encoder_depth = encoder_depth
        self.patch_embed = nn.Linear(in_dim, h_dim)
        self.decoder_pos_embed = nn.Embedding(self.all_timesteps, decoder_dim)
        self.decoder_agent_embed = nn.Embedding(n_agent, decoder_dim)
        self.patch_embed = nn.Linear(32, h_dim)
        self.noise_dev = noise_dev
        self.part_noise = part_noise
        self.part_noise_ratio = part_noise_ratio
        self.denoise_mode = denoise_mode
        self.add_joint_token = add_joint_token
        self.agent_embed = nn.Parameter(torch.randn(1, n_agent, 1, h_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.all_timesteps, h_dim))
        self.range_noise_dev = range_noise_dev
        self.graph_j = Graph_J(**graph_args_j)
        A_j = torch.tensor(self.graph_j.A_j, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_j', A_j)
        self.emul_s1 = nn.ParameterList([nn.Parameter(torch.ones(self.A_j.size())) for i in range(1)])
        self.eadd_s1 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_j.size())) for i in range(1)])
        self.encoder = []
        self.decoder = []
        for _ in range(self.encoder_depth):
            self.encoder.append(STTrans(self.all_timesteps,h_dim,depth=1,mlp_dim=mlp_dim,num_heads=num_heads,dim_per_head=dim_per_head,dropout=dropout))
            self.decoder.append(STTrans(self.all_timesteps,h_dim,depth=1,mlp_dim=mlp_dim,num_heads=num_heads,dim_per_head=dim_per_head,dropout=dropout))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.activite = Encoderj(in_dim, 32, kernel_size)
        self.st_gcn1 = St_gcn(decoder_dim, 64, kernel_size, 1, 1)
    
    def forward(self,all_traj):
        B,N,T = all_traj.shape[0],all_traj.shape[1],all_traj.shape[2]
        all_traj = all_traj.view(B,N,T,3)
        act_mask = torch.zeros((B, N, T, 3)).type_as(all_traj)
        act_mask[:, :, :self.past_timestep,:] = 1.
        expanded_act_mask = act_mask
        all_traj = self.activite(all_traj, self.A_j * self.emul_s1[0] + self.eadd_s1[0], expanded_act_mask)
        
        batch_ind = torch.arange(B)[:,None,None].cuda()
        agent_ind = torch.arange(N)[None,:,None].cuda()
        ordinary_mask = torch.zeros((B,N,T)).type_as(all_traj)
        ordinary_mask[:,:,:self.past_timestep] = 1.

        if self.range_noise_dev:
            noise_dev = np.random.uniform(low=0.1, high=1.0)
        else:
            noise_dev = self.noise_dev
        denoise_mask = torch.zeros((B,N,T)).type_as(all_traj)
        denoise_mask[:,:,:self.past_timestep] = 1.
        noise = torch.from_numpy(np.random.normal(loc=0., scale=noise_dev, size=(B,N,T,32))).type_as(all_traj)
        if self.part_noise:
            noise_mask = torch.rand(B,N,T).cuda()
            noise_mask = (noise_mask < self.part_noise_ratio).type_as(all_traj)
            noise = noise * noise_mask[:,:,:,None].repeat(1,1,1,32)
        all_traj_noise = all_traj + noise

        decoded_tokens = self.mask_forward(all_traj,ordinary_mask)
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous()  
        decoded_tokens = self.st_gcn1(decoded_tokens,self.A_j * self.emul_s1[0] + self.eadd_s1[0])
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous() 

        decoded_tokens_noise = self.mask_forward(all_traj_noise,denoise_mask)
        decoded_tokens_noise = decoded_tokens_noise.permute(0, 3, 2, 1).contiguous()  
        decoded_tokens_noise = self.st_gcn1 (decoded_tokens_noise,self.A_j * self.emul_s1[0] + self.eadd_s1[0])
        decoded_tokens_noise = decoded_tokens_noise.permute(0, 3, 2, 1).contiguous()

        past_future_indices = torch.arange(T)[None].repeat(B,1).cuda()
        past_future_indices = past_future_indices[:,None,:].repeat(1,N,1)
        past_ind, future_ind = past_future_indices[:,:, :self.past_timestep], past_future_indices[:,:, self.past_timestep:]
        dec_future_tokens = decoded_tokens[batch_ind, agent_ind, future_ind, :]
        pred_future_coord_values = dec_future_tokens

        if self.denoise_mode == 'all':
            dec_mask_tokens_noise = decoded_tokens_noise
        elif self.denoise_mode == 'past':
            dec_mask_tokens_noise = decoded_tokens_noise[batch_ind,agent_ind, past_ind, :]
        elif self.denoise_mode == 'future':
            dec_mask_tokens_noise = decoded_tokens_noise[batch_ind,agent_ind, future_ind, :]
        pred_mask_coord_values_noise = dec_mask_tokens_noise

        return pred_future_coord_values, pred_mask_coord_values_noise,

    def predict(self, all_traj):
        B, N, T = all_traj.shape[0], all_traj.shape[1], all_traj.shape[2]
        all_traj = all_traj.view(B, N, T, 3)
        act_mask = torch.zeros((B, N, T, 3)).type_as(all_traj)
        act_mask[:, :, :self.past_timestep, :] = 1.
        expanded_act_mask = act_mask
        all_traj = self.activite(all_traj, self.A_j * self.emul_s1[0] + self.eadd_s1[0], expanded_act_mask)

        past_future_indices = torch.arange(T)[None].repeat(B, 1).to(all_traj.device)
        past_future_indices = past_future_indices[:, None, :].repeat(1, N, 1)
        past_ind, future_ind = past_future_indices[:, :, :self.past_timestep], past_future_indices[:, :,
                                                                               self.past_timestep:]

        batch_ind = torch.arange(B)[:, None, None].to(all_traj.device)
        agent_ind = torch.arange(N)[None, :, None].to(all_traj.device)

        ordinary_mask = torch.zeros((B, N, T)).type_as(all_traj)
        ordinary_mask[:, :, :self.past_timestep] = 1.

        decoded_tokens = self.mask_forward(all_traj, ordinary_mask)
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous()
        decoded_tokens = self.st_gcn1(decoded_tokens, self.A_j * self.emul_s1[0] + self.eadd_s1[0])
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous()

        dec_future_tokens = decoded_tokens[batch_ind, agent_ind, future_ind, :]
        pred_future_coord_values = dec_future_tokens

        return pred_future_coord_values

    def mask_forward(self, all_trajs, mask, all_out=False):
        B,N,T = all_trajs.shape[0],all_trajs.shape[1],all_trajs.shape[2]
        all_traj_input = all_trajs
        agent_ind = torch.arange(N)[None,:,None].cuda()
        inverse_mask = 1 - mask
        unmask_tokens = self.patch_embed(all_traj_input) * mask[:,:,:,None]
        unmask_tokens += self.pos_embed.repeat(B,N,1,1)

        if self.add_joint_token:
            unmask_tokens += self.agent_embed.repeat(B,1,T,1)

        unmask_tokens = unmask_tokens * mask[:,:,:,None]
        mask_s = mask.permute(0,2,1).contiguous().view(B*T,N)
        mask_s = torch.matmul(mask_s[:,:,None],mask_s[:,None,:])
        mask_t = mask.contiguous().view(B*N,T)
        mask_t = torch.matmul(mask_t[:,:,None],mask_t[:,None,:])
        unmask_tokens_pad = unmask_tokens
        mask_ind = torch.arange(T)[None,None,:].repeat(B,N,1).cuda()

        for l in range(self.encoder_depth):
            if l == 0:
                encoded_tokens = self.encoder[l](unmask_tokens_pad,mask_s,mask_t)
                enc_to_dec_tokens = encoded_tokens * mask[:,:,:,None]
                mask_tokens = self.mask_embed[None, None, None,:].repeat(B,N,T,1)
                mask_tokens += self.decoder_pos_embed(mask_ind)
                if self.add_joint_token:
                    mask_tokens += self.decoder_agent_embed(agent_ind.repeat(B,1,mask_tokens.shape[2]))
                mask_tokens = mask_tokens * inverse_mask[:,:,:,None]
                concat_tokens = enc_to_dec_tokens + mask_tokens
                dec_input_tokens = concat_tokens
                decoded_tokens = self.decoder[l](dec_input_tokens)

            else:
                encoder_input = decoded_tokens
                encoder_input_pad = encoder_input * mask[:,:,:,None]
                encoder_output = self.encoder[l](encoder_input_pad,mask_s,mask_t)
                decoded_tokens = encoder_output * mask[:,:,:,None] + decoded_tokens * inverse_mask[:,:,:,None]
                decoded_tokens = self.decoder[l](decoded_tokens)

        return decoded_tokens

class STblockb(nn.Module):
    def __init__(self, 
                in_dim,
                h_dim, 
                past_timestep, 
                future_timestep,
                kernel_size,
                graph_args_b,
                decoder_dim=64, 
                num_heads=8, 
                encoder_depth=1,
                dim_per_head=64,
                mlp_dim=64,
                noise_dev=0.6,
                part_noise=True,
                denoise_mode='past',
                part_noise_ratio=0.36,
                add_joint_token=True,
                n_agent=5,
                dropout=0.,
                range_noise_dev=False):
        super(STblockb, self).__init__()
        self.all_timesteps = past_timestep + future_timestep
        self.mask_embed = nn.Parameter(torch.randn(decoder_dim))
        self.past_timestep = past_timestep
        self.future_timestep = future_timestep
        self.encoder_depth = encoder_depth
        self.patch_embed = nn.Linear(in_dim, h_dim)
        self.decoder_pos_embed = nn.Embedding(self.all_timesteps, decoder_dim)
        self.decoder_agent_embed = nn.Embedding(n_agent, decoder_dim)
        self.patch_embed = nn.Linear(32, h_dim)
        self.noise_dev = noise_dev
        self.part_noise = part_noise
        self.part_noise_ratio = part_noise_ratio
        self.denoise_mode = denoise_mode
        self.add_joint_token = add_joint_token
        self.agent_embed = nn.Parameter(torch.randn(1, n_agent, 1, h_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.all_timesteps, h_dim))
        self.range_noise_dev = range_noise_dev
        self.graph_b = Graph_B(**graph_args_b)
        A_b = torch.tensor(self.graph_b.A_b, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_b', A_b)
        self.emul_s1 = nn.ParameterList([nn.Parameter(torch.ones(self.A_b.size())) for i in range(1)])
        self.eadd_s1 = nn.ParameterList([nn.Parameter(torch.zeros(self.A_b.size())) for i in range(1)])
        self.encoder = []
        self.decoder = []
        for _ in range(self.encoder_depth):
            self.encoder.append(STTrans(self.all_timesteps,h_dim,depth=1,mlp_dim=mlp_dim,num_heads=num_heads,dim_per_head=dim_per_head,dropout=dropout))
            self.decoder.append(STTrans(self.all_timesteps,h_dim,depth=1,mlp_dim=mlp_dim,num_heads=num_heads,dim_per_head=dim_per_head,dropout=dropout))
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.activite = Encoderb(in_dim, 32, kernel_size)
        self.st_gcn1 = St_gcn(decoder_dim, 64, kernel_size, 1, 1)
        self.scale3 = AveargePart()
        self.s3_nback = nn.Linear(5,25)
        self.s3_back = nn.Linear(5,25)
    def forward(self,all_traj):
        all_traj =all_traj.permute(0, 3, 2, 1).contiguous()
        all_traj = self.scale3(all_traj)
        all_traj = all_traj.permute(0, 3, 2, 1).contiguous()
        B,N,T = all_traj.shape[0],all_traj.shape[1],all_traj.shape[2]
        all_traj = all_traj.view(B,N,T,3)
        act_mask = torch.zeros((B, N, T, 3)).type_as(all_traj)
        act_mask[:, :, :self.past_timestep,:] = 1.
        expanded_act_mask = act_mask
        
        all_traj= self.activite(all_traj, self.A_b * self.emul_s1[0] + self.eadd_s1[0], expanded_act_mask)
        batch_ind = torch.arange(B)[:,None,None].cuda()
        agent_ind = torch.arange(N)[None,:,None].cuda()
        ordinary_mask = torch.zeros((B,N,T)).type_as(all_traj)
        ordinary_mask[:,:,:self.past_timestep] = 1.
        if self.range_noise_dev:
            noise_dev = np.random.uniform(low=0.1, high=1.0)
        else:
            noise_dev = self.noise_dev
        denoise_mask = torch.zeros((B,N,T)).type_as(all_traj)
        denoise_mask[:,:,:self.past_timestep] = 1.
        noise = torch.from_numpy(np.random.normal(loc=0., scale=noise_dev, size=(B,N,T,32))).type_as(all_traj)
        if self.part_noise:
            noise_mask = torch.rand(B,N,T).cuda()
            noise_mask = (noise_mask < self.part_noise_ratio).type_as(all_traj)
            noise = noise * noise_mask[:,:,:,None].repeat(1,1,1,32)
        all_traj_noise = all_traj + noise

        decoded_tokens = self.mask_forward(all_traj, ordinary_mask)
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous()  
        decoded_tokens = self.st_gcn1(decoded_tokens,self.A_b * self.emul_s1[0] + self.eadd_s1[0])
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous()

        decoded_tokens_noise = self.mask_forward(all_traj_noise, denoise_mask)
        decoded_tokens_noise =  decoded_tokens_noise.permute(0, 3, 2, 1).contiguous()  
        decoded_tokens_noise = self.st_gcn1(decoded_tokens_noise,self.A_b * self.emul_s1[0] + self.eadd_s1[0])
        decoded_tokens_noise =  decoded_tokens_noise.permute(0, 3, 2, 1).contiguous()

        past_future_indices = torch.arange(T)[None].repeat(B,1).cuda()
        past_future_indices = past_future_indices[:,None,:].repeat(1,N,1)
        past_ind, future_ind = past_future_indices[:,:, :self.past_timestep], past_future_indices[:,:, self.past_timestep:]
        dec_future_tokens = decoded_tokens[batch_ind,agent_ind, future_ind, :]

        dec_future_tokens= dec_future_tokens.permute(0, 3, 2, 1).contiguous()
        pred_future_coord_values = self.s3_back(dec_future_tokens)
        pred_future_coord_values=  pred_future_coord_values.permute(0, 3, 2, 1).contiguous()

        if self.denoise_mode == 'all':
            dec_mask_tokens_noise = decoded_tokens_noise
        elif self.denoise_mode == 'past':
            dec_mask_tokens_noise = decoded_tokens_noise[batch_ind,agent_ind, past_ind, :]
        elif self.denoise_mode == 'future':
            dec_mask_tokens_noise = decoded_tokens_noise[batch_ind,agent_ind, future_ind, :]

        dec_mask_tokens_noise = dec_mask_tokens_noise.permute(0, 3, 2, 1).contiguous()
        pred_mask_coord_values_noise = self.s3_nback(dec_mask_tokens_noise)
        pred_mask_coord_values_noise = pred_mask_coord_values_noise.permute(0, 3, 2, 1).contiguous()

        return pred_future_coord_values, pred_mask_coord_values_noise

    def predict(self, all_traj):
        all_traj = all_traj.permute(0, 3, 2, 1).contiguous()
        all_traj = self.scale3(all_traj)
        all_traj = all_traj.permute(0, 3, 2, 1).contiguous()
        B, N, T = all_traj.shape[0], all_traj.shape[1], all_traj.shape[2]
        all_traj = all_traj.view(B, N, T, 3)
        act_mask = torch.zeros((B, N, T, 3)).type_as(all_traj)
        act_mask[:, :, :self.past_timestep, :] = 1.
        expanded_act_mask = act_mask
        all_traj = self.activite(all_traj, self.A_b * self.emul_s1[0] + self.eadd_s1[0], expanded_act_mask)

        past_future_indices = torch.arange(T)[None].repeat(B, 1).to(all_traj.device)
        past_future_indices = past_future_indices[:, None, :].repeat(1, N, 1)
        past_ind, future_ind = past_future_indices[:, :, :self.past_timestep], past_future_indices[:, :,
                                                                               self.past_timestep:]

        batch_ind = torch.arange(B)[:, None, None].to(all_traj.device)
        agent_ind = torch.arange(N)[None, :, None].to(all_traj.device)

        ordinary_mask = torch.zeros((B, N, T)).type_as(all_traj)
        ordinary_mask[:, :, :self.past_timestep] = 1.

        decoded_tokens = self.mask_forward(all_traj, ordinary_mask)
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous()
        decoded_tokens = self.st_gcn1(decoded_tokens, self.A_b * self.emul_s1[0] + self.eadd_s1[0])
        decoded_tokens = decoded_tokens.permute(0, 3, 2, 1).contiguous()

        dec_future_tokens = decoded_tokens[batch_ind, agent_ind, future_ind, :]
        dec_future_tokens = dec_future_tokens.permute(0, 3, 2, 1).contiguous()
        pred_future_coord_values = self.s3_back(dec_future_tokens)
        pred_future_coord_values = pred_future_coord_values.permute(0, 3, 2, 1).contiguous()

        return pred_future_coord_values

    def mask_forward(self,all_trajs,mask,all_out=False):
        B,N,T = all_trajs.shape[0],all_trajs.shape[1],all_trajs.shape[2]
        all_traj_input = all_trajs
        agent_ind = torch.arange(N)[None,:,None].cuda()
        inverse_mask = 1 - mask
        unmask_tokens = self.patch_embed(all_traj_input) * mask[:,:,:,None]
        unmask_tokens += self.pos_embed.repeat(B,N,1,1)
        if self.add_joint_token:
            unmask_tokens += self.agent_embed.repeat(B,1,T,1)
        unmask_tokens = unmask_tokens * mask[:,:,:,None]
        mask_s = mask.permute(0,2,1).contiguous().view(B*T,N)
        mask_s = torch.matmul(mask_s[:,:,None],mask_s[:,None,:])
        mask_t = mask.contiguous().view(B*N,T)
        mask_t = torch.matmul(mask_t[:,:,None],mask_t[:,None,:])
        unmask_tokens_pad = unmask_tokens
        mask_ind = torch.arange(T)[None,None,:].repeat(B,N,1).cuda()

        for l in range(self.encoder_depth):
            if l == 0:
                encoded_tokens = self.encoder[l](unmask_tokens_pad,mask_s,mask_t)

                enc_to_dec_tokens = encoded_tokens * mask[:,:,:,None]
                mask_tokens = self.mask_embed[None, None, None,:].repeat(B,N,T,1)
                mask_tokens += self.decoder_pos_embed(mask_ind)

                if self.add_joint_token:
                    mask_tokens += self.decoder_agent_embed(agent_ind.repeat(B,1,mask_tokens.shape[2]))
                
                mask_tokens = mask_tokens * inverse_mask[:,:,:,None]
                concat_tokens = enc_to_dec_tokens + mask_tokens
                dec_input_tokens = concat_tokens
                decoded_tokens = self.decoder[l](dec_input_tokens)

            else:
                encoder_input = decoded_tokens
                encoder_input_pad = encoder_input * mask[:,:,:,None]
                encoder_output = self.encoder[l](encoder_input_pad,mask_s,mask_t)
                decoded_tokens = encoder_output * mask[:,:,:,None] + decoded_tokens * inverse_mask[:,:,:,None]
                decoded_tokens = self.decoder[l](decoded_tokens)

        return decoded_tokens
class STTrans(nn.Module):
    def __init__(
        self, num_patches, h_dim, depth=3, num_heads=8, mlp_dim=128,
        pool='cls', dim_per_head=64, dropout=0., embed_dropout=0., multi_output=False
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=embed_dropout)
        self.multi_output = multi_output

        self.transformer_s = []
        self.transformer_t = []
        self.depth = depth
        for i in range(depth):
            self.transformer_t.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
            self.transformer_s.append(Transformer(
                h_dim, mlp_dim, depth=1, num_heads=num_heads,
                dim_per_head=dim_per_head, dropout=dropout
            ))
        self.transformer_t = nn.ModuleList(self.transformer_t)
        self.transformer_s = nn.ModuleList(self.transformer_s)
    def forward(self, x, mask_s=None, mask_t=None):
        B,N = x.shape[0], x.shape[1]
        out = []
        for i in range(self.depth):
            x_t = x.contiguous().view(B * N, -1, x.shape[-1])
            x_t = self.transformer_t[i](x_t, mask_t)
            x_t = x_t.view(B, N, -1, x_t.shape[-1])
            x_s = x.permute(0, 2, 1, 3).contiguous().view(-1, N, x.shape[-1])
            x_s = self.transformer_s[i](x_s, mask_s)
            x_s = x_s.view(B, -1, N, x_s.shape[-1]).permute(0, 2, 1, 3)
            x = x_t + x_s
            
        if self.multi_output:
            return out
        else:
            return x
class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net
    
    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, l, d = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        if mask is not None:
            mask = mask[:, None, :, :].repeat(1, self.num_heads, 1, 1)
            attn = attn * mask
            attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-10)

        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).reshape(b, l, -1)
        out = self.out(z)
        return out
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        return self.net(x)
class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=1, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x, mask=None):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x,mask=mask)
            x = x + norm_ffn(x)
        
        return x
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x
class AveargePart(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8, 9, 10, 11, 12]
        self.left_leg = [0, 1, 2, 3]
        self.right_leg = [4, 5, 6, 7]
        self.left_arm = [13, 14, 15, 16, 17, 18]
        self.right_arm = [19, 20, 21, 22, 23, 24]

    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 5))  # [N, C, T, V=1]
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 4))  # [N, C, T, V=1]
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 4))  # [N, C, T, V=1]
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 6))  # [N, C, T, V=1]
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 6))  # [N, C, T, V=1]
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm),
                           dim=-1)  # [N, C, T, V=1]), dim=-1)        # [N, C, T, 5]
        return x_body
class MultiScaleModel(nn.Module):
    def __init__(self, in_dim, h_dim, past_timestep, future_timestep, kernel_size, graph_args_j,graph_args_b):
        super(MultiScaleModel, self).__init__()

        self.past_timestep = past_timestep
        self.st_block_j = STblockj(
            in_dim=in_dim, h_dim=h_dim, past_timestep=past_timestep, future_timestep=future_timestep,
            kernel_size=kernel_size,graph_args_j=graph_args_j)
        self.st_block_b = STblockb(
            in_dim=in_dim, h_dim=h_dim, past_timestep=past_timestep, future_timestep=future_timestep,
            kernel_size=kernel_size,graph_args_b=graph_args_b)
        self.mlp1 = nn.Linear(64,3)
        self.mlp2 = nn.Linear(64,3)
        self.ffn = FFN(64, 128, dropout=0.)
        self.cross_attention_jb = nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.,batch_first=True)

    def forward(self, x):
        x_future = x[:, :,self.past_timestep - 1:self.past_timestep, :]
        pred_j, nois_j = self.st_block_j(x)
        pred_b, nois_b = self.st_block_b(x)
        
        B, V, t, i = pred_j.size()
        pred_j = pred_j.reshape(B * V, t, -1).contiguous()
        pred_b = pred_b.reshape(B * V, t, -1).contiguous()
        att_jb, attn_weights_jb = self.cross_attention_jb(pred_j, pred_b, pred_b)  # [T, B*V, d]
        att_jb = att_jb + pred_j
        att_jb = self.ffn(att_jb) + att_jb  # [T, B*V, d]

        B, V, c, i = nois_j.size()
        nois_j = nois_j.reshape(B * V, c, -1).contiguous()
        nois_b = nois_b.reshape(B * V, c, -1).contiguous()
        attn_jb, attn_weights_jb = self.cross_attention_jb(nois_j, nois_b, nois_b)
        attn_jb = attn_jb + nois_j
        attn_jb = self.ffn(attn_jb) + attn_jb
       
        pred = self.mlp1(att_jb)
        pred = pred.view(B, V, t, -1).contiguous()
        pred = pred + x_future 
        
        pred_nois = self.mlp2(attn_jb)
        pred_nois = pred_nois.view(B, V, c, -1).contiguous()
        pred_nois = pred_nois + x_future

        return pred, pred_nois

    def predict(self, x):
        x_future = x[:, :, self.past_timestep - 1:self.past_timestep, :]
        pred_j = self.st_block_j.predict(x)
        pred_b = self.st_block_b.predict(x)
        B, V, t, i = pred_j.size()
        pred_j = pred_j.reshape(B * V, t, -1).contiguous()
        pred_b = pred_b.reshape(B * V, t, -1).contiguous()
        att_jb, attn_weights_jb = self.cross_attention_jb(pred_j, pred_b, pred_b)
        att_jb = att_jb + pred_j
        att_jb = self.ffn(att_jb) + att_jb
        pred = self.mlp1(att_jb)
        pred = pred.view(B, V, t, -1).contiguous()
        pred = pred + x_future

        return pred