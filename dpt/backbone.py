import torch
import torch.nn as nn
import numpy as np


class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm,
                                            enable_nested_tensor=False)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        # Checkpoint compatibility requires the original post-norm encoder.
        for mod in self.layers:
            src2 = mod.self_attn(output, output, output, attn_mask=mask,
                                 key_padding_mask=src_key_padding_mask)[0]
            output = output + mod.dropout1(src2)
            output = mod.norm1(output)
            src2 = mod.linear2(mod.dropout(mod.activation(mod.linear1(output))))
            output = output + mod.dropout2(src2)
            output = mod.norm2(output)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class AuxilliaryEncoderST(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderST, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm,
                                            enable_nested_tensor=False)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)

        return output
    
class LearnedIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.person_encoding = nn.Embedding(1000, d_model, max_norm=None).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:

        seq_len = 21
   
        x = x + self.person_encoding(torch.arange(num_people).repeat_interleave(seq_len, dim=0).to(self.device)).unsqueeze(1)
        return self.dropout(x)
    

class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model//2, max_norm=None).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=None).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        seq_len = 21
        half = x.size(3)//2

        x_clone = x.clone()
        op1_idx = torch.arange(seq_len).to(self.device)
        op1_add = self.learned_encoding(op1_idx).unsqueeze(1).unsqueeze(0)
        x_clone[:,:,:,0:half*2:2] = x_clone[:,:,:,0:half*2:2] + op1_add
        op2_idx = torch.arange(num_people).unsqueeze(0).repeat_interleave(seq_len, dim=0).to(self.device)
        op2_add = self.person_encoding(op2_idx).unsqueeze(0)
        x_clone[:,:,:,1:half*2:2] = x_clone[:,:,:,1:half*2:2] + op2_add

        return self.dropout(x_clone) 

class Learnedbb3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=None).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
   
        seq_len = 9
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
class Learnedbb2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=None).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        seq_len = 9
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
class Learnedpose3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=None).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
   
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
class Learnedpose2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=None).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)

class TransMotion(nn.Module):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0'):

        super(TransMotion, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.obs_and_pred = obs_and_pred
        self.device = device
        
        self.fc_in_traj = nn.Linear(2,nhid)
        self.fc_out_traj_ego = nn.Linear(nhid, 2)
        self.fc_out_traj_neigh = nn.Linear(nhid, 2)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=21, device=device) 
        self.id_encoder = LearnedIDEncoding(nhid, dropout, device=device)

        # Unused modality modules are retained for checkpoint compatibility.
        self.fc_in_3dbb = nn.Linear(4,nhid)
        self.bb3d_encoder = Learnedbb3dEncoding(nhid, dropout, device=device)

        self.fc_in_2dbb = nn.Linear(4,nhid)
        self.bb2d_encoder = Learnedbb2dEncoding(nhid, dropout, device=device)

        self.fc_in_3dpose = nn.Linear(3, nhid)
        self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)

        self.fc_in_2dpose = nn.Linear(2, nhid)
        self.pose2d_encoder = Learnedpose2dEncoding(nhid, dropout, device=device)


        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)
        

    def forward(self, tgt, padding_mask):
        B, in_F, NJ, K = tgt.shape 

        F = self.obs_and_pred 
        J = self.token_num

        out_F = F - in_F
        N = NJ // J
        
        # Append the last observation over the prediction horizon.
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)  
        tgt = tgt[:,i_idx]        
        tgt = tgt.reshape(B,F,N,J,K)

        # Both draws are part of the original training RNG sequence.
        mask_ratio_traj = 0.0 
        mask_ratio_modality = 0.0 

        tgt_traj = tgt[:,:,:,0,:2].to(self.device)
        rand_tensor_traj = torch.rand((B,F,N)) 
        traj_mask = rand_tensor_traj.float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        tgt_2dbb = tgt[:,:,:,1,:4].to(self.device)

        rand_tensor_2dbb = torch.rand((B,1,N))
        modality_selection_2dbb = (rand_tensor_2dbb.float().to(self.device) > mask_ratio_modality).unsqueeze(3).repeat(1,F,1,4)
        tgt_vis = tgt_2dbb*modality_selection_2dbb
        tgt_2dbb = tgt_vis.to(self.device)

        tgt_traj = self.fc_in_traj(tgt_traj) 
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)

        tgt_2dbb = self.fc_in_2dbb(tgt_2dbb[:,:9]) 
        tgt_2dbb = self.bb2d_encoder(tgt_2dbb)

        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1) 
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1) 
  
        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid) 
        tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid) 
        tgt = torch.cat((tgt_traj,tgt_2dbb),0) 

        out_local = self.local_former(
            tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local
        )
        out_local = out_local * self.output_scale + tgt
        out_local = out_local[:21].reshape(21,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)
        out_global = out_global * self.output_scale + out_local

        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)
        out_ego = self.fc_out_traj_ego(out_primary[0:1])
        out_neigh = self.fc_out_traj_neigh(out_primary[1:])

        out_primary = torch.cat((out_ego, out_neigh), dim=0)
        out = out_primary.permute(2, 1, 0, 3)
        return out

def create_model(config, logger):
    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nlayers_local=config["MODEL"]["num_layers_local"]
    nlayers_global=config["MODEL"]["num_layers_global"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]

    if config["MODEL"]["type"] == "transmotion":
        logger.info("Creating TransMotion model.")
        model = TransMotion(tok_dim=seq_len,
            nhid=nhid,
            nhead=nhead,
            dim_feedfwd=dim_feedforward,
            nlayers_local=nlayers_local,
            nlayers_global=nlayers_global,
            output_scale=config["MODEL"]["output_scale"],
            obs_and_pred=(
                config["TRAIN"]["input_track_size"]
                + config["TRAIN"]["output_track_size"]
            ),
            num_tokens=token_num,
            device=config["DEVICE"]
        ).to(config["DEVICE"]).float()
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model


def clip_embedding_norms(model, max_norm=1.0, parameter_mask=None):
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.Embedding):
                continue

            norms = module.weight.data.norm(p=2, dim=1, keepdim=True)
            coefficient = max_norm / (norms + 1e-6)
            coefficient = torch.min(
                coefficient, torch.ones_like(coefficient)
            )
            if parameter_mask is None:
                module.weight.data.mul_(coefficient)
                continue

            parameter_name = f"{module_name}.weight"
            if parameter_name not in parameter_mask:
                continue
            clipped = module.weight.data * coefficient
            module.weight.data.copy_(
                torch.where(
                    parameter_mask[parameter_name],
                    clipped,
                    module.weight.data,
                )
            )
  
