import torch


def generate_adjacency_matrix(v, mask=None):
    # return adjacency matrix for Social-STGCNN baseline
    n_ped = v.size(-1)
    temp = v.permute(0, 2, 3, 1).unsqueeze(dim=-2).repeat_interleave(repeats=n_ped, dim=-2)
    a = (temp - temp.transpose(2, 3)).norm(p=2, dim=-1)
    # inverse kernel
    a_inv = 1. / a
    a_inv[a == 0] = 0
    # masking
    a_inv = a_inv if mask is None else a_inv * mask
    # normalize
    a_hat = a_inv + torch.eye(n=n_ped, device=v.device)
    node_degrees = a_hat.sum(dim=-1).unsqueeze(dim=-1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0
    norm_degs_matrix = torch.eye(n=n_ped, device=v.device) * degs_inv_sqrt
    return torch.eye(n=n_ped, device=v.device) - norm_degs_matrix @ a_hat @ norm_degs_matrix


# def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
#     # Pre-process input data for the baseline model
#     if obs_ori is not None:
#         obs_data = torch.cat([obs_data, obs_ori], dim=0)
#     # print(f'obs_data_forward: {obs_data.shape}')
#     features = addl_info
#     con = [torch.cat([features, c], dim=0) for c in torch.transpose(obs_data, 0, 1)]
#     increased_obs_data = torch.stack(con, dim=1)
#     v = increased_obs_data[None, :, :, None].detach()
#     v = v.permute(0, 3, 1, 2)# v = torch.Size([1, 1, 8, 3])
#     a = generate_adjacency_matrix(v).squeeze(dim=0).detach()# a = (8,3,3) 
#     input_data = (v, a)
#     return input_data


def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0) # obs_data_forward =>torch.Size([8, 3])
    # print(f'obs_data_forward: {obs_data.shape}')
    v = obs_data[None, :, :, None].detach()
    v = v.permute(0, 3, 1, 2) # v => torch.Size([1, 1, 8, 3])
    a = generate_adjacency_matrix(v).squeeze(dim=0).detach() # a => torch.Size([8,3,3])
    input_data = (v, a)
    return input_data


def model_forward(input_data, baseline_model , addl_info=None):
    # Forward the baseline model with input data
    feature = addl_info
    output_data = baseline_model(*input_data , feature)
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data.permute(0, 2, 3, 1).squeeze(dim=0)
    return pred_data

