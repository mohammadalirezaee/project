import torch
import torch.nn as nn
from .anchor import ETAnchor
from .descriptor import ETDescriptor
from .feature_extractor import modified_UNet


class EigenTrajectory(nn.Module):
    r"""The EigenTrajectory model

    Args:
        baseline_model (nn.Module): The baseline model
        hook_func (dict): The bridge functions for the baseline model
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, baseline_model, hook_func, hyper_params):
        super().__init__()

        self.baseline_model = baseline_model
        self.hook_func = hook_func
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.static_dist = hyper_params.static_dist

        self.Scene_features = modified_UNet() #new add
        self.ET_m_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=True)
        self.ET_s_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=False)
        self.ET_m_anchor = ETAnchor(hyper_params=hyper_params)
        self.ET_s_anchor = ETAnchor(hyper_params=hyper_params)

    def calculate_parameters(self, obs_traj, pred_traj):
        r"""Calculate the ET descriptors of the EigenTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Note:
            This function should be called once before training the model.
        """

        # Mask out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj, pred_m_traj = obs_traj[mask], pred_traj[mask]
        obs_s_traj, pred_s_traj = obs_traj[~mask], pred_traj[~mask]

        # Descriptor initialization
        data_m = self.ET_m_descriptor.parameter_initialization(obs_m_traj, pred_m_traj) # return  pred_traj_norm, U_pred_trunc
        data_s = self.ET_s_descriptor.parameter_initialization(obs_s_traj, pred_s_traj)
        # Anchor generation
        self.ET_m_anchor.anchor_generation(*data_m)
        self.ET_s_anchor.anchor_generation(*data_s)

    def forward(self, obs_traj, pred_traj=None, addl_info=None):
        r"""The forward function of the EigenTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)
            addl_info (dict): The additional information (optional, if baseline model requires)
            addl_info : frame list that contain 8 frame of pedestrain
        Returns:
            output (dict): The output of the model (recon_traj, loss, etc.)
        """
        
        n_ped = obs_traj.size(0)

        # Filter out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj = obs_traj[mask]
        obs_s_traj = obs_traj[~mask]
        pred_m_traj_gt = pred_traj[mask] if pred_traj is not None else None
        pred_s_traj_gt = pred_traj[~mask] if pred_traj is not None else None

        # Projection
        C_m_obs, C_m_pred_gt = self.ET_m_descriptor.projection(obs_m_traj, pred_m_traj_gt)
        C_s_obs, C_s_pred_gt = self.ET_s_descriptor.projection(obs_s_traj, pred_s_traj_gt)
        C_obs = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
        C_obs[:, mask], C_obs[:, ~mask] = C_m_obs, C_s_obs  # KN

        # Absolute coordinate
        obs_m_ori = self.ET_m_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_s_ori = self.ET_s_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_ori = torch.zeros((2, n_ped), dtype=torch.float, device=obs_traj.device)
        obs_ori[:, mask], obs_ori[:, ~mask] = obs_m_ori, obs_s_ori
        # print(f'obs_ori_before(input to GCN): {obs_ori.shape}')
        obs_ori -= obs_ori.mean(dim=1, keepdim=True)  # move scene to origin
        lis = []
        print(f'C_observation(input to GCN): {C_obs.shape}')
        # print(f'C_observation(input to GCN): {features.shape}')
        print('******************')
        # print(type(addl_info))
        # print(len(addl_info))
        # print(addl_info[0].shape)
        # print(addl_info[7].shape)
        features = self.Scene_features.forward(addl_info[0]).squeeze(0) #new add
        # print(f'features: {features.shape}')
        # con = [torch.cat([features, c], dim=0) for c in torch.transpose(C_obs, 0, 1)]
        # increased_C = torch.stack(con, dim=1)
        # print(f'increased_C: {increased_C.shape}')
        
        # print(f'obs_ori(input to GCN): {obs_ori}')
        # Trajectory prediction

        input_data = self.hook_func.model_forward_pre_hook(C_obs, obs_ori)

        # v = input_data[0]
        # a = input_data[1]
        # print(f'a: {a.shape}') # torch.Size([8, 3, 3])
        # print(f'v: {v.shape}') #  torch.Size([1, 1, 8, 3])

        output_data = self.hook_func.model_forward(input_data, self.baseline_model , addl_info=features)
        C_pred_refine = self.hook_func.model_forward_post_hook(output_data)
        lis.append(C_pred_refine)
        # print(len(lis))
        print(f'C_pred(output of GCN): {C_pred_refine.shape}')
        # Anchor refinement
        C_m_pred = self.ET_m_anchor(C_pred_refine[:, mask])
        C_s_pred = self.ET_s_anchor(C_pred_refine[:, ~mask])

        # Reconstruction
        pred_m_traj_recon = self.ET_m_descriptor.reconstruction(C_m_pred)
        pred_s_traj_recon = self.ET_s_descriptor.reconstruction(C_s_pred)
        pred_traj_recon = torch.zeros((self.s, n_ped, self.t_pred, self.dim), dtype=torch.float, device=obs_traj.device)
        pred_traj_recon[:, mask], pred_traj_recon[:, ~mask] = pred_m_traj_recon, pred_s_traj_recon
        # print(pred_traj_recon.shape)
        output = {"recon_traj": pred_traj_recon}

        if pred_traj is not None:
            C_pred = torch.zeros((self.k, n_ped, self.s), dtype=torch.float, device=obs_traj.device)
            C_pred[:, mask], C_pred[:, ~mask] = C_m_pred, C_s_pred
            print(f'C_prediction after refinment(add with trainable anchor points): {C_pred.shape}')

            # Low-rank approximation for gt trajectory
            C_pred_gt = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
            C_pred_gt[:, mask], C_pred_gt[:, ~mask] = C_m_pred_gt, C_s_pred_gt
            C_pred_gt = C_pred_gt.detach()

            # Loss calculation
            error_coefficient = (C_pred - C_pred_gt.unsqueeze(dim=-1)).norm(p=2, dim=0)
            error_displacement = (pred_traj_recon - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
            output["loss_eigentraj"] = error_coefficient.min(dim=-1)[0].mean()
            output["loss_euclidean_ade"] = error_displacement.mean(dim=-1).min(dim=0)[0].mean()
            output["loss_euclidean_fde"] = error_displacement[:, :, -1].min(dim=0)[0].mean()

        return output 


