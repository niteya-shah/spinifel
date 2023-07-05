import glob, os, sys
import numpy as np
import PyNVTX as nvtx
from spinifel import settings, image
from spinifel.prep import save_mrc
import torch
from scipy.interpolate import RegularGridInterpolator

sys.path.append('/global/homes/z/zhantao/Projects/dgp3d_spi/spi-reconstruction/')
from dgp3d.model.autoencoders.contrastive import ContrastiveVolumeEstimatorLightningModule
from dgp3d.utils_general import array2tensor, tensor2array
from dgp3d.utils_align import align_volumes

@nvtx.annotate("mpi/main.py", is_prefix=True)
def network_estimation(slices_, logger=None, device='cpu'):
    print("using network estimation")
    ckpt_file = settings.network_path
    torch.set_default_dtype(torch.float64)
    if logger is not None:
        logger.log("loading model from ", ckpt_file)
    else:
        print("loading model from ", ckpt_file)
    model = ContrastiveVolumeEstimatorLightningModule.load_from_checkpoint(ckpt_file).float()

    slices_ = array2tensor(slices_)
    print(settings.M)
    output_resizer = torch.nn.Upsample(
        size=model.real_mesh.shape[:3], mode='trilinear', align_corners=True)
    if not slices_.shape[0] == model.encoder_input_size:
        input_resizer = torch.nn.Upsample(
                size=(model.encoder_input_size,)*2,
                mode='bilinear', align_corners=True)
        slices_ = input_resizer(slices_)
    rho_pred = evaluate_slices_realvolume(model, slices_, 128, device=device)
    rho_pred = output_resizer(rho_pred).squeeze()
    rho_pred = tensor2array(rho_pred)
    image.show_volume(rho_pred, settings.Mquat, "rho_network_raw.png")
    save_mrc(settings.out_dir / f"rho-network-raw.mrc", rho_pred.squeeze())
    rho_func = get_rho_function(tensor2array(model.real_mesh), rho_pred)
    # image.show_volume(rho_pred, settings.Mquat, "rho_network_0.png")
    return rho_func

def evaluate_slices_realvolume(model, slices, batch_size, align_vols=False, device='cpu'):
    model.to(device)
    rho_pred_list = []
    batches = torch.split(slices.clone().to(device), batch_size, dim=0)
    for batch in batches:
        batch = model.encoder_normalize_input_max * \
            batch / torch.amax(batch, dim=(-2,-1), keepdim=True)
        batch[batch < 0] = 0
        batch = torch.log(batch + 1.)
        with torch.no_grad():
            rho_tmp = model(batch.float())[0].detach().cpu()
        rho_pred_list.append(rho_tmp)

    rho_pred = torch.vstack(rho_pred_list)
    print(rho_pred.shape)
    if align_vols:
        rho_pred = align_volumes_by_anchor(rho_pred.numpy().squeeze(), rho_pred.numpy()[0,0])
        rho_pred = torch.vstack(rho_pred_list).mean(dim=0, keepdim=True)
    else:
        rho_pred = rho_pred.mean(dim=0).unsqueeze(0)
        print(rho_pred.shape)
    # rho_pred = rho_pred[0,None]
    return rho_pred

def get_rho_function(real_mesh, rho):
    print(real_mesh.shape)
    print(rho.shape)
    real_mesh = np.around(real_mesh, 3)
    x = np.sort(np.unique(real_mesh[...,0]))
    y = np.sort(np.unique(real_mesh[...,1]))
    z = np.sort(np.unique(real_mesh[...,2]))
    assert len(x) == rho.shape[0]
    assert len(y) == rho.shape[1]
    assert len(z) == rho.shape[2]
    rho_func = RegularGridInterpolator((x, y, z), rho, bounds_error=False, fill_value=0.)
    return rho_func

def align_volumes_by_anchor(volumes, anchor_volume):
    print("performing alignment")
    print(volumes.shape, anchor_volume.shape)
    volumes_aligned = []
    from tqdm import tqdm
    # for volume in tqdm(volumes, miniters=100):
    for volume in volumes:
        volume_aligned, _, _ = align_volumes(volume, anchor_volume, zoom=1/4)
        volumes_aligned.append(volume_aligned[None])
    volumes_aligned = np.vstack(volumes_aligned)[:,None]
    return volumes_aligned


def get_real_mesh(voxel_number_1d, distance_reciprocal_max, xp):
    """
    
    Parameters
    ----------
    voxel_number_1d : int
        number of voxels per axis
    distance_reciprocal_max : float
        maximum voxel resolution in inverse Angstrom
    xp: package
        use numpy or cupy

    Returns
    -------
    real_mesh : numpy.ndarray, shape (n,n,n,3)
        grid of real space vectors for each voxel
    """
    
    max_reciprocal_value = distance_reciprocal_max
    _, step = xp.linspace(-max_reciprocal_value, max_reciprocal_value, voxel_number_1d, retstep=True)
    max_real_value = 1 / (2*step)
    linspace = np.linspace(-max_real_value, max_real_value, voxel_number_1d) * 1e10
    real_mesh_stack = np.asarray(
            np.meshgrid(linspace, linspace, linspace, indexing='ij'))
    real_mesh = np.moveaxis(real_mesh_stack, 0, -1)
    
    return real_mesh
    
