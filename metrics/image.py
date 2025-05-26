import numpy as np
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)



import numpy as np
import torch
from scipy import linalg

acts = [torch.randn(32, 64, 16, 16).view(32, -1) for i in range(100)]
acts = torch.cat(acts, dim=0)
acts = acts.detach().cpu().numpy()[:2000].astype(np.float64)

mu = np.mean(acts, axis=0)
sigma = np.cov(acts, rowvar=False)

mu1 = np.atleast_1d(mu)
mu2 = np.atleast_1d(mu)

sigma1 = np.atleast_2d(sigma)
sigma2 = np.atleast_2d(sigma)

assert mu1.shape == mu2.shape, "Training and test mean vector"

diff = mu1 - mu2

eps=1e-6

covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
if not np.isfinite(covmean).all():
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
        
tr_covmean = np.trace(covmean)