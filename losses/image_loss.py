import torch
import torch.nn as nn
import torch.nn.functional as F

from .insight_face.model_irse import Backbone

def tv_loss(img):
    """
    노이즈를 줄이고 부드럽게 만드는 정규화

    Total Variation Loss:
    TV(x) = sum_{i,j} ( (x[i,j+1] - x[i,j])^2 + (x[i+1,j] - x[i,j])^2 )

    Args:
        img (torch.Tensor): 이미지 텐서. shape = [batch_size, channels, height, width].
    https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/image/tv.py
    """
    # W (width) 방향 차이: 오른쪽 픽셀과 현재 픽셀의 차이
    # img[:,:,:,:-1] : 마지막 픽셀 제외 (width)
    # img[:,:,:,1:] : 첫 번째 픽셀 제외 (width)
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))

    # H (height) 방향 차이: 아래쪽 픽셀과 현재 픽셀의 차이
    # img[:,:,:-1,:] : 마지막 픽셀 제외 (height)
    # img[:,:,1:,:] : 첫 번째 픽셀 제외 (height)
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = (h_variance + w_variance)
    return loss


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, tgt, m):
        return self.mse(m*pred, m*tgt)
    
class IDLoss(nn.Module):
    # sh ./download_from_google_drive.sh 1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn model_ir_se50.pth
    def __init__(self, backbone_path):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        try:
            self.facenet.load_state_dict(torch.load(backbone_path, map_location=torch.device('cpu'))['state_dict'])
        except IOError:
            self.facenet.load_state_dict(torch.load('/apdcephfs/share_916081/amosyhliu/pretrained_models/model_ir_se50.pth'))

        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, x_hat):
        self.facenet.eval()
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        x_feats = x_feats.detach()

        x_hat_feats = self.extract_feats(x_hat)
        losses = []
        for i in range(n_samples):
            loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
            losses.append(loss_sample.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        return losses / n_samples

class SSIMLoss():
    def __init__(self, pkg_name='pytorch_ssim', is_multiscale=False):
        

        if pkg_name == 'pytorch_ssim':
            try:
                import pytorch_ssim
            except ImportError:
                raise ImportError("Please install pip install pytorch-ssim")
            self.loss_fn = pytorch_ssim.SSIM(window_size=11)
        elif pkg_name == 'torchmetrics':
            try:
                from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
            except ImportError:
                raise ImportError("Please install pip install torchmetrics")
            self.loss_fn = StructuralSimilarityIndexMeasure(data_range=1.0)

        else:
            raise ValueError(f"Unsupported package name: {pkg_name}")
        
        self.is_multiscale = is_multiscale

    def forward(self, pred, target):
        """
        pred: 예측 이미지
        target: 실제 이미지
        """
        return 1 - self.loss_fn(pred, target)


class MS_SSIM_Loss:
    """
    다양한 해상도(스케일)에서 구조적 유사성을 평가해 더 정교한 품질 평가가 가능한 MS-SSIM Loss
    """
    def __init__(self):
        super.().__init__()
        try:
            from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
        except ImportError:
            raise ImportError("Please install torchmetrics: pip install torchmetrics")

        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, pred, target):
        """
        pred: 예측 이미지 (Tensor, shape [B, C, H, W])
        target: 실제 이미지 (Tensor, shape [B, C, H, W])
        """
        return 1 - self.ms_ssim(pred, target)


class I2ILoss(nn.Module):
    def __init__(self, domain_rgb, l1_lambda , l2_lambda, 
                 lpips_lambda, lpips_type, lpips_model_path,
                 tv_lambda, ssim_lambda, cnt_lambda, 
                 id_lambda, id_backbone_path, ffl_w, ffl_alpha, gan_loss_type, r1_gamma,
                 lpips_start_step=0, tv_start_step=0, ssim_start_step=0, cnt_start_step=0,
                 id_start_step=0, ffl_start_step=0
                 ):
        super().__init__()

        params = locals()
        # print(params)
        # self.domain_rgb = domain_rgb
        # self.l1_lambda = l1_lambda
        # self.l2_lambda = l2_lambda
        # self.lpips_lambda = lpips_lambda
        # self.lpips_type = lpips_type
        # self.tv_lambda = tv_lambda
        # self.ssim_lambda = ssim_lambda
        # self.cnt_lambda = cnt_lambda
        # self.id_lambda = id_lambda
        # self.ffl_w = ffl_w
        # self.ffl_alpha = ffl_alpha
        # self.gan_loss_type = gan_loss_type
        # self.r1_gamma = r1_gamma

        params.pop('self')
        for key, value in params.items():
            setattr(self, key, value)

        self.loss_apply = dict()
        self.use_loss = dict()
        
        self.recon_loss = nn.L1Loss() if self.l1_lambda else nn.MSELoss() if self.l2_lambda else None


        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        self.idwt = DWTInverse(mode="zero", wave="db1")

        self.lpips_loss = LPIPS(net=lpips_type, pnet_rand=True, pretrained=True, model_path=lpips_model_path) if self.lpips_lambda else None
        self.loss_apply.update({'lpips': lpips_start_step})
        self.use_loss.update({'lpips': False})
        
        self.tv_loss = tv_loss if tv_lambda else None
        self.loss_apply.update({'tv': tv_start_step})
        self.use_loss.update({'tv': False})
        
        self.ssim_loss = SSIM_Loss() if self.ssim_lambda else None
        self.loss_apply.update({'ssim': ssim_start_step})
        self.use_loss.update({'ssim': False})
        
        self.content_loss = ContentLoss() if self.cnt_lambda else None
        self.loss_apply.update({'cnt': cnt_start_step})
        self.use_loss.update({'cnt': False})

        self.id_loss = IDLoss(id_backbone_path) if self.id_lambda else None
        self.loss_apply.update({'id': id_start_step})
        self.use_loss.update({'id': False})

        self.ff_loss = FFL(loss_weight=self.ffl_w, alpha=self.ffl_alpha,
                           patch_factor=1,ave_spectrum=True, log_matrix=True, batch_matrix=True) if self.ffl_w else None
        self.loss_apply.update({'ffl': ffl_start_step})
        self.use_loss.update({'ffl': False})

        
        self.g_loss = gan_losses[gan_loss_type][0] if gan_loss_type else None
        self.d_loss = gan_losses[gan_loss_type][1] if gan_loss_type else None
        self.r1_reg = d_r1_loss if self.r1_gamma else None

        self.toggle_loss(0)

    def toggle_loss(self, step):
        for key, value in self.loss_apply.items():
            if step >= step:
                self.use_loss[key] = True
                
    def loss_g(self, pred, gt, fake_logit, m=None):
        loss = 0.0
        loss_dict = {}
        # Recon loss
        pred_new_domain = self.img_to_dwt(pred) if self.domain_rgb else self.dwt_to_img(pred)
        gt_new_domain = self.img_to_dwt(gt) if self.domain_rgb else self.dwt_to_img(gt)
        loss = self.recon_loss(pred, gt)
        loss += self.recon_loss(pred_new_domain, gt_new_domain)

        if exists(self.lpips_loss) and self.use_loss['lpips'] : loss += self.lpips_lambda * self.lpips_loss(pred, gt).mean()
        if exists(self.tv_loss) and self.use_loss['tv'] : loss += self.tv_lambda * self.tv_loss(pred)
        if exists(self.ssim_loss) and self.use_loss['ssim'] : loss += self.ssim_lambda * self.ssim_loss(pred, gt)
        if exists(self.content_loss) and self.use_loss['cnt'] : loss += self.cnt_lambda * self.content_loss(pred, gt, m)
        if exists(self.id_loss) and self.use_loss['id'] : loss += self.id_lambda * self.id_loss(pred, gt).mean()
        if exists(self.ff_loss) and self.use_loss['ffl'] : loss += self.ff_loss(pred, gt)
        if exists(self.g_loss) : loss += self.g_loss(fake_logit)

        loss_dict.update({'g_loss' : loss})
        return loss_dict, loss
        
    def loss_d(self, fake_logit, real_logit):
        loss = self.d_loss(real_logit, fake_logit)
        loss_dict = {}
        loss_dict.update({'d_loss' : loss})
        return loss_dict, loss

    def regularize_d(self, real_logit, real_img):
        reg_d_dict = {}
        if exists(self.r1_reg):
            real_img.requires_grad = True
            r1 = 0.5 * self.r1_gamma * self.r1_reg(real_logit, real_img, self.r1_gamma)
            reg_d_dict.update({'r1_reg' : r1})
            return r1

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))