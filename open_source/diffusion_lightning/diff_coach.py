import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch import LightningMoudle

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from models.diffuser_models.ip-adapters import IPAdapter

class DiffuserCoach(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.cfg = args
        self.build_model(self.cfg)

    def build_model(self, args):
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(

        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
        )

        if args.additional_module is not None:
            module_state_dict = torch.load(args.additional_module, map_location='cpu')
            missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.train()

        if args.use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(

            )

        self.pipeline_cls = StableDiffusionPipeline

    def configure_optimizers(self):
        optim_type = self.config.solver.optimizer
        lr = self.config.solver.lr

        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = util_common.discard_kwargs(optimizers[optim_type])(
            params=self.unet.parameters(),
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
        return [optimizer],[]

    def training_step(self, batch, batch_idx):
        latents = self.encode(batch['pixel_values'].to()).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        if self.cfg.noise_offset:
            noise += self.cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=self.device
            )

        timesteps = torch.randint()
        timesteps = timesteps.long()

        encoder_hidden_states = self.text_encoder(batch["input_ids"], return_dict=False)[0]

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

        if self.cfg.snr_gamma is None:
            F.mse_loss(model_pred.float(), target.)
        else:
            snr = compute_snr(noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

        
    def on_validation_start(self):
        self.pipeline = self.pipeline_cls.from_pretrained(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            revision=self.cfg.revision,
            variant=self.cfg.variant,
        )
        self.pipeline.save_pretrained(self.cfg.output_dir)
        self.pipeline = self.pipeline.to(self.device)

    def validation_step(self, batch, batch_idx):
        images = []

        if self.cfg.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        for i in range(len(batch['prompts']))
            image = self.pipeline(
                batch['prompts'][i],
                generator=generator,
                num_inference_steps=20
            ).images[0]
            images.append(image)

        self.logger.log_image(
            "samples", images=[sketch_grid, reference_grid, target_grid, out_grid], 
            caption=["sketch", "reference", "target", "prediction"]
        )

def main():
    args = parse_args()