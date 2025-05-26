"""

"""


noise_scheduler = DDPMScheduler.from_pretrained()


teacher_unet = UNet2DConditionModel(,
                                    subfolder="unet",
                                    revision=args.teacher_revision)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
teacher_unet.requires_grad_(False)

unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_teacher_model,
    subfolder="unet",
    revision=args.teacher_revision
)

unet.train()

# 학습용 모델은 full precision으로 해야하는듯
# Check that all trainable models are in full precision