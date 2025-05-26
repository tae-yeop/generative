





tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                          subfolder="tokenizer",
                                          revision=args.revision,
                                          use_fast=False,)




if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet)


vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
    