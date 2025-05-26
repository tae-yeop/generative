

@classmethod
def from_unet(
    cls,
    unet,
    controlnet_conditioning_channel_order: str = "rgb",
    conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
    load_weights_from_unet: bool = True,
    conditioning_channels: int = 3,
    
):
    """
    기본적으로 Unet의 weight로 초기화한다. Mid block + Down blocks로 초기화
    """
    controlnet = cls(
        ...
    )

    if load_weights_from_unet:
        controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
        controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
        controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

        # class_embedding은 뭐지?
        if controlnet.class_embedding:
            controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

        controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
        controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())