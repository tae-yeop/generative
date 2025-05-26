import os
import torch
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import load_file

# ---------------------------
# [utils.py 부분] 통합
# ---------------------------
def get_lora_weight_hkl(weight_path, dtype: torch.dtype):
    lora_weight = {}
    with safe_open(weight_path, framework='pt', device=0) as f:
        for k in f.keys():
            lora_weight[k] = f.get_tensor(k)
    lora_state_dict = convert_module_name_official(lora_weight, dtype)
    return lora_state_dict

def convert_module_name_official(module, dtype: torch.dtype):
    kohya_ss_state_dict = {}
    for peft_key, weight in module.items():
        # base_model.model --> lora_unet
        kohya_key = peft_key.replace("base_model.model", "lora_unet")
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        # 특정 "."만 "_"로 교체하는 처리
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)

        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # alpha 설정
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(64).to(dtype)

    return kohya_ss_state_dict

def optimize_pipeline(pipeline, args):
    from importlib.metadata import version 
    from packaging.version import parse

    MIN_DIFFUSERS_VERSION_FUSE_QKV = "0.25"
    _diffusers_version = parse(version("diffusers")).base_version

    # Ampere+ GPU(Compute Capability 8.0 이상)에서 TF32 허용
    if torch.cuda.get_device_capability(0) >= (8, 0):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    torch.set_grad_enabled(False)

    pipeline.unet.to(memory_format=torch.channels_last)
    pipeline.vae.to(memory_format=torch.channels_last)

    # diffusers>=0.25 이면 fuse_qkv_projections 사용 가능
    if parse(_diffusers_version) >= parse(MIN_DIFFUSERS_VERSION_FUSE_QKV):
        pipeline.unet.fuse_qkv_projections()
        pipeline.vae.fuse_qkv_projections()

    # compile 옵션
    if args.compile is not None:
        pipeline = complie_pipeline(pipeline, args.compile)

    # deepcache
    if args.deepcache and args.compile != 'torch':
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(
            cache_interval=3,
            cache_branch_id=0,
        )
        helper.enable()

    # quantization
    if args.quantization == 'torch_int8' and args.compile is None:
        raise AssertionError("Quantization is set to 'torch_int8' but compile is None.")

    if not args.deepcache and args.compile != 'torch':
        pipeline.vae = quantize(pipeline.vae, args)
        pipeline.controlnet = quantize(pipeline.controlnet, args)
    
    return pipeline

def complie_pipeline(pipeline, compile_type):
    if compile_type == 'sfast':
        from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
        config = CompilationConfig.Default()
        try:
            import xformers
            config.enable_xformers = False
        except ImportError:
            print('xformers not installed, skip')
        try:
            import triton
            config.enable_triton = True
        except ImportError:
            print('Triton not installed, skip')

        config.enable_cuda_graph = True
        pipeline = compile(pipeline, config)

    elif compile_type == 'torch':
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=False)
        pipeline.vae = torch.compile(pipeline.vae, mode="max-autotune", fullgraph=False)
        pipeline.controlnet = torch.compile(pipeline.controlnet, mode="max-autotune", fullgraph=False)
        
    elif compile_type == 'tensorrt':
        import gc
        import os
        from pathlib import Path
        from polygraphy import cuda
        from streamdiffusion.acceleration.tensorrt import (
            TorchVAEEncoder,
            compile_unet,
            compile_vae_decoder,
            compile_vae_encoder,
        )
        from streamdiffusion.acceleration.tensorrt.engine import (
            AutoencoderKLEngine,
            UNet2DConditionModelEngine,
        )
        from streamdiffusion.acceleration.tensorrt.models import (
            VAE,
            UNet,
            VAEEncoder,
        )

        engine_dir = Path("./engine")
        unet_path = os.path.join(engine_dir, "unet.engine")
        vae_encoder_path = os.path.join(engine_dir, "vae_encoder.engine")
        vae_decoder_path = os.path.join(engine_dir, "vae_decoder.engine")
        device = "cuda"
        batch_size = 1

        # compile unet
        if not os.path.exists(unet_path):
            os.makedirs(os.path.dirname(unet_path), exist_ok=True)
            unet_model = UNet(
                fp16=True,
                device=device,
                max_batch_size=batch_size,
                min_batch_size=batch_size,
                embedding_dim=pipeline.text_encoder.config.hidden_size,
                unet_dim=pipeline.unet.config.in_channels,
            )
            compile_unet(
                pipeline.unet,
                unet_model,
                unet_path + ".onnx",
                unet_path + ".opt.onnx",
                unet_path,
                opt_batch_size=batch_size,
            )
        # compile vae-decoder
        if not os.path.exists(vae_decoder_path):
            os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
            pipeline.vae.forward = pipeline.vae.decode
            vae_decoder_model = VAE(
                device=device,
                max_batch_size=batch_size,
                min_batch_size=batch_size
            )
            compile_vae_decoder(
                pipeline.vae,
                vae_decoder_model,
                vae_decoder_path + ".onnx",
                vae_decoder_path + ".opt.onnx",
                vae_decoder_path,
                opt_batch_size=batch_size
            )
            delattr(pipeline.vae, "forward")

        # compile vae-encoder
        if not os.path.exists(vae_encoder_path):
            os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
            vae_encoder = TorchVAEEncoder(pipeline.vae).to(torch.device("cuda"))
            vae_encoder_model = VAEEncoder(
                device=device,
                max_batch_size=batch_size,
                min_batch_size=batch_size
            )
            compile_vae_encoder(
                vae_encoder,
                vae_encoder_model,
                vae_encoder_path + ".onnx",
                vae_encoder_path + ".opt.onnx",
                vae_encoder_path,
                opt_batch_size=batch_size
            )

        cuda_stream = cuda.Stream()
        vae_config = pipeline.vae.config
        vae_dtype = pipeline.vae.dtype

        pipeline.unet = UNet2DConditionModelEngine(
            unet_path, cuda_stream, use_cuda_graph=False
        )

        pipeline.vae = AutoencoderKLEngine(
            vae_encoder_path,
            vae_decoder_path,
            cuda_stream,
            pipeline.vae_scale_factor,
            use_cuda_graph=False,
        )
        setattr(pipeline.vae, "config", vae_config)
        setattr(pipeline.vae, "dtype", vae_dtype)

        gc.collect()
        torch.cuda.empty_cache()

    return pipeline

def quantize(m, args):
    if args.quantization == 'torch_int8':
        from diffusers.utils import USE_PEFT_BACKEND
        assert USE_PEFT_BACKEND  # diffusers>=0.18부터 PEFT 백엔드에서 int8 지원
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                dtype=torch.qint8,
                                                inplace=True)
        return m
    else:
        return m

def benchmark(func, bench_dict):
    # warmup
    warmup = 5
    for _ in range(warmup):
        func(**bench_dict)

    results = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    iterations = 10
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_event.record()
        func(**bench_dict)
        end_event.record()
        torch.cuda.synchronize()
        results.append(start_event.elapsed_time(end_event))

    avg_ms = sum(results) / len(results)
    print(f"Average time: {avg_ms} ms")
    print(f"Average FPS: {1000 / avg_ms}")

# ---------------------------
# [infer.py 부분] 통합
# ---------------------------
import torch
from compel import Compel, DiffusersTextualInversionManager
from diffusers import ControlNetModel, DPMSolverSinglestepScheduler
from diffusers import LCMScheduler

# 만약 custom_pipeline.py 내부 구현이 있다면
# from custom_pipeline import CustomStableDiffusionControlNetPipeline
# 여기서는 코드가 없으니 그대로 import만 남김

SAMPLE_PREFIX = './samples'

class Temp:
    def __init__(self) -> None:
        pass

def prepare_pipeline(args):
    # 여기서 CustomStableDiffusionControlNetPipeline.from_single_file() 사용
    from custom_pipeline import CustomStableDiffusionControlNetPipeline

    pipeline: CustomStableDiffusionControlNetPipeline = CustomStableDiffusionControlNetPipeline.from_single_file(
        './models/epicphotogasm_lastUnicorn.safetensors',
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        load_safety_checker=None,
        controlnet=ControlNetModel.from_pretrained('./models/controlnet'),
    )

    pipeline.load_ip_adapter(".", subfolder="models", weight_name="ip_adapter_total.bin")
    pipeline.set_ip_adapter_scale(args.ip_adapter_scale)

    pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
        pipeline.scheduler.config, use_karras_sigmas=True
    )

    if args.use_lcm:
        # lora_state_dict 로드
        lora_state_dict = get_lora_weight_hkl('sub_models/lcm-lora/epicphotogasm_100k.safetensors', torch.float16)
        pipeline.load_lora_weights(lora_state_dict, None)
        pipeline.fuse_lora(
            fuse_unet=True,
            fuse_text_encoder=True,
            lora_scale=args.lcm_lora_scale,
            safe_fusing=False
        )
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        args.num_inference_steps = 4
        args.guidance_scale = 1.0

    # face encoders
    pipeline.set_ir200(weight_path='./models/face_encoder/ir200.pth')
    pipeline.set_scrfd(weight_path='./models/face_detector/scrfd_2.5g_bnkps.onnx')

    pipeline.to('cuda')

    if args.optimize:
        pipeline = optimize_pipeline(pipeline, args)

    return pipeline

def general_infer(pipeline, args):
    # controlNet conditioning image, ip-adapter embedding
    ip_adap_emb, controlnet_img = pipeline.get_custom_conditions(
        style_img=args.style_image,
        face_infos=args.face_infos,
        width=args.width,
        height=args.height,
    )
    ip_adap_emb = ip_adap_emb.to(dtype=torch.float16)

    textual_inversion_manager = DiffusersTextualInversionManager(pipeline)
    compel_proc = Compel(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        textual_inversion_manager=textual_inversion_manager
    )

    out = pipeline(
        prompt_embeds=compel_proc(args.prompt),
        image=controlnet_img,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt_embeds=compel_proc(args.neg_prompt),
        num_images_per_prompt=1,
        ip_adapter_embed=ip_adap_emb,
        generator=torch.Generator('cuda').manual_seed(args.seed)
    ).images

    out[0].save('./out_samples/test2.jpg')

def make_before_and_after(
    pipeline,
    args,
    before_input_image_path,
    after_input_image_path,
    after_scale=0.7
):
    # before
    before_ip_adap_emb, controlnet_img = pipeline.get_custom_conditions(
        style_img=args.style_image,
        face_infos=[{'bgr_image': cv2.imread(before_input_image_path), 'scale': 1.0}],
        width=args.width,
        height=args.height,
    )
    before_ip_adap_emb = before_ip_adap_emb.to(dtype=torch.float16)

    # after
    after_ip_adap_emb, _ = pipeline.get_custom_conditions(
        style_img=args.style_image,
        face_infos=[
            {'bgr_image': cv2.imread(after_input_image_path), 'scale': after_scale},
            {'bgr_image': cv2.imread(before_input_image_path), 'scale': 1.0 - after_scale},
        ],
        width=args.width,
        height=args.height,
    )
    after_ip_adap_emb = after_ip_adap_emb.to(dtype=torch.float16)

    textual_inversion_manager = DiffusersTextualInversionManager(pipeline)
    compel_proc = Compel(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        textual_inversion_manager=textual_inversion_manager
    )

    # before
    before_outs = pipeline(
        prompt_embeds=compel_proc(args.prompt),
        image=controlnet_img,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt_embeds=compel_proc(args.neg_prompt),
        num_images_per_prompt=1,
        ip_adapter_embed=before_ip_adap_emb,
        generator=torch.Generator('cuda').manual_seed(args.seed)
    ).images
    before_outs[0].save('./out_samples/before3.jpg')

    # after
    after_outs = pipeline(
        prompt_embeds=compel_proc(args.prompt),
        image=controlnet_img,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt_embeds=compel_proc(args.neg_prompt),
        num_images_per_prompt=1,
        ip_adapter_embed=after_ip_adap_emb,
        generator=torch.Generator('cuda').manual_seed(args.seed)
    ).images
    after_outs[0].save('./out_samples/after3.jpg')

# ---------------------------
# main 구간
# ---------------------------
if __name__ == '__main__':
    # 간단히 Temp 클래스로 args 비슷하게 세팅
    args = Temp()
    args.ip_adapter_scale = 1.0
    args.controlnet_conditioning_scale = 0.8
    args.width = 512
    args.height = 768
    args.num_inference_steps = 20
    args.guidance_scale = 1.5
    args.seed = 127966481

    args.prompt = ''
    args.neg_prompt = ''

    # LCM
    args.use_lcm = True
    args.lcm_lora_scale = 1.0

    # style & face infos
    args.style_image = Image.open(f'{SAMPLE_PREFIX}/style00.png')
    args.face_infos = [
        {'bgr_image': cv2.imread(f'{SAMPLE_PREFIX}/source00.jpg'), 'scale': 0.5},
        {'bgr_image': cv2.imread(f'{SAMPLE_PREFIX}/source01.jpg'), 'scale': 0.5},
    ]

    # optimization options
    args.optimize = False
    args.compile = 'sfast'  # torch, sfast, None
    args.deepcache = True
    args.quantization = 'torch_int8'  # torch_int8 or None
    pipeline = prepare_pipeline(args)

    # 예: before/after 이미지 생성
    # make_before_and_after(
    #     pipeline,
    #     args,
    #     before_input_image_path=f'{SAMPLE_PREFIX}/target00.jpg',
    #     after_input_image_path=f'{SAMPLE_PREFIX}/fused.jpg',
    #     after_scale=0.7
    # )

    # 일반 이미지 생성
    # general_infer(pipeline, args)

    # 벤치마크
    bench_dict = dict(
        pipeline=pipeline,
        args=args,
        before_input_image_path=f'{SAMPLE_PREFIX}/target00.jpg',
        after_input_image_path=f'{SAMPLE_PREFIX}/fused.jpg',
        after_scale=0.7
    )
    benchmark(make_before_and_after, bench_dict)
