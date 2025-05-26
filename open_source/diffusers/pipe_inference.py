import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from diffusers.utils import load_image


def ip_adapter_lora():
    # vae 개별 모델을 따로 파이프라인에 로딩 할 수 있다
    pipe_id = "sd-dreambooth-library/herge-style"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipeline = StableDiffusionPipeline.from_pretrained(
        pipe_id,
        torch_dtype=torch.float16,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
        use_safetensors=True
    ).to("cuda")

    # satetensor가 있으면 from_single_file을 사용
    # pipeline = StableDiffusionPipeline.from_single_file(
    #     '/purestorage/project/tyk/9_Animation/MutliView/script/mixProV4_v4.safetensors',
    #     load_safety_checker=False).to('cuda')

    # 파이프라인 로딩 후 개별적인 모듈도 적용할 수 있다
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # 파이프라인의 ip adapter 적용 메소드, "h94/IP-Adapter" 허깅페이스의 폴더로 가서 weight를 다운 받아서 적용
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")
    pipeline.set_ip_adapter_scale(0.9)

    # 다중 lora 적용
    pipeline.load_lora_weights('/purestorage/project/tyk/9_Animation/MutliView/script/multiple views.safetensors', lora_scale=1.0)
    pipeline.load_lora_weights('/purestorage/project/tyk/9_Animation/MutliView/script/the_legend_of_korra_v2_offset.safetensors', lora_scale=1.0)


    image = load_image("/purestorage/project/tyk/9_Animation/MutliView/script/kora3.png").convert("RGB")
    generator = torch.Generator(device="cpu").manual_seed(233)

    image = pipeline(
        prompt="multiple views, 1girl, full body",
        ip_adapter_image=image,
        negative_prompt="lowres, bad anatomy, worst quality, low quality",
        num_inference_steps=100,
        generator=generator,
        guidance_scale=7
    ).images[0]

    image.save('test.png')

def animatediff():
    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
    from diffusers.utils import export_to_gif
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = "cuda"
    dtype = torch.float16

    adapter = MotionAdapter().to(device, dtype)
    # 허깅페이스 repo의 subfolder없이 바로 올려진 ckpt 다운
    repo = "ByteDance/AnimateDiff-Lightning"
    step = 4  # Options: [1,2,4,8]
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))

    base = "emilianJR/epiCRealism" # unet, vae 등 모두 subfolder로 있고 pipeline에선 이걸 다 처리
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    output = pipe(prompt="A girl smiling", guidance_scale=1.0, num_inference_steps=step)
    # duffusers에 gif 저장하는 함수
    export_to_gif(output.frames[0], "animation.gif")



def lcm_lora():
    from diffusers import DiffusionPipeline, LCMScheduler
    import torch

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16", 
                                            torch_dtype=torch.float16).to("cuda")

    # lora를 허깅페이스에서 받아서 로딩
    lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    # adapter 이름은 나중에 구분을 위한 사용자가 지정하는 옵션 값
    pipe.load_lora_weights(lcm_lora_id, adapter_name='lcm_lora',)

    # 특정 lora의 경우 scheduler를 바꾼다
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    prompt = "close-up photography of kpop girl, leica 35mm summilux"
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=1,
    ).images[0]

    # 추가 lora 로드
    # weight_name은 허깅페이스 스페이스 내의 파일 이름
    pipe.load_lora_weights("CiroN2022/toy-face", 
                        weight_name="toy_face_sdxl.safetensors", 
                        adapter_name="toy",)

    pipe.set_adapters(["lcm_lora", "toy"], adapter_weights=[0.5, 1.0])

    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=1,
        cross_attention_kwargs={"scale": 1.0}, 
        generator=torch.manual_seed(0)
    ).images[0]