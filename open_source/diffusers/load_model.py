from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel



def load_model_from_huggingface():
    # 주소가 이렇다면 https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    # "stabilityai/stable-diffusion-xl-base-1.0" 이부분을 사용
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

# =========================================================================== #
# =========================================================================== #
def load_pipeline_with_revision():
    # 파이프라인의 경우 깃의 브랜치 값을 넣으면 됨
    # 그외에 discussion에 올라온 PR을 사용하는 용도로도 사용됨
    pipeline = DiffusionPipeline.StableDiffusionPipeline(
        "runwayml/stable-diffusion-v1-5",
        custom_pipeline="clip_guided_stable_diffusion",
        custom_revision="main", # "v0.25.0",
        clip_model=clip_model,
        feature_extractor=feature_extractor,
        use_safetensors=True,
    )

def load_model_with_reivision():
    # 만약 허깅페이스 https://huggingface.co/CompVis/stable-diffusion-v1-4/discussions/171의 내용을 쓰고 싶다면
    # revision = f"refs/pr/171" 넣음
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=f"refs/pr/171",
        cache_dir='/purestorage/project/tyk/tmp'
    )
    for name, children in unet.named_children():
        print(name, type(children))


    print(unet.add_embedding)

# =========================================================================== #
# =========================================================================== #
# safetensor 직접 로딩하기
# import torch

# safetensor를 바로 torch.load하려면 에러가 남
# safe_tensor_path = "/purestorage/project/tyk/tmp/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/76d28af79639c28a79fa5c6c6468febd3490a37e/unet/diffusion_pytorch_model.safetensors"
# # sd = torch.load(safe_tensor_path, map_location="cpu")
# # print(type(sd))


# from safetensors import safe_open

# tensors = {}
# with safe_open(safe_tensor_path, framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)

# print(tensors.keys()) 

# add_embedding.linear_1.weight
# 'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2.weight'
# 이런식으로 키가 잡히기 때문에 attribute name이 잘 맞아 떨어지고 weight, bias만 맞으면 될듯


def open_from_bin(unet_resume_path):
    optimizer_fp = os.path.join(unet_resume_path, "optimizer.bin")
    if os.path.exists(optimizer_fp):
        optimizer_resume_path = optimizer_fp

    unet = UNet3DConditionModel.from_pretrained(args.unet_resume_path,
                                                subfolder="unet",
                                                low_cpu_mem_usage=False,
                                                device_map=None)


# =========================================================================== #
# =========================================================================== #
# config 얻기

config = UNet2DConditionModel.load_config(
    "stabilityai/stable-diffusion-xl-base-1.0",
    cache_dir='/purestorage/project/tyk/tmp',
    subfolder="unet"
)

# 구성 출력
print(config)


def load_model_with_hugging_login():
    from huggingface_hub import login
    login(token="your access token")

    from transformers import AutoTokenizer
    import transformers
    import torch

    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir='/purestorage/project/tyk/tmp')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
