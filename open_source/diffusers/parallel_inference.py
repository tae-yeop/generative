import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid


# 1) 분산 상태 초기화 ---------------------------------------------------------
state = PartialState()                     # rank, world_size, device 등을 담고 있음
torch.cuda.set_device(state.device)        # 안전 장치

# 2) 파이프라인 로드 & 메모리 최적화 ----------------------------------------
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
)


pipe.to(state.device)


# 3) 전체 작업 목록 정의 ------------------------------------------------------
prompts = [
    "A steampunk airship above neo-Tokyo at dusk, cinematic",
    "Retro-futuristic rover exploring a dusty red planet, high detail",
    "An ancient Korean palace under aurora night sky, ultra-wide",
    "Close-up portrait of a cyberpunk samurai with neon reflection",
    "A cozy studio full of vintage synthesizers and plants, soft light",
    "Low-poly isometric forest village, voxel art style",
    "Ultra-realistic macro photo of a dewdrop on a rose petal",
    "Concept art of underwater alien city with glowing corals",
]

# 각 프롬프트를 dict로 래핑해 job 리스트를 만든다
jobs = [
    {
        "prompt": p,
        "height": 512,
        "width": 512,
        "num_inference_steps": 30,
        "generator": torch.Generator(state.device).manual_seed(1234 + i),
    }
    for i, p in enumerate(prompts)
]

# 4) 프로세스별 작업 분배 ------------------------------------------------------
# (오타 주의! split_between_processes)
with state.split_between_processes(jobs) as my_jobs:
    local_paths = []

    for j, kwargs in enumerate(my_jobs):
        img = pipe(**kwargs).images[0]
        path = f"result_rank{state.process_index}_{j}.png"
        img.save(path)
        local_paths.append(path)

    # 5) 결과 경로를 모든 프로세스에서 모아 rank-0 로 전달 --------------------
    gathered = gather_object(local_paths, state)   # list[list[str]]

    # 6) rank-0 에서만 최종 로그 출력 ----------------------------------------
    if state.is_main_process:
        flat = [p for sub in gathered for p in sub]
        print(f"\n✓ Finished, saved {len(flat)} images:")
        for p in flat:
            print(" •", p)

# 7) 모든 프로세스 동기화 후 종료 --------------------------------------------
state.wait_for_everyone()