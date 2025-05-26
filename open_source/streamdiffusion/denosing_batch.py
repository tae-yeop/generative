
def __init__()
    self.denoising_steps_num = len(t_index_list) # 여기서 


    # ---------- 전체 배치 사이즈 결정 --------------------
    # 배치파이 되었을 때 전체 배치 사이즈를 결정한다
    if use_denoising_batch:
        self.batch_size = self.denoising_steps_num * frame_buffer_size

        # 여기선 self.trt_unet_batch_size 라는 것을 준비한다, CFG랑 관련
        if 
# controlnet에선 step마다 control 정보가 들어가나?
# 일단 여기서 준비하는건 UNet에 들어가는 녀석
# prepare에서 denoising_steps_num > 1일때
@torch.no_grad()
def prepare():
    # buffer 준비, SD Turbor가 아니라면 미리 준비
    if self.denoising_steps_num >1:
        # batchify
        self.x_t_latent_buffer = torch.zeros(
            (
                (self.denoising_steps_num -1)  * self.frame_bff_size,
                4,
                self.latent_height,
                self.latent_width,
            ),
            dtpye=self.dtype,
            device = self.device
        )
    else:
        self.x_t_latent_buffer = None


    # denosing 쓰고 CFG도 full로 쓸것이므로 
    if self.use_denoising_batch and self.cfg_type == "full":
        uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)

    ...


    # ------------ timestep 설정 --------------------
    self.scheduler.set_timesteps(num_inference_steps, self.device)
    self.timesteps = self.scheduler.timesteps.to(self.device)
    
    self.sub_timesteps = []
    for t in self.t_list: # t 리스트에 있는 값으로 구성함, timestep 값으로 구성
        self.sub_timesteps.append(self.timesteps[t])
    
    # 이건 그냥 리스트에서 텐서로
    sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
    
    # denosing batch를 쓴다면 반복해서 time index를 만들어 준다
    # [999, 500, 200] => [999, 999, 500, 500, 200, 200]
    self.sub_timesteps_tensor = torch.repeat_interleave(
        sub_timesteps_tensor,
        repeats=self.frame_bff_size if self.use_denoising_batch else 1,
        dim = 0
    )

    ...

    alpha_prod_t_sqrt_list = []
    beta_prod_t_sqrt_list = []

    # sqrt를 걸어서 리스트에 모은다
    for timestep in self.sub_timesteps:
        alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
        beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
        alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
        beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
    
    # 리스트를 torch tensor로 만들고 shape를 [len, 1,1,1]
    alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
    )
    beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
    )

    # 디노이징 배치를 쓴다면 buffer 갯수만큼 반복함
    self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
    )
    self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
    )
    

    # ------------ noise 설정 ----------------------

    # [self.denoising_steps_num * frame_buffer_size, 4, 64, 64]
    self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)
    

    self.stock_noise = torch.zeros_like(self.init_noise)


    # ------------ LCM 셋업 --------------------
    c_skip_list = []
    c_out_list = []
    # 미리 모든 timestep에 대한 c_skip, c_out을 얻어놓음
    for timestep in self.sub_timesteps:
        # 원래 timestep을 넣으면 c_skp, c_out을 계산해주는 함수
        c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
        c_skip_list.append(c_skip)
        c_out_list.append(c_out)

    self.c_skip = (torch.stack(c_skip_list)
                   .view(len(self.t_list), 1,1,1)
                   .to(dtype=self.dtype, device=self.device)
                   )
    self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

def predict_x0_batch(self, x_t_latent: torch.Tensor):
    # x_T를 받는다 : (1, 4, self.latent_height, self.latent_width)
    # x_0을 예측하도록 한다
    prev_latent_batch = self.x_t_latent_buffer
    if self.use_denoising_batch:
        t_list = self.sub_timesteps_tensor
        # unet_step을 밟을 것인데 denoising step num에 맞춰서 이전 latent batch를 합쳐준다
        if self.denoising_steps_num > 1:

            # 배치 방향으로 
            # (self.denoising_steps_num -1)  * self.frame_bff_size 
            # + 1
            x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
            # 배치 방향으로 1 + self.denoising_steps_num * frame_buffer_size - 1 zero로 구성됨
            self.stock_noise = torch.cat((self.init_noise[0:1], self.stock_noise[:-1]), dim=0)

        x_0_pred_batch, model_pred = self.unet_step(x_t_latent, t_list)

        if self.denoising_steps_num > 1:
            # 제일 마지막꺼 뽑아내서 [4, 64, 64] 얻음
            x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
            # self.x_t_latent_buffer를 구한다
            if self.do_add_noise:
                self.x_t_latent_buffer = (
                    self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1] + + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                )
            else: # noise 안 더하는 경우
                self.x_t_latent_buffer = self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
        
        else: # 스텝 크기 없었따면 그냥 마지막에 바로 나오나 보네
            x_0_pred_out = x_0_pred_batch
            self.x_t_latent_buffer = None
    else:
        ...
    
    return x_0_pred_out



def scheduler_step_batch(self, model_pred_batch, x_t_latent_batch, idx=None):
    if idx is None:
        F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch)/self.alpha_prod_t_sqrt
        denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
    else:
        F_theta = (x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch)/self.alpha_prod_t_sqrt[idx]
        denoised_batch = self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch

    return denoised_batch



def unet_step(self, x_t_latent, t_list, idx=None):
    if self.guidance_scale > 1.0 and (self.cfg_type == "full"):
        x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
        t_list = torch.concat([t_list, t_list], dim=0)
    else:
        x_t_latent_plus_uc = x_t_latent
    
    model_pred = self.unet(
        x_t_latent_plus_uc,
        t_list,
        encoder_hidden_states=self.prompt_embeds, # 이미 계산해놓은거 들어감
        return_dict=False,
    )[0]

    if self.guidance_scale > 1.0 and (self.cfg_type == "full"):
        noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
    else:
        noise_pred_text = model_pred

    if self.guidance_scale > 1.0 and self.cfg_type != "none":
        model_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        model_pred = noise_pred_text

    # compute the previous noisy sample x_t -> x_t-1
    if self.use_denoising_batch:
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
    else:
        denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
    
    return denoised_batch, model_pred