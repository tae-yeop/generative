# Diffusion Model 개요

## Diffusion Model 이란?

![Image](https://github.com/user-attachments/assets/7f6d9704-cb9a-4e8c-8817-38b6edfeab5e)


Diffusion model $p_{\theta}(x_0)$은 다음과 같이 정의되는 Reverse process로 구성된 latent model이다. 

$$
p_{\theta}(x_{t-1}|x_t) := \mathcal{N}(x_{t-1} ; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t)) : \text{trainsition kernel}\\

p_{\theta}(x_{0:T}) := p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t) : \text{Parameterized Reverse Diffusion process} \\

p(x_T) = \mathcal{N}(x_T; \mu_0, \Sigma_0)\\

p_{\theta}(x_{0}) := \int p_{\theta}(x_{0:T}) dx_{1:T} : \text{Diffusion model}
$$

다른 모델처럼 Prior $p(x_T)$에서 real data 분포가 되게끔 학습된 모델이다. 이때 Markov chain으로 정의된 Reverse process는 transition kernel의 곱들 $p_{\theta}(x_{t-1} | x_t)$로 표현된다.


그리고 Forward process는 미리 특정한 확률적인 다이나믹스를 사전에 정의한 Stochastic process로 표현하도록 한다. 이 또한 Markov Chain으로 표현할 수 있다. 

$$
q(x_{1:T} | x_0) = \prod_{t \geq 1} q(x_t | x_{t-1}) \\
q(x_t |x_{t-1}, x_0) = N(x_t ; \sqrt{1-\beta_t}x_{t-1}, \beta_tI)
$$

이 때 개별적인 한번 움직임을 가우시안으로 두도록 한다. 

$$
q(\mathbf{x}_{t}|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_{t} ; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I}) : \text{Forward process} \\

\beta_t \in \{ \beta_1, \dots \beta_T\}

$$

그러면 이로 부터 one step posterior $q(x_{t-1} | x_t, x_0)$를 생각할 수 있는데 다른 latent variable와 차이점이 바로 이를 근사화하는 것이다. 

$$
q(x_{t-1} | x_t, x_0) := \mathcal{N}(x_{t-1} ; \sqrt{\alpha_t}x_t, (1-\alpha_t)\mathbf{I})
$$

이를 통해 한방에 Prior에서 Data 분포로 보내는게 아니라 훨씬 쉬운 샘플링의 연속(sequence of easier sampling problems)으로 접근할 수 있다.

![Image](https://github.com/user-attachments/assets/b64919b8-df44-4714-a168-ef3ce0c662dc)


이때 latent modeling 관점에서의 posterior은 $q(x_{1:T} | x_0)$로 표현되는데 헷갈린다.
$$
q(\mathbf{x}_{1:T}|\mathbf{x}_{0}) := \prod_{t=1}^T q(\mathbf{x}_{t}|\mathbf{x}_{t-1}) 
$$



흥미롭게도 이런 포뮬레이션을 하면 다음과 같은 사실들이 보장이 되고 따라서 생성 모델로써 사용할 수 있음을 알 수 있다. 
어쨌든 아이디어는 각각의 $x_t$가 존재하는 분포 $p_t$가 결국 있을텐데 근방의 분포 $(p_{t-1}, p_t)$는 e marginally “close” in some appropriate sense.

$$
p_0 , p_1 , p_2 , . . . , p_T,
$$

이후 Revers process를 밟을 수 있는 Reverse Sampler를 학습하는데 이는 분포 $p_t$를 $p_{t-1}$로 바꿔준다. 위에서 말한 Gaussian forward process를 따르고 $\beta_t$가 작다면 $q(x_{t-1}|x_t)$도 gaussian이 된다는게 알려져있다. 즉 gaussian이니 mean parameter $\mu \in \Bbb{R}^d$를 찾을 수 있고 Reverse sampler를 써서 $p_0$에서 올법한 녀석들을 만들어낼 수 있다.


> with sufficiently small time steps, reverse process $q(x_{t-1}|x_t)$ can be approximated by $ p_{\theta}(x_{t-1}|x_t)$


참고로 Revers Sampler는 다음과 같은 함수 $F_t$인데 $x_t \sim p_t$가 주어졌을 때 $F_t(x)$의 marginal distribution이 $p_{t-1}$이 되도록 하는 함수이다.

$$
\{ F_t(z) : z \sim p_t \} = p_{t-1}
$$

정리하면 데이터가 Gaussian Noise로 변화하는 Process를 생각하고 이로 부터 역방향 Process를 모델로 근사화시키면 우리가 원하는 생성 모델, 즉 Gaussian Noise에서 데이터를 샘플링할 수 있는 함수를 얻을 수 있다는게 요지이다. 

물리적인 현상을 확률적으로 표현할 수 있었던 것을 적용한 사례. - Gas molecule이 high densitiy → low densitiy 이동 : increase entropy
- 이 현상이 정보 이론에선 loss of information과 equivalent함
    - gradual intervention of noise



## 수학적 지식

[1] VIPs of Gaussian Distribution

Any Gaussian distribution can be implemented by standard Gaussian with a
translation by mean and a scaling by standard deviation (covariance matrix)

$$
X \overset{d}{=} \mu + \sigma Z, X \sim \mathcal{N}(\mu, \sigma^2), Z \sim \mathcal{N}(0, 1)\\

X \overset{d}{=} \mu + \Sigma^{\frac{1}{2}} \bold{Z} \bold{X} \sim \mathcal{N}(\mu, \Sigma), \bold{Z} \sim \mathcal{N}(0, I), 

$$

$\overset{d}{=}$ : 분포적인 관점에서 같다는 뜻, 값이 같다는 뜻이 아님!
![Image](https://github.com/user-attachments/assets/2ac19200-cc56-4125-940e-dc6594bf3b12)

Any linear combination of independent Gaussian distributions follows Gaussian

$$

X_1 \sim \mathcal{N}(\mu_1, \Sigma_1), Y = aX_1 + bX_2 \to Y \sim \mathcal{N}(a\mu_1 + b \mu_2, a^2 \Sigma_1 + b^2 \Sigma_2) \\
X_2 \sim \mathcal{N}(\mu_2, \Sigma_2), Y = AX_1 + BX_2 \to Y \sim \mathcal{N}(A\mu_1 + B\mu_2 , A\Sigma_1A^T + B\Sigma_2B^T)

$$


[2] Kullback-Leibler Divergence between Gaussians
% ⇠ N (`?, ⌃?) & ⇠ N (`@, ⌃@)



[3] Markov Process


### Forward Process

앞에서 정의했듯이 Forward process의 한칸 transition은 다음과 같다.

$$
q(\mathbf{x}_{t}|\mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_{t} ; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I}) : \text{Forward process} \\

\beta_t \in \{ \beta_1, \dots \beta_T\}
$$




### Backward Process


### 학습과 추론

![Image](https://github.com/user-attachments/assets/de9e865c-d209-475a-af74-21971087fd75)

위의 포뮬레이션을 보면 VAE와 유사한데 실제로 Markovian Hierarchical Variational Autoencoder의 일종이다. 좀 더 구체적으론 (1) latent dimension은 데이터 차원과 같고 `z.shape == x.shape` (2) 각 step의 encoder의 Gaussian parameters들은 학습되지 않고 고정인 형태인 스폐셜 케이스이다. 특히 (3) encoder의 Gaussian parameters은 마지막 T step에서 standard Gaussian이 됨을 보장하게끔 구성됨  : $p(x_T) \sim \mathcal{N}(0, I)$. 

(3)의 특성을 만족하는 셋팅은 분명 유일하진 않다. DDPM의 경우 다음과 같은 셋팅을 사용했는데 이는  latent variable의 variance가 같은 스케일로 유지하는 효과가 있어서 variance-preserving이라고 불림

$$
\mu_t(x_t) = \sqrt{\alpha_t}x_{t-1} \\

\Sigma_t(x_t) = (1 - \alpha_t) \rm{I}
$$


최초로 제안했던 Sohl-Dickstein et al.에선 매번 단계별로 posterior를 네트워크 $p_{\theta}$가 모두 맞춰야 했음. 반면에 DDPM에선  $x_0$에서 $x_t$를 예측하도록 네트워크를 구성. DDPM에선 어떤 시점이던지(노이즈가 얼만큼 주어지던지 간에) $x_0$를 예측하도록 한다 ⇒ 네트워크의 부담이 줄어듬. 막상 해보니 


학습은 결국 간단한게 미니배치에서 뽑은 데이터셋 $x_0$에 noise를 추가한 뒤 모델에 추론을 하게 만들고 예측값과 MSE loss를 계산하도록 한다. 

```python
def p_losses(self, x_start, t, noise=None):
	"""
	Args:
		데이터에서 공급받은 x_start
	"""
	noise = torch.randn_like(x_start)
	x_noisy = self.q_sample(x_start=x_start, noise)
	model_out = self.model(x_noisy, t)

	# 네트워크 예측값을 noise로 할 것인지, x_start로 할것인지?
	if self.parameterization == 'eps':
		target = noise
	elif self.parameterzation == 'x0':
		target = x_start
	elif self.parameterzation == 'v':
		target = self.get_v(x_start, noise, t)
		
	loss = self.get_loss(model_out, target)
	loss_simple = loss.mean() * self.l_simple_weight

	loss_vlb = (self.lvlb_weights[t] * loss).mean()
	
	loss = loss_simple + self.original_elbo_weight * loss_vlb
```

학습에 필요한 함수는 `add_noise(q_sample)`이 필요하다.


추론에는 `step(p_sample)`이 필요하다.




## SDE

![Image](https://github.com/user-attachments/assets/52dd10ca-ef58-4814-a67d-8f022fbce53d)

알고 봤더니 관련성이 있는 걸로 알려짐. SDE라는 통합된 프레임워크 안에서 모두 생각할 수 있게 됨. 2019년에 NSCS를 쓰고 Jonahtan Ho가 loss function이 비슷한것 같다라고 언급. 이후 2020년 ICRL 에 [Y.song](http://Y.song) 논문. 이후 Diffusion model, 이라고 부름

- 2015년 Sohl-Dickstein et al.,은 Diffusion model을 최초로 제시한다.
- 2020년 Jonathan Ho et al.,은 Diffusio model을 개선한 DDPM을 제시하여
- 이와 별개로 Yang Song et al.,은 EBM 모델을 개선하기 위한 연구를 하고 있었다. 그렇게 해서
- Fundamental적인 foundation은 현재 3가지이다 : DDPM, SGM, Score SDE
- Sohl-Dickstein et al.은 Generative model에 dissfusion을 도입한 최초의 연구을 보여주었으나 성능 자체는 좋지 않았다 : Diffusion model을 위한 이론적 발판을 구축
- 이후 Jonathan Ho et al.,은 DDPM을 기초로도 고해상도 생성이 가능함을 보여주었다
- Yang Song et al.,은 Score 기반의 foundation을 제공하였다
- Yang Song et al.,은 Score SDEs을 통해 Score 기반 이론과 DDPM 기반 이론을 SDE 프레임워크 아래 모두 통합하였다

![Image](https://github.com/user-attachments/assets/56af205c-fcdf-4616-8f1d-6a4e5c52fd5d)



### Stochastic Differential Equation

먼저 Differential Equation는 시간에 따른 어떤 변화하는 대상 X와 이에 대한 관계를 나타내는 f가 있다면 다음과 같이 나타낼 수 있음. 

$$
\Delta t = t_{k+1} - t_k
$$
$$
X_{t_{k+1}} = X_{t_k} + f(t_k, X_{t_k})\Delta t
$$
$$
\to (equivalent) \frac{X_{t_k+\Delta t} - X_{t_k}}{\Delta t} = f(t_k, X_{t_k})
$$
$$
\lim_{\Delta t \rightarrow 0} \to \frac{dX_t}{dt} = f(t, X_t)
$$

여기서 Noise term이 추가되어 다음 타임 step에 변화가 있다면 다음과 같이 나타낼 수 있음. 이게 SDE임. Noise term은 원래 미분이 안되지만 미분이 되는것 처럼 다룸.

$$
\Delta t = t_{k+1} - t_k \quad \text{and} \quad \Delta N_{t_k} = N_{t_{k+1}} - N_{t_k} : \text{noise increment}
$$
$$
X_{t_{k+1}} = X_{t_k} + f(t_k, X_{t_k})\Delta t + g(t_k, X_{t_k})\Delta N_{t_k}
$$
$$
\to \frac{X_{t_k+\Delta t} - X_{t_k}}{\Delta t} = f(t_k, X_{t_k}) + g(t_k, X_{t_k})\frac{\Delta N_{t_k}}{\Delta t}
$$
$$
\lim_{\Delta t \rightarrow 0} \to \frac{dX_t}{dt} = f(t, X_t) + g(t, X_t)\dot{N}_t
$$

이때 마지막 수식을 Intergral 형태로 풀면 다음과 같음 (stochastic integration)

$$
X_t = X_0 + \int_0^t f(s, X_s)ds + \int_0^t g(s, X_s)dN_s
$$

즉 마지막 이 수식을 이용해서 SDE를 풀 수 있음. 

그럼 이런 SDE 개념을 어디에 적용할 수 있을까? 앞에서 보았던 Diffusion Model에 Markov Chain에 대해 다음과 같이 적을 수 있다.

$$
X_t | X_{t-1} \sim \mathcal{N}(\sqrt{1-\beta_t}X_{t-1}, \beta_t I) \quad \text{Discrete-time Markov Chain } t \in \{0, 1, \dots, N\}
$$
$$
\to X_t = \sqrt{1-\beta_t}X_{t-1} + \sqrt{\beta_t}\epsilon_{t-1} \quad \epsilon_{t-1} \sim \mathcal{N}(0,I)
$$
$$
\to (discretization) X_{t_{k+1}} = X_{t_k} - \frac{1}{2}\beta_{t_k}X_{t_k}\Delta t + \sqrt{\beta_{t_k}\Delta t}\epsilon_{t_k} \quad \Delta t = \frac{1}{N}
$$
$$
\lim_{\Delta t \rightarrow 0} \to dX_t = -\frac{1}{2}\beta_t X_t dt + \sqrt{\beta_t}dW_t \quad \text{Continuous-time SDE } t \in [0,1]
$$

즉 알고 봤더니 Makov Chain은 SDE 형태로 나타낼 수 있었음. 그렇다고 적분 수식을 활용하는건 힘들고 discretization을 활용하여 SDE solver를 구현할 수 있다. 그리고 SDE는 알고 보면 어떤 변화하는 과정의 Chain에 대한 또 다른 표현이라는 점.


### Langevin MCMC
Langevin Monte Carlo는 MCMC 알고리즘의 일종으로 관심있는 분포에서 샘플을 생성하게 해줌. 이때 Langevin Equation이라는것을 풀면서 진행됨

$$
\lambda\frac{dX_t}{dt} = -\frac{\partial V(x)}{\partial x} + \eta(t)
$$

이는 SDE 폼으로 다음과 같이 적을 수 있음


### SDE

SDE의 일반식과 이에 대한 Reverse SDE를 다음과 같이 적을 수 있음. Reverse SDE의 경우는 Score function이 필요함. 이런 Score function은 Forward SDE로 부터 유도할 수 있음. 
$$
dX_t = f(t, X_t)dt + \sigma(t)dW_t
$$
$$
d\hat{X}_t = (-f(T-t,\hat{X}_t) + \sigma^2(T-t)S(T-t,\hat{X}_t))dt + \sigma(T-t)dW_t
$$



Score model에서 네트워크 $s_{\theta}(x_t, t)$는 score function $\nabla_{x_t} \log p(x_t)$를 예측하게 된다. 이는 특정 noise level $t$에서 data space에서 $x_t$의 gradient를 예측하는 것이다. Tweedie formula와 reparameterization trick을 같이 사용 ⇒ score function이 source noise $\epsilon_0$와 유사함을 알 수 있다

$$
\mathbf{x}_0 = \frac{\mathbf{x}_t + (1 - \bar{\alpha}_t)\nabla \log p(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}_t}}
$$
$$
\therefore (1 - \bar{\alpha}_t)\nabla \log p(\mathbf{x}_t) = -\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0
$$
$$
\nabla \log p(\mathbf{x}_t) = -\frac{1}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_0
$$


Forward SDE

$$
dX_t = d(t, X_t) dt + \sigma(t) dW_t \\

\to dX_t = - \frac{1}{2} \beta_t X_t dt + \sqrt{\beta_t} dW_t \\

d\hat{X}_t = [-f(T-t, \hat{X}_t) - \sigma^2(T-t)S(T-t, \hat{X}_t)]dt + \sigma(T-t) dW_t \\

\to - [-\frac{1}{2} \beta_{T-t} \hat{X}_t + \beta_{T-t} S(T-t, \hat{X}_t)]dt + \sqrt{\beta_{T-t}}
dW_t$$

여기서 $S(\cdot)$이 SGM의 score model이다. 

$$
L_t(\theta) = ||s_{\theta} (t, z_t) - \nabla_{z_t} \log q(z_t | x) || \\

L_t(\theta) = || \epsilon_{\theta}(t, z_t) - \epsilon || \\

q(z_t | x) = \mathcal{N} (\sqrt{\alpha}X, (1-\alpha_t)I) \implies z_t(x) = \sqrt{\alpha_t}x + \sqrt{1-\alpha_t}\epsilon
\\
\to

\nabla_{z_t} \log q(z_t | x) = \nabla_{z_t} [-\frac{1}{2} \frac{(z_t - \sqrt{\alpha_t x})^T(z_t - \sqrt{\alpha_t x})}{1 - \alpha_t}] = - \frac{z_t - \sqrt(\alpha_t)x}{1-\alpha_t} = - \frac{\epsilon}{\sqrt{1 - \alpha_t}}
$$

Reverse MC를 학습시킨 것과 score model 학습시킨건 같은 target을 가지고 한다. 차이점은 t를 연속 하지 않음
$t=[0,1]$ 가 연속이면 샘플링 속도를 빠르게 할 수 있다.





미분방정식을 풀 수 있다. 

성능을 결정하는 건 $\beta_t$ (variance scheduling)

data가 없는 영역에선 score 관찰 x \to 이 부분 score model 학습 x.
데이터가 없는 영역에선 $\beta_t$로 pertubation된 score function을 쓰고 데이터가 가까운 영역에선
원래 score function을 써서 가깝게 한다.




$f$ (drift term) : 평균적으로 어디로 흘러가야할지 결정, 시그널 역할
$\sigma$ : noise

SNR : forward pass에서 정보가 얼만큼 파괴되는지 척도

$$
X_t = X_0 e^{-\frac{1}{2} \int_0^t \beta_s ds} + \int_0^t X_0 e^{-\frac{1}{2} \int_0^s \beta_s ds} \sqrt{\beta_s} dW_s \\

dX_t = -\frac{1}{2} \beta_t X_t dt + \sqrt{\beta_t} dW_t
$$


PF-ODE

$$
\frac{dX}{dt} = f(X, t) \\

f(X, t) = f(X, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(x)
$$



DDPM = A discretized special case of the reverse-time SDE

DDIM = A discretized special case of PF-ODE


### VP-SDE
- 구현 간단
- 퀄리티 보장 (그나마 안정적)
- acceleration ok
- DDPM의 continuous version (DDPM과 연결점이 있다)

$$
dX_t = f(t, X_t) dt + \sigma(t) dW_t \\

f(t, X_t) = -\frac{1}{2} \beta_t X_t \\
\sigma(t) = \sqrt{\beta_t} \\

dX_t = -\frac{1}{2} \beta_t X_t dt + \sqrt{\beta_t} dW_t
$$

closed-form solution을 구할 수 있음
VP-SDE는 maruyama를 써서 iterative하게 할 필요없이 한번에 구할 수 있다.

$f, g$가 high-frequency singal, low-frequency signal을 복원할지 결정한다.

Variance-exploding $\to$ high-frequency signal을 복원이 어려움 $\to$ 퀄리티 떨어짐


# 대표 연구
DDPM
Diffusion Beat GAN (ADM, ADM-G(Guided diffusion))
IDDPM


# 샘플링 속도 향상


FlashDiffusion 


# 자료

[Understanding Diffusion Models: A Unified Perspective, Calvin Luo,](https://arxiv.org/pdf/2208.11970.pdf)

https://alexxthiery.github.io/posts/reverse_and_tweedie/reverse_and_tweedie.html


- [https://www.icts.res.in/sites/default/files/paap-2019-08-08-Eric Moulines.pdf](https://www.icts.res.in/sites/default/files/paap-2019-08-08-Eric%20Moulines.pdf)
- https://abdulfatir.com/blog/2020/Langevin-Monte-Carlo/