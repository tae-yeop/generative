Continuous Normalizing Flows(CNF)를 simulation 없이 학습하도록 하자. 즉 기존의 디퓨전 학습처럼 배치 단위로 Noise를 씌워서 학습할 수 있게 하자. 

probability density path : $p : [0,1] \times \Bbb{R}^d \to \Bbb{R}_{>0}$

flow라는 것은 다음과 같이 ODE로 정의
$$
\frac{d}{dt} \phi_t(x) = v_t(\phi_t(x))
$$
$$
\phi_0(x) = x
$$

Neural ODE에선 vector filed $v_t$를 신경망으로 모델링하였음 ⇒ $v_t(x;\theta)$

이게 발전해서 flow