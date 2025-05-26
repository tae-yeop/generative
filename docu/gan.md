Research Avenue

(1) Better Architecture
- Better discriminator
    - auxilary classifier
- Better generator
    - handle artifacts
        - no max-pooling
        - Resize+Conv over transposedConv
        - Bilinear or resize
    - Network capacity
    - long range dependancy

(2) Better objectives and regularizations
- one-sided label smoothing
- Stable Traning
    - Vanishing gradients
(3) Contol and Interpretability



## Discriminator 관련 기법

## Post-Sampling

Discriminator Rejection Sampling
- Generator가 생성한 샘플중 고품질만 뽑도록 하자
- Discriminator가 높은 real 확률을 부여한 샘플만 선택하면 된다

Rejection Sampling은 다음과 같이 수행한다.

이때 이 값을 Dissctiorot




data distribution $p_d(x)$에서 샘플하기 어려움

대신에 Proposal distribution $p_g(x)$에서 샘플링을 하자

$$
\exists M < \infty s.t \\
Mp_g(x) > p_d(x) , \forall x \in \text{supp}(p_d)
$$

이 경우 다음과 같은 accept prob으로 $p_d$에 대한 샘플을 생성할 수 있다

$$
\displaystyle \frac{p_d (x)}{Mp_g  (x)}
$$

GAN에선 generator가 proposal dist가 된다

그리고 true data distribution을 매칭하려고 한다

이때 Rejection Sampling을 통해 $p_d$와 매칭되지 않음에도 불구하고 $p_d$로 부터 샘플을 생성하도록 한다

하지만 $p_g(x), p_d(x)$를 계산할 수 없다.

대신에 Discriminator를 이용해서 likelihood ratio을 estimate할 수 있다

D는 $x \sim p_{mix}$를 받도록 함 $p_{mix}$는 data distribution $p_d$와 generator distribution $p_g$의 balanced mix이다

D를 학습할 수 있다면 

$$
p_{mix}(x) = \frac{1}{2}p_{d}(x) + \frac{1}{2}p_{g}(x) \\

D(x) = p(\text{real}|x) \\
= \frac{p(x|\text{real}) p(\text{real})}{p_{mix}(x)} \\
= \frac{p_d(x) \times \frac{1}{2}}{\frac{1}{2}p_d(x) + \frac{1}{2}p_g(x)} \\
= \frac{p_d(x)}{p_d(x) + p_g(x)} \\
= \frac{1}{1 + \frac{p_g
    (x)}{p_d (x)}} 
$$

보통 Discriminator는 logit $\tilde{D}$의 sigmoid로 출력

$$
\displaystyle D (x) = \sigma (\tilde{D} (x)) =
    \frac{1}{1 + e^{- \tilde{D} (x)}}
$$

$$
\displaystyle \frac{p_d (x)}{p_g (x)} = \exp (\tilde{D} (x))
$$

상수 $M$을 결정하면 된다!!

이는 likelihood ratio의 최대값으로 결정

$$
M = \max_x \frac{p_d (x)}{p_g (x)} = \max_x \exp
    (\tilde{D} (x))
$$

실제로는 generator에서 계속 샘플링을 해서 $M$을 찾는다 : $x \sim p_g$이

ㅇ

이후 다음 accept prob으로 rejection sampling을 수행

$$
\frac{\exp(\tilde{D}(x))}{M}
$$



