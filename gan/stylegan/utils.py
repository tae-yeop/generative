import torch

def sample_truncated_z(batch, dim, tau=1.0, device="cuda"):
    z = torch.empty(batch, dim, device=device) # 빈 버퍼
    valid = torch.zeros(batch, dtype=torch.bool, device=device) # 조건 통과 여부를 표시하는 불리언 마스크

    while not valid.all(): # 아직 탈락자(row)가 남아 있으면 반복
        # 새로 뽑을 위치 인덱스
        idx = ~valid # 탈락자 인덱스만 추출 (logical NOT)
        # 표준 정규 샘플링
        z[idx] = torch.randn(idx.sum(), dim, device=device)
        valid[idx] = (z[idx].abs() <= tau).all(dim=1) # 조건 검사 → mask 갱신
    return z
