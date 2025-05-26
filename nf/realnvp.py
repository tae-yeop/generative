# RealNVP(Real-valued Non-Volume Preserving)
import torch
import torch.nn as nn

if __name__ == "__main__":
    data = torch.cat([
        torch.randn(500, 2) + torch.tensor([3.0, 0.0]),
        torch.randn(500, 2) + torch.tensor([-3.0, 0.0]),
    ], dim=0)


    flow_model = RealNVP(in_features=2, hidden_features=64, n_coupling_layers=6)
    optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)

    epochs = 500
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        log_prob = flow_model.log_prob(data)
        loss = -log_prob.mean()  # 최대우도추정을 위해 음의 log-likelihood 최소화
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
                print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")


    with torch.no_grad():
        # base distribution에서 표준정규 샘플 100개
        z = torch.randn(100, 2)
        # flow의 역변환으로 x 공간에서의 샘플을 얻음
        samples, _ = flow_model.inverse(z)
        print("샘플 shape:", samples.shape)