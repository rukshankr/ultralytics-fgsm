import torch
import torch.nn.functional as F

def compute_style(x):
    mu = x.mean(dim=[2, 3], keepdim=True)
    sigma = x.std(dim=[2, 3], keepdim=True) + 1e-5
    x_norm = (x - mu) / sigma
    return mu, sigma, x_norm

def restyle(x_norm, mu, sigma):
    return x_norm * sigma + mu

def advstyle_step_yolo(model, images, targets, loss_fn, gamma=0.5, inner_steps=1):
    model.eval()  # Freeze model
    mu, sigma, x_norm = compute_style(images)
    mu_adv = mu.clone().detach().requires_grad_(True)
    sigma_adv = sigma.clone().detach().requires_grad_(True)
    optimizer_adv = torch.optim.SGD([mu_adv, sigma_adv], lr=gamma)

    for _ in range(inner_steps):
        x_adv = restyle(x_norm, mu_adv, sigma_adv)
        preds_adv = model(x_adv)[0]  # [0] returns predictions, [1] would return loss
        loss_adv = -loss_fn(preds_adv, targets)  # maximize loss
        optimizer_adv.zero_grad()
        loss_adv.backward()
        optimizer_adv.step()

    model.train()
    x_adv_final = restyle(x_norm, mu_adv.detach(), sigma_adv.detach())
    preds_orig = model(images)[0]
    preds_adv = model(x_adv_final)[0]

    loss_orig = loss_fn(preds_orig, targets)
    loss_adv = loss_fn(preds_adv, targets)
    total_loss = loss_orig + loss_adv
    return total_loss
