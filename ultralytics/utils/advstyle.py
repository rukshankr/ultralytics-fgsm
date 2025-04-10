import torch

def advstyle_style_encode(self, x, eps=1e-5):
    mu = x.mean(dim=[2, 3], keepdim=True)
    std = x.std(dim=[2, 3], keepdim=True) + eps
    x_norm = (x - mu) / std
    return x_norm, mu, std

def advstyle_generate_adversarial_style(mu, std, model, x_norm, labels, gamma=1e-1, steps=1):
    mu_adv = mu.clone().detach().requires_grad_(True)
    std_adv = std.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([mu_adv, std_adv], lr=gamma)
    for _ in range(steps):
        x_adv = x_norm * std_adv + mu_adv
        pred = model(x_adv)
        loss = -torch.nn.functional.cross_entropy(pred, labels, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mu_adv.detach(), std_adv.detach()

def advstyle_apply(x_norm, mu_adv, std_adv):
    return x_norm * std_adv + mu_adv


# After Paper's Python code
def advstyle_forward(model, batch, adv_lr=1e-1):
    input = batch["img"].detach()  # shape: (B, C, H, W)
    B, C, H, W = input.size()

    # 1. Extract style features
    mu = input.mean(dim=[2, 3], keepdim=True).detach()
    var = input.var(dim=[2, 3], keepdim=True).detach()
    sig = (var + 1e-5).sqrt()

    # 2. Normalize image
    input_normed = (input - mu) / sig
    input_normed = input_normed.clone().detach()

    # 3. Set adv_mu and adv_sig as learnable
    adv_mu = mu.clone().detach().requires_grad_(True)
    adv_sig = sig.clone().detach().requires_grad_(True)
    adv_optim = torch.optim.SGD([adv_mu, adv_sig], lr=adv_lr)

    # 4. Adversarial step: generate image that *maximizes YOLO loss*
    adv_input = input_normed * adv_sig + adv_mu
    adv_batch = batch.copy()
    adv_batch["img"] = adv_input

    adv_optim.zero_grad()
    adv_loss, _ = model(adv_batch)  # this returns full loss dict
    loss_adv = adv_loss.sum()
    (-loss_adv).backward()
    adv_optim.step()

    # 5. Return final adversarial image
    adv_input = input_normed * adv_sig.detach() + adv_mu.detach()
    return adv_input
