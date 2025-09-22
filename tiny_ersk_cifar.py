#!/usr/bin/env python3
import argparse, time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.autograd.functional import jvp

# ─────────────────────────────────────────────────────────────────────────────
#  ESRK-15 tableau (your coefficients)
# ─────────────────────────────────────────────────────────────────────────────
a_np = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0243586417803786,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0258303808904268,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0667956303329210,0,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0140960387721938,0,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0412105997557866,0,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0149469583607297,0,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.414086419082813,0,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00395908281378477,0,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.480561088337756,0,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.319660987317690,0,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0.00668808071535874,0,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.0374638233561973,0,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.422645975498266,0.439499983548480,0,0],
    [0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.0358989324994081,0.00661245794721050,0.216746869496930,0,0.422645975498266,0.0327614907498598,0.367805790222090,0]
]
b_np = [0.035898932499408134,0.035898932499408134,0.035898932499408134,0.035898932499408134,
        0.035898932499408134,0.035898932499408134,0.035898932499408134,0.035898932499408134,
        0.006612457947210495,0.21674686949693006,0.0,0.42264597549826616,
        0.03276149074985981,0.0330623263939421,0.0009799086295048407]
A = torch.tensor(a_np, dtype=torch.float32)
b = torch.tensor(b_np, dtype=torch.float32)
c = A.tril(-1).sum(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Activations & Vector field
# ─────────────────────────────────────────────────────────────────────────────
class ApproxSiLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, -4, 4)
        return x * (0.5 + 0.25*x - (1/12)*x**2 + (1/48)*x**3)

def make_f(ch, use_groupnorm=False, approx_act=False):
    act = ApproxSiLU() if approx_act else nn.SiLU()
    layers = [nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    if use_groupnorm:
        layers.append(nn.GroupNorm(8, ch))
    layers += [act, nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    net = nn.Sequential(*layers)
    class VF(nn.Module):
        def __init__(self):
            super().__init__(); self.net = net
        def forward(self, t, x):  # t kept for API parity
            return self.net(x)
    return VF()

# ─────────────────────────────────────────────────────────────────────────────
#  Integrator blocks: Euler / RK4 / ESRK-15 (general explicit RK)
# ─────────────────────────────────────────────────────────────────────────────
class EulerBlock(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f,self.h,self.steps=f,h,steps
    def forward(self, x, t0=0.0):
        t=t0
        for _ in range(self.steps):
            x = x + self.h * self.f(t, x); t += self.h
        return x

class RK4Block(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f,self.h,self.steps=f,h,steps
    def _step(self, x, t):
        h, f = self.h, self.f
        k1 = f(t, x)
        k2 = f(t + 0.5*h, x + 0.5*h*k1)
        k3 = f(t + 0.5*h, x + 0.5*h*k2)
        k4 = f(t + h,     x + h*k3)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    def forward(self, x, t0=0.0):
        t=t0
        for _ in range(self.steps):
            x = self._step(x, t); t += self.h
        return x

class ESRKBlock(nn.Module):
    """Generic explicit RK using full lower-triangular A (correct for your tableau)."""
    def __init__(self, f, A, b, c, h=1.0, steps=1):
        super().__init__()
        self.f,self.h,self.steps=f,float(h),int(steps)
        A_l = torch.as_tensor(A, dtype=torch.float32).tril(-1)
        self.register_buffer("A_l", A_l)
        self.register_buffer("b",   torch.as_tensor(b, dtype=torch.float32))
        self.register_buffer("c",   torch.as_tensor(c, dtype=torch.float32))
        self.S = A_l.shape[0]
    def _step(self, x0, t):
        h, A_l, b, c, S = self.h, self.A_l, self.b, self.c, self.S
        k = [None]*S
        for i in range(S):
            xi = x0
            if i>0:
                acc = 0.0
                for j in range(i):
                    aij = A_l[i,j]
                    if float(aij) != 0.0:
                        acc = acc + aij * k[j]
                xi = x0 + h * acc
            ti = t + c[i]*h
            k[i] = self.f(ti, xi)
        return x0 + h * sum(b[i]*k[i] for i in range(S) if float(b[i])!=0.0)
    def forward(self, x, t0=0.0):
        t=t0
        for _ in range(self.steps):
            x = self._step(x, t); t += self.h
        return x

# ─────────────────────────────────────────────────────────────────────────────
#  Tiny NODE backbone
# ─────────────────────────────────────────────────────────────────────────────
class TinyNODE(nn.Module):
    def __init__(self, solver='esrk', width=64, use_groupnorm=False, approx_act=False,
                 h=1.0, steps=1):
        super().__init__()
        act = ApproxSiLU() if approx_act else nn.SiLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            act,
            nn.MaxPool2d(2)  # 32->16
        )
        f = make_f(width, use_groupnorm, approx_act)
        if solver == 'euler':
            self.ode = EulerBlock(f, h=h, steps=steps)
        elif solver == 'rk4':
            self.ode = RK4Block(f, h=h, steps=steps)
        elif solver == 'esrk':
            self.ode = ESRKBlock(f, A, b, c, h=h, steps=steps)
        else:
            raise ValueError(f"Unknown solver: {solver!r}")
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 10, bias=False)
        )
    def forward(self, x):
        return self.head(self.ode(self.encoder(x)))

# ─────────────────────────────────────────────────────────────────────────────
#  Spectral-norm estimate via JVP power iteration on the vector field
# ─────────────────────────────────────────────────────────────────────────────
def estimate_spectral_norm(model, x, iters=5):
    model.eval()
    with torch.no_grad():
        z = model.encoder(x)
    z = z.detach().requires_grad_(True)
    def vf(z_in): return model.ode.f(0.0, z_in)
    v = torch.randn_like(z)
    sigma = 0.0
    for _ in range(iters):
        _, jv = jvp(vf, (z,), (v,), create_graph=False)
        sigma = jv.norm().item()
        v = jv / (sigma + 1e-12)
    return sigma

# ─────────────────────────────────────────────────────────────────────────────
#  Exact gradient dump helper
# ─────────────────────────────────────────────────────────────────────────────
def dump_exact_grads(model, x, y, path):
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()  # exact grads through the integrator
    grads = {}
    for n,p in model.named_parameters():
        if p.grad is not None:
            grads[f"param::{n}"] = p.grad.detach().cpu().float().numpy()
    if x.grad is not None:
        grads["input_grad"] = x.grad.detach().cpu().float().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **grads)
    return float(loss.item())

# ─────────────────────────────────────────────────────────────────────────────
#  Small image-space perturbations
# ─────────────────────────────────────────────────────────────────────────────
def apply_perturbations(x, noise_std=0.0, brightness=0.0, contrast=0.0, translate_px=0, rng=None):
    """
    x: Tensor [B,3,H,W], already normalized.
    """
    B, C, H, W = x.shape

    # 1) Gaussian noise (optionally deterministic via rng)
    if noise_std and noise_std > 0:
        noise = (torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng)
                 if rng is not None else torch.randn_like(x))
        x = x + noise_std * noise

    # 2) Brightness / contrast in *pixel space* (denorm → adjust → renorm)
    if (brightness and brightness != 0) or (contrast and contrast != 0):
        MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device, dtype=x.dtype).view(1,3,1,1)
        STD  = torch.tensor([0.2023, 0.1994, 0.2010], device=x.device, dtype=x.dtype).view(1,3,1,1)
        x_pix = x * STD + MEAN
        xs = []
        for i in range(B):
            xi = x_pix[i]
            if brightness:
                xi = TF.adjust_brightness(xi, 1.0 + brightness)
            if contrast:
                xi = TF.adjust_contrast(xi, 1.0 + contrast)
            xs.append(xi)
        x = torch.stack(xs, dim=0)
        x = (x - MEAN) / STD

    # 3) Tiny translation (integer pixels; zero padding)
    if translate_px and translate_px != 0:
        tx = (2.0 * translate_px) / max(W - 1, 1)
        ty = (2.0 * translate_px) / max(H - 1, 1)
        theta = torch.tensor([[1, 0, tx],
                              [0, 1, ty]], dtype=x.dtype, device=x.device)
        theta = theta.unsqueeze(0).repeat(B, 1, 1)
        grid = Fnn.affine_grid(theta, size=x.size(), align_corners=False)
        x = Fnn.grid_sample(x, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)

    return x.clamp_(-10, 10)

# ─────────────────────────────────────────────────────────────────────────────
#  Validate (clean or perturbed)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, perturb=False, args=None):
    model.eval(); total=corr=0
    gen = None
    if perturb and args is not None and getattr(args, 'val_seed', None) is not None:
        gen = torch.Generator(device=device).manual_seed(args.val_seed)
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if perturb and args is not None:
            x = apply_perturbations(
                x,
                noise_std=args.val_noise_std,
                brightness=args.val_brightness,
                contrast=args.val_contrast,
                translate_px=args.val_translate_px,
                rng=gen,
            )
        p = model(x).argmax(1)
        corr += (p==y).sum().item(); total += y.size(0)
    return 100.0*corr/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────────────────────────
#  Train / Eval
# ─────────────────────────────────────────────────────────────────────────────
def train_loop(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    train_ds = datasets.CIFAR10(args.data, train=True,  transform=tfm_train, download=True)
    test_ds  = datasets.CIFAR10(args.data, train=False, transform=tfm_test,  download=True)
    tr_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    te_ld = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = TinyNODE(solver=args.solver, width=args.width,
                     use_groupnorm=args.groupnorm,
                     approx_act=args.approx_act,
                     h=args.h, steps=args.steps).to(device)
    print(f"Trainable parameters: {count_parameters(model):,d}")

    # Optim / sched
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    def lr_lambda(e):
        if args.warmup==0 or e >= args.warmup:
            t = (e-args.warmup)/max(1, args.epochs-args.warmup)
            return 0.5*(1+math.cos(math.pi*t))
        return (e+1)/args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossF = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    dumped = False
    start = time.time()
    for ep in range(args.epochs):
        ep0 = time.time()
        model.train()
        for bi, (x,y) in enumerate(tr_ld):
            x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)

            # ─ training perturbations (simple, stochastic) ─
            if args.train_perturb and torch.rand((), device=x.device) < args.train_perturb_p:
                x = apply_perturbations(
                    x,
                    noise_std=args.val_noise_std,
                    brightness=args.val_brightness,
                    contrast=args.val_contrast,
                    translate_px=args.val_translate_px,
                    rng=None,  # stochastic per batch
                )

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = lossF(logits, y)
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()

            # optional: dump exact grads once on a chosen batch
            if args.dump_grads and (not dumped) and (bi == args.dump_batch_index):
                xb = x[:min(32, x.size(0))].detach()
                yb = y[:min(32, y.size(0))].detach()
                path = f"grad_dump/{args.solver}_esrk_dump_ep{ep}_b{bi}.npz"
                l = dump_exact_grads(model, xb, yb, path)
                print(f"[grad-dump] saved {path} (loss={l:.4f})")
                dumped = True

        # Eval + spectral diagnostic
        train_acc = validate(model, tr_ld, device)
        val_clean = validate(model, te_ld, device)
        val_pert  = validate(model, te_ld, device, perturb=True, args=args) if args.enable_val_perturb else None

        with torch.no_grad():
            xb,_ = next(iter(te_ld))
            xb = xb[:args.spec_batch].to(device)
            sigma = estimate_spectral_norm(model, xb, iters=args.spec_iters)

        if val_pert is None:
            print(f"Epoch {ep:02d}  train={train_acc:.2f}%  val={val_clean:.2f}%  "
                  f"σ_est≈{sigma:.3f}  ΔT=steps*h={args.steps*args.h:g}  time={time.time()-ep0:.1f}s")
        else:
            r = (val_pert/val_clean) if val_clean > 0 else float('nan')
            print(f"Epoch {ep:02d}  train={train_acc:.2f}%  val_clean={val_clean:.2f}%  "
                  f"val_pert={val_pert:.2f}%  r={r:.2f}  σ_est≈{sigma:.3f}  "
                  f"ΔT=steps*h={args.steps*args.h:g}  time={time.time()-ep0:.1f}s")
        sched.step()

    dur = (time.time()-start)/60
    print(f"Total training: {dur:.1f} min")
    torch.save(model.state_dict(), f"tiny_{args.solver}.pth")
    print("Saved checkpoint:", f"tiny_{args.solver}.pth")

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser("ESRK-15 vs RK4 on CIFAR-10 with exact-grad dump + perturbed training/validation")
    p.add_argument('--solver', choices=['euler','rk4','esrk'], default='rk4')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch',  type=int, default=128)
    p.add_argument('--width',  type=int, default=64)
    p.add_argument('--h',      type=float, default=7.5)
    p.add_argument('--steps',  type=int,   default=4)
    p.add_argument('--groupnorm', action='store_true')
    p.add_argument('--approx_act', action='store_true')
    p.add_argument('--data',   type=str, default='./cifar_data')
    p.add_argument('--lr',     type=float, default=3e-4)
    p.add_argument('--wd',     type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--clip_grad', type=float, default=None)
    # spectral diag
    p.add_argument('--spec_iters', type=int, default=5)
    p.add_argument('--spec_batch', type=int, default=1)
    # gradient dump
    p.add_argument('--dump_grads', action='store_true')
    p.add_argument('--dump_batch_index', type=int, default=0)
    # validation perturbations
    p.add_argument('--enable_val_perturb', action='store_true',
                   help='Also report accuracy under small image-space perturbations at validation.')
    p.add_argument('--val_noise_std', type=float, default=0.0,
                   help='Gaussian noise std in normalized space (e.g., 0.01).')
    p.add_argument('--val_brightness', type=float, default=0.0,
                   help='Brightness factor delta (e.g., 0.01 => +1%).')
    p.add_argument('--val_contrast', type=float, default=0.0,
                   help='Contrast factor delta (e.g., 0.01 => +1%).')
    p.add_argument('--val_translate_px', type=int, default=0,
                   help='Integer pixel translation (can be negative).')
    p.add_argument('--val_seed', type=int, default=1234,
                   help='Fixed RNG seed for perturbed validation noise (for comparable r across epochs).')
    # training perturbations
    p.add_argument('--train_perturb', action='store_true',
                   help='Apply small perturbations to a fraction of training batches.')
    p.add_argument('--train_perturb_p', type=float, default=0.25,
                   help='Probability that a batch is perturbed during training.')
    return p.parse_args()

if __name__=='__main__':
    args = parse()
    train_loop(args)
