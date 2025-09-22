#!/usr/bin/env python3
import argparse, time, os, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
#  Integrator blocks: Euler / RK4 / ESRK-15 (general explicit RK) / DoPri5
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

class Dopri5Block(nn.Module):
    """
    Adaptive Dormand–Prince 5(4) RK with embedded error control.
    Integrates from t0 to t0 + steps*h0 (ΔT matches fixed-step runs).
    """
    def __init__(self, f, h=1.0, steps=1,
                 rtol=1e-3, atol=1e-6, safety=0.9,
                 h_min=1e-3, h_max=None, max_nsteps=10_000):
        super().__init__()
        self.f = f
        self.h0 = float(h)
        self.steps = int(steps)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.safety = float(safety)
        self.h_min = float(h_min)
        self.h_max = float('inf') if h_max is None else float(h_max)
        self.max_nsteps = int(max_nsteps)

        # Butcher tableau for DOPRI5
        c = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
        A = [
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168,  -355/33,    46732/5247,   49/176,   -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ]
        b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]

        self.register_buffer("c", torch.tensor(c, dtype=torch.float32))
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32).tril(-1))
        self.register_buffer("b5", torch.tensor(b5, dtype=torch.float32))
        self.register_buffer("b4", torch.tensor(b4, dtype=torch.float32))

    def _error_norm(self, y5, y4):
        err = y5 - y4
        scale = self.atol + self.rtol * torch.maximum(y5.abs(), y4.abs())
        e = ((err/scale)**2).mean().sqrt()
        return e

    @torch.no_grad()
    def _choose_h(self, h, err, order=5):
        if err <= 1e-14:
            factor = 5.0
        else:
            factor = (1.0/err)**(1.0/order)
        factor = self.safety * float(factor)
        factor = max(0.2, min(5.0, factor))
        return h * factor

    def _one_adaptive_step(self, t, y, h):
        k = []
        for i in range(7):
            yi = y
            if i > 0:
                acc = 0.0
                for j in range(i):
                    aij = self.A[i, j]
                    if float(aij) != 0.0:
                        acc = acc + aij * k[j]
                yi = y + h * acc
            ti = t + self.c[i] * h
            k.append(self.f(ti, yi))
        y5 = y + h * sum(self.b5[i]*k[i] for i in range(7) if float(self.b5[i]) != 0.0)
        y4 = y + h * sum(self.b4[i]*k[i] for i in range(7) if float(self.b4[i]) != 0.0)
        err = self._error_norm(y5, y4)
        return y5, err

    def forward(self, x, t0=0.0):
        T = self.steps * self.h0
        t_end = t0 + T
        t = torch.as_tensor(t0, dtype=torch.float32, device=x.device)
        h = min(self.h0, T)
        h = max(self.h_min, min(self.h_max, h))
        y = x
        nsteps = 0

        while float(t) < float(t_end):
            if nsteps >= self.max_nsteps:
                break
            h = min(h, float(t_end) - float(t))
            y_prop, err = self._one_adaptive_step(t, y, h)
            if float(err) <= 1.0 or h <= self.h_min*1.0001:
                y = y_prop
                t = t + h
                h = self._choose_h(h, float(err) + 1e-12, order=5)
                h = max(self.h_min, min(self.h_max, h))
            else:
                h = self._choose_h(h, float(err) + 1e-12, order=5)
                h = max(self.h_min, min(self.h_max, h))
            nsteps += 1
        return y

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
        elif solver == 'dopri5':
            # placeholder; hyperparams set after construction in train_loop
            self.ode = Dopri5Block(f, h=h, steps=steps)
        else:
            raise ValueError(f"Unknown solver: {solver!r}")
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 100, bias=False)#Changed from cifar10 from 10 -> 100 
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
#  Train / Eval
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device):
    model.eval(); total=corr=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        corr += (p==y).sum().item(); total += y.size(0)
    return 100.0*corr/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    train_ds = datasets.CIFAR100(args.data, train=True,  transform=tfm_train, download=True)
    test_ds  = datasets.CIFAR100(args.data, train=False, transform=tfm_test,  download=True)
    tr_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    te_ld = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = TinyNODE(solver=args.solver, width=args.width,
                     use_groupnorm=args.groupnorm,
                     approx_act=args.approx_act,
                     h=args.h, steps=args.steps).to(device)

    # If using dopri5, set its hyperparameters from CLI
    if args.solver == 'dopri5':
        ode = model.ode
        ode.rtol = args.rtol
        ode.atol = args.atol
        ode.safety = args.safety
        ode.h_min = args.h_min
        ode.h_max = (float('inf') if args.h_max == 0.0 else args.h_max)
        ode.max_nsteps = args.max_nsteps

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
        val_acc   = validate(model, te_ld, device)
        with torch.no_grad():
            xb,_ = next(iter(te_ld))
            xb = xb[:args.spec_batch].to(device)
            sigma = estimate_spectral_norm(model, xb, iters=args.spec_iters)
        print(f"Epoch {ep:02d}  train={train_acc:.2f}%  val={val_acc:.2f}%  "
              f"σ_est≈{sigma:.3f}  ΔT=steps*h={args.steps*args.h:g}  time={time.time()-ep0:.1f}s")
        sched.step()

    dur = (time.time()-start)/60
    print(f"Total training: {dur:.1f} min")
    torch.save(model.state_dict(), f"tiny_{args.solver}.pth")
    print("Saved checkpoint:", f"tiny_{args.solver}.pth")

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser("ESRK-15 vs RK4 vs DoPri5 on CIFAR-100 with exact-grad dump")
    p.add_argument('--solver', choices=['euler','rk4','esrk','dopri5'], default='rk4')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch',  type=int, default=128)
    p.add_argument('--width',  type=int, default=64)
    p.add_argument('--h',      type=float, default=7.5)
    p.add_argument('--steps',  type=int,   default=4)
    p.add_argument('--groupnorm', action='store_true')
    p.add_argument('--approx_act', action='store_true')
    p.add_argument('--data',   type=str, default='./cifar_dat')
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
    # DoPri5 tuning
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--atol', type=float, default=1e-6)
    p.add_argument('--safety', type=float, default=0.9)
    p.add_argument('--h_min', type=float, default=1e-3)
    p.add_argument('--h_max', type=float, default=0.0, help='0 => no explicit max')
    p.add_argument('--max_nsteps', type=int, default=10000)
    return p.parse_args()

if __name__=='__main__':
    args = parse()
    train_loop(args)
