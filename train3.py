#Extended to tinyimage net 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny-ImageNet (or CIFAR-10) training for TinyNODE with ESRK / RK4 / Euler / DoPri5.
- Auto-downloads Tiny-ImageNet-200 and fixes the validation split into class folders.
- Adds AMP support, GroupNorm option, label smoothing, cosine LR with warmup.
- Keeps your ESRK-15 coefficients and exact integrator implementation.

Example:
  python train_node_esrk.py --dataset tinyimagenet --solver esrk --h 30 --steps 1 \
    --epochs 150 --batch 128 --width 64 --groupnorm --label_smoothing 0.05 --amp
  python train_node_esrk.py --dataset tinyimagenet --solver rk4  --h 7.5 --steps 4 \
    --epochs 150 --batch 128 --width 64 --groupnorm --label_smoothing 0.05 --amp
"""
import argparse, os, time, math, shutil, zipfile, tarfile, io, sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

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
A_tbl = torch.tensor(a_np, dtype=torch.float32)
b_tbl = torch.tensor(b_np, dtype=torch.float32)
c_tbl = A_tbl.tril(-1).sum(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Utilities: seed, downloader for Tiny-ImageNet-200
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _download(url: str, dest: Path):
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)

def prepare_tiny_imagenet(root: Path):
    """
    Downloads Tiny-ImageNet-200 to: <root>/tiny-imagenet-200
    and restructures validation set into class-subfolders.
    """
    root = Path(root)
    out_dir = root / "tiny-imagenet-200"
    if out_dir.exists() and (out_dir / "train").exists() and (out_dir / "val").exists():
        # try to ensure val has class folders
        val_dir = out_dir / "val"
        images_dir = val_dir / "images"
        anno_file = val_dir / "val_annotations.txt"
        if images_dir.exists() and anno_file.exists():
            # needs restructuring
            _restructure_val(val_dir)
        print(f"[tiny-imagenet] found at {out_dir}")
        return str(out_dir)

    # Try official mirror(s)
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "https://raw.githubusercontent.com/finxter/tiny-imagenet/master/tiny-imagenet-200.zip",  # backup mirror
    ]
    zip_path = root / "tiny-imagenet-200.zip"
    for u in urls:
        try:
            _download(u, zip_path)
            break
        except Exception as e:
            print(f"[warn] download failed from {u}: {e}")
    if not zip_path.exists():
        raise RuntimeError("Could not download tiny-imagenet-200.zip from known mirrors.")

    print("[tiny-imagenet] extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(root)

    # Fix validation structure (move images into per-class subfolders)
    _restructure_val(out_dir / "val")
    print(f"[tiny-imagenet] ready at {out_dir}")
    return str(out_dir)

def _restructure_val(val_dir: Path):
    """
    Tiny-ImageNet validation originally has all images in val/images and
    a file val_annotations.txt mapping <img> <class> ...
    We create val/<class> folders and move images accordingly.
    """
    images_dir = val_dir / "images"
    anno_file = val_dir / "val_annotations.txt"
    if not images_dir.exists() or not anno_file.exists():
        return
    print("[tiny-imagenet] restructuring val split ...")
    with open(anno_file, 'r') as f:
        lines = f.read().strip().splitlines()
    mapping = []
    for ln in lines:
        parts = ln.split('\t')
        if len(parts) >= 2:
            mapping.append((parts[0], parts[1]))
    for img, cls in mapping:
        cls_dir = val_dir / cls
        cls_dir.mkdir(exist_ok=True)
        src = images_dir / img
        dst = cls_dir / img
        if src.exists():
            shutil.move(str(src), str(dst))
    # cleanup: remove images dir and annotations
    try:
        shutil.rmtree(images_dir)
    except Exception:
        pass
    # keep annotations for provenance

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
#  Tiny NODE backbone (now flexible num_classes & img_size)
# ─────────────────────────────────────────────────────────────────────────────
class TinyNODE(nn.Module):
    def __init__(self, solver='esrk', width=64, use_groupnorm=False, approx_act=False,
                 h=1.0, steps=1, num_classes=10, img_size=32):
        super().__init__()
        act = ApproxSiLU() if approx_act else nn.SiLU()

        # Downsampling: 64x64 -> 32 -> 16 ; 32x32 -> 16 only
        enc = [nn.Conv2d(3, width, 3, padding=1, bias=False), act]
        if img_size >= 64:
            enc += [nn.MaxPool2d(2)]   # 64 -> 32
        enc += [nn.MaxPool2d(2)]       # 32 -> 16
        self.encoder = nn.Sequential(*enc)

        f = make_f(width, use_groupnorm, approx_act)
        if solver == 'euler':
            self.ode = EulerBlock(f, h=h, steps=steps)
        elif solver == 'rk4':
            self.ode = RK4Block(f, h=h, steps=steps)
        elif solver == 'esrk':
            self.ode = ESRKBlock(f, A_tbl, b_tbl, c_tbl, h=h, steps=steps)
        elif solver == 'dopri5':
            self.ode = Dopri5Block(f, h=h, steps=steps)
        else:
            raise ValueError(f"Unknown solver: {solver!r}")

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, num_classes, bias=False)
        )

    def forward(self, x):
        return self.head(self.ode(self.encoder(x)))

# ─────────────────────────────────────────────────────────────────────────────
#  Eval / train helpers
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

# ─────────────────────────────────────────────────────────────────────────────
#  Data: CIFAR-10 or Tiny-ImageNet
# ─────────────────────────────────────────────────────────────────────────────
def build_loaders(args):
    if args.dataset == 'cifar10':
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
        num_classes, img_size = 10, 32

    elif args.dataset == 'tinyimagenet':
        root = prepare_tiny_imagenet(Path(args.data))
        train_dir = os.path.join(root, 'train')
        val_dir   = os.path.join(root, 'val')

        # Strong but simple aug for 64x64
        tfm_train = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        if args.autoaugment:
            tfm_train = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                AutoAugment(AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
            ])

        tfm_test = transforms.Compose([
            transforms.Resize(72),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        train_ds = ImageFolder(train_dir, transform=tfm_train)
        test_ds  = ImageFolder(val_dir,   transform=tfm_test)
        num_classes, img_size = 200, 64

    else:
        raise ValueError("args.dataset must be 'cifar10' or 'tinyimagenet'")

    tr_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, persistent_workers=True)
    te_ld = DataLoader(test_ds,  batch_size=max(128, args.batch),
                       shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    return tr_ld, te_ld, num_classes, img_size

# ─────────────────────────────────────────────────────────────────────────────
#  Train Loop (with AMP, cosine LR w/ warmup)
# ─────────────────────────────────────────────────────────────────────────────
def train_loop(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tr_ld, te_ld, num_classes, img_size = build_loaders(args)

    model = TinyNODE(
        solver=args.solver, width=args.width,
        use_groupnorm=args.groupnorm, approx_act=args.approx_act,
        h=args.h, steps=args.steps,
        num_classes=num_classes, img_size=img_size
    ).to(device)

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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    def lr_lambda(e):
        if args.warmup==0 or e >= args.warmup:
            t = (e-args.warmup)/max(1, args.epochs-args.warmup)
            return 0.5*(1+math.cos(math.pi*t))
        return (e+1)/max(1,args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossF = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = GradScaler(enabled=args.amp and device=='cuda')

    start = time.time()
    for ep in range(args.epochs):
        ep0 = time.time()
        model.train()
        running = 0.0
        for x,y in tr_ld:
            x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device=='cuda'):
                logits = model(x)
                loss = lossF(logits, y)
            scaler.scale(loss).backward()
            if args.clip_grad is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * x.size(0)

        # Eval
        train_acc = validate(model, tr_ld, device)
        val_acc   = validate(model, te_ld, device)
        print(f"Epoch {ep:03d}  train={train_acc:.2f}%  val={val_acc:.2f}%  "
              f"ΔT=steps*h={args.steps*args.h:g}  time/ep={time.time()-ep0:.1f}s")
        sched.step()

    dur = (time.time()-start)/60
    print(f"Total training: {dur:.1f} min")
    ckpt = f"tiny_{args.dataset}_{args.solver}_dt{args.steps}x{args.h}.pth"
    torch.save(model.state_dict(), ckpt)
    print("Saved checkpoint:", ckpt)

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser("TinyNODE ESRK/RK4/Euler/DoPri5 on CIFAR-10 or Tiny-ImageNet")
    p.add_argument('--dataset', choices=['cifar10','tinyimagenet'], default='tinyimagenet')
    p.add_argument('--data',   type=str, default='./data')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed',   type=int, default=1337)

    p.add_argument('--solver', choices=['euler','rk4','esrk','dopri5'], default='esrk')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch',  type=int, default=128)
    p.add_argument('--width',  type=int, default=64)
    p.add_argument('--h',      type=float, default=30.0)   # ESRK single step default
    p.add_argument('--steps',  type=int,   default=1)

    p.add_argument('--groupnorm', action='store_true')
    p.add_argument('--approx_act', action='store_true')
    p.add_argument('--lr',     type=float, default=3e-4)
    p.add_argument('--wd',     type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--clip_grad', type=float, default=None)

    # DoPri5 tuning
    p.add_argument('--rtol', type=float, default=1e-3)
    p.add_argument('--atol', type=float, default=1e-6)
    p.add_argument('--safety', type=float, default=0.9)
    p.add_argument('--h_min', type=float, default=1e-3)
    p.add_argument('--h_max', type=float, default=0.0, help='0 => no explicit max')
    p.add_argument('--max_nsteps', type=int, default=10000)

    # training extras
    p.add_argument('--amp', action='store_true')
    p.add_argument('--autoaugment', action='store_true')
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    args = parse()

    # sensible parity defaults:
    # for ESRK (ΔT=30 with 1 step):
    #   --solver esrk --h 30 --steps 1
    # for RK4 (ΔT=30 with 4 steps):
    #   --solver rk4  --h 7.5 --steps 4
    if args.solver == 'rk4' and (args.h==30.0 and args.steps==1):
        # if the user kept defaults that match ESRK, nudge to RK4 parity
        args.h, args.steps = 7.5, 4

    train_loop(args)
