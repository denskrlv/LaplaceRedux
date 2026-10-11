"""Generates assets/banner.png and assets/social-preview.png.

Usage: python make_banner.py <output-prefix>   (writes <prefix>.png at 1280x640 and <prefix>@2x.png)
Requires matplotlib and numpy; the fonts (Avenir Next, Helvetica Neue, Menlo) ship with macOS.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path

out = sys.argv[1]

BG_TOP, BG_BOT = "#0B1022", "#151C3A"
VIOLET, CYAN, PINK = "#8B5CF6", "#22D3EE", "#F472B6"
TEXT, MUTED = "#F5F7FF", "#AEB6D4"
grad = LinearSegmentedColormap.from_list("g", [VIOLET, CYAN])

plt.rcParams["mathtext.fontset"] = "stix"
fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(BG_TOP)

# --- background: vertical gradient + soft radial glow + faint grid ---
bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off()
yy, xx = np.mgrid[0:1:320j, 0:2:640j]
base = np.zeros(yy.shape + (3,))
c0 = np.array(matplotlib.colors.to_rgb(BG_TOP)); c1 = np.array(matplotlib.colors.to_rgb(BG_BOT))
base[:] = c0 + (c1 - c0) * (1 - yy)[..., None]
glow = np.exp(-(((xx - 1.45) / 0.55) ** 2 + ((yy - 0.42) / 0.45) ** 2))
base += 0.10 * glow[..., None] * np.array(matplotlib.colors.to_rgb(VIOLET))
bg.imshow(np.clip(base, 0, 1), extent=[0, 1, 0, 1], aspect="auto", origin="lower")
for gx in np.linspace(0, 1, 33):
    bg.axvline(gx, color="white", alpha=0.025, lw=0.8)
for gy in np.linspace(0, 1, 17):
    bg.axhline(gy, color="white", alpha=0.025, lw=0.8)
bg.set_xlim(0, 1); bg.set_ylim(0, 1)

# --- posterior and its Laplace approximation ---
ax = fig.add_axes([0.575, 0.12, 0.40, 0.76]); ax.set_axis_off()
x = np.linspace(-3.6, 5.2, 1600)
def logp(t):
    return np.log(0.68 * np.exp(-0.5 * (t / 0.85) ** 2) / 0.85
                  + 0.32 * np.exp(-0.5 * ((t - 2.1) / 0.75) ** 2) / 0.75)
p = np.exp(logp(x))
i = np.argmax(p); t_map = x[i]
h = 1e-3
curv = -(logp(t_map + h) - 2 * logp(t_map) + logp(t_map - h)) / h**2  # Hessian of -log p
q = p[i] * np.exp(-0.5 * curv * (x - t_map) ** 2)                        # Laplace approx.
p, q = p / p[i], q / p[i]

# true posterior: muted fill + outline
ax.fill_between(x, 0, p, color="#7C86AD", alpha=0.16, lw=0)
ax.plot(x, p, color="#9AA3C7", lw=2.2, alpha=0.75)

# Laplace Gaussian: gradient fill clipped to the curve
verts = np.column_stack([np.r_[x, x[::-1]], np.r_[q, np.zeros_like(q)]])
clip = PathPatch(Path(verts), facecolor="none", edgecolor="none")
ax.add_patch(clip)
gy, gx = np.mgrid[0:1:200j, 0:1:400j]
fillimg = grad(gx)
fillimg[..., 3] = 0.08 + 0.32 * gy
im = ax.imshow(fillimg, extent=[x[0], x[-1], 0, 1.0], aspect="auto", origin="lower", zorder=2)
im.set_clip_path(clip)

# Laplace Gaussian: glowing gradient line
pts = np.column_stack([x, q]).reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
qm = 0.5 * (q[:-1] + q[1:])
taper = np.clip(qm / 0.12, 0, 1) ** 0.8                      # fade the line out in the tails
cols = grad(np.linspace(0, 1, len(segs)))
for lw, a in [(16, 0.05), (10, 0.08), (6, 0.15), (3.2, 1.0)]:
    c = cols.copy(); c[:, 3] = a * taper
    ax.add_collection(LineCollection(segs, colors=c, linewidths=0.4 + lw * taper, zorder=3,
                                     capstyle="round"))

# MAP marker
ax.plot([t_map, t_map], [0, 1.0], color=TEXT, lw=1.2, ls=(0, (3, 4)), alpha=0.55, zorder=4)
for s, a in [(520, 0.08), (260, 0.18), (70, 1.0)]:
    ax.scatter([t_map], [1.0], s=s, color=CYAN if a < 1 else TEXT, alpha=a, zorder=5, lw=0)

# posterior samples as a rug
rng = np.random.default_rng(7)
samples = rng.normal(t_map, 1 / np.sqrt(curv), 70)
ax.scatter(samples, np.full_like(samples, -0.045), s=10, c=grad((samples - x[0]) / (x[-1] - x[0])),
           alpha=0.8, lw=0, zorder=4)
ax.axhline(0, color="white", alpha=0.18, lw=1)

# labels
ax.text(t_map, 1.09, r"$\theta_{\mathrm{MAP}}$", color=TEXT, fontsize=19, ha="center", va="bottom")
ax.text(3.0, 0.50, r"$p(\theta\,|\,\mathcal{D})$", color="#AEB6D4", fontsize=19, ha="left")
ax.text(1.05, 0.86, r"$\mathcal{N}(\theta_{\mathrm{MAP}},\,\mathbf{H}^{-1})$", color=CYAN,
        fontsize=19, ha="left")
ax.set_xlim(x[0], x[-1]); ax.set_ylim(-0.1, 1.25)

# --- text block ---
fig.text(0.065, 0.735, "B A Y E S I A N   D E E P   L E A R N I N G", family="Menlo",
         fontsize=12.5, color=CYAN, alpha=0.9)
t1 = fig.text(0.062, 0.555, "Laplace", family="Avenir Next", fontsize=60, color=TEXT)
fig.canvas.draw()
bb = t1.get_window_extent().transformed(fig.transFigure.inverted())
fig.text(bb.x1 + 0.002, 0.555, "Redux", family="Avenir Next", fontsize=60, color="#A78BFA")
fig.text(0.065, 0.385, "Effortless, post-hoc uncertainty\nfor deep neural networks",
         family="Helvetica Neue", fontsize=21, color=MUTED, linespacing=1.45)

cx = 0.065
for label in ["PyTorch", "Laplace Approximation", "Subspace LA", "SWAG-LA"]:
    t = fig.text(cx, 0.19, label, family="Menlo", fontsize=11.5, color=TEXT,
                 bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.9", fc="#FFFFFF10",
                           ec="#FFFFFF33", lw=1))
    fig.canvas.draw()
    cx = t.get_bbox_patch().get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.014

for path, dpi in [(out + ".png", 100), (out + "@2x.png", 200)]:
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
print("t_map", round(t_map, 3), "sigma", round(1 / np.sqrt(curv), 3))
