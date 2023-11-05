import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

# QR vs PLU vs SVD

# settings
n = 200
m = 150
S_mult = 10**(8)  # > 1
X_mult = 10**(-4)  # > 0
IS_NORMAL = False
IS_COMPLEX = True
# const
cnt = 400
ALPHA_POS = 0.5
ALPHA_ZERO = 0.1

# containers
Id = np.eye(m)
Mat = []
x = []
b = []
cond = []
s_min = []
s_max = []
s_mean = []
randf = None
if IS_NORMAL:
    randf = np.random.randn
else:
    randf = np.random.rand
prob = [1 - ALPHA_POS - ALPHA_ZERO, ALPHA_POS, ALPHA_ZERO]

# mat generator
for i in range(0, cnt):
    mask1 = np.random.choice([-1, 1, 0], size=n * m, p=prob)
    mask1 = mask1.reshape((n, m))
    mask2 = np.random.choice([-1, 1, 0], size=n * m, p=prob)
    mask2 = mask2.reshape((n, m))
    maskx = np.random.choice([-1, 1, 0], size=m, p=prob)
    S_ = randf(m)
    med = np.median(S_)
    S_[S_ > med] *= S_mult
    S_[S_ < med] /= S_mult
    if IS_COMPLEX:
        U, S, Vh = np.linalg.svd(mask1 * randf(n, m) + 1.j * mask2 * randf(n, m), full_matrices=False)
        Mat.append(U @ np.diag(S_) @ Vh)
        x.append(maskx * X_mult * (randf(m) + 1.j * randf(m)))
    else:
        U, S, Vh = np.linalg.svd(mask1 * randf(n, m), full_matrices=False)
        Mat.append(U @ np.diag(S_) @ Vh)
        x.append(maskx * X_mult * randf(m))
    cond.append(np.linalg.cond(Mat[i]))
    b.append(Mat[i] @ x[i])
    s_min.append(np.min(S_))
    s_max.append(np.max(S_))
    s_mean.append(np.mean(S_))

# PLU
plu_pinv = []
plu_time = []  # need only for decomposition
for i in range(0, cnt):
    start = time.time()
    if IS_COMPLEX:
        B = np.linalg.solve(Mat[i].conjugate().T @ Mat[i], Id)
        plu_pinv.append(B @ Mat[i].conjugate().T)
    else:
        B = np.linalg.solve(Mat[i].T @ Mat[i], Id)
        plu_pinv.append(B @ Mat[i].T)
    end = time.time()
    plu_time.append(end - start)

# QR
qr_pinv = []
qr_time = []
for i in range(0, cnt):
    start = time.time()
    Q, R = np.linalg.qr(Mat[i])
    invR = scipy.linalg.solve_triangular(R, Id)
    if IS_COMPLEX:
        qr_pinv.append(invR @ Q.conjugate().T)
    else:
        qr_pinv.append(invR @ Q.T)
    end = time.time()
    qr_time.append(end - start)

# SVD
svd_pinv = []
svd_time = []
for i in range(0, cnt):
    start = time.time()
    U, S, Vh = np.linalg.svd(Mat[i], full_matrices=False)
    if IS_COMPLEX:
        svd_pinv.append(Vh.conjugate().T @ np.diag(1 / S) @ U.conjugate().T)
    else:
        svd_pinv.append(Vh.T @ np.diag(1/S) @ U.T)
    end = time.time()
    svd_time.append(end - start)

# norm
plu_norm = []
qr_norm = []
svd_norm = []
for i in range(0, cnt):
    plu_x = plu_pinv[i] @ b[i]
    qr_x = qr_pinv[i] @ b[i]
    svd_x = svd_pinv[i] @ b[i]
    norm_xi = np.linalg.norm(x[i])
    plu_norm.append(np.linalg.norm(x[i] - plu_x) / norm_xi)
    qr_norm.append(np.linalg.norm(x[i] - qr_x) / norm_xi)
    svd_norm.append(np.linalg.norm(x[i] - svd_x) / norm_xi)

# report
norms = [plu_norm, qr_norm, svd_norm]
mean_norm = [np.mean(plu_norm), np.mean(qr_norm), np.mean(svd_norm)]
max_norm = [np.max(plu_norm), np.max(qr_norm), np.max(svd_norm)]
mean_time = [np.mean(plu_time)*1000, np.mean(qr_time)*1000, np.mean(svd_time)*1000]
s_min_mean = np.round(np.mean(s_min), 1)
s_max_mean = np.round(np.mean(s_max), 1)
s_mean_mean = np.round(np.mean(s_mean), 1)

names = ['PLU', 'QR', 'SVD']
clrs = ['tab:blue', 'tab:orange', 'tab:brown']
hists = ['a', 'b', 'c']
bars = ['d', 'e', 'f']
bars_ylabel = ['mean time [ms]', 'mean norm', 'max norm']
plot = 'g'
plot_xlabel = 'condition number'
plot_ylabel = 'norm'

NORMAL_FS = 11
SMALL_FS = 9
HIST_BINS = 28
PLOT_BINS = 16
number = 'real'
destr = 'linear'
if IS_COMPLEX: number = 'complex'
if IS_NORMAL: destr = 'normal'
suptitle = 'Matrices (' + str(n) + ' x ' + str(m) + '), ' + str(cnt) + ' count, with ' + number + ' numbers'
suptitle += ', S in [' + str(s_min_mean) + ', ' + str(s_max_mean) + '], mean(S) = ' + str(s_mean_mean)
suptitle += ', destribution: ' + destr

fig, ax = plt.subplot_mosaic("""
    aabbcc
    aabbcc
    ddeeff
    ddeeff
    gggggg
    gggggg
    gggggg
""")

for i, name in enumerate(names):
    ax[hists[i]].hist(norms[i], bins=HIST_BINS, histtype='bar', color=clrs[i], edgecolor='black')
    ax[hists[i]].set_xlabel('')
    ax[hists[i]].set_ylabel('')
    ax[hists[i]].set_title(names[i] + ' norm hist', fontsize=NORMAL_FS)
    ax[hists[i]].locator_params(axis='both', nbins=np.floor(HIST_BINS/4))
    ax[hists[i]].tick_params(axis='both', labelsize=SMALL_FS)
    ax[hists[i]].ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)

    ax[bars[0]].bar(name, mean_time[i], color=clrs[i])
    ax[bars[1]].bar(name, mean_norm[i], color=clrs[i])
    ax[bars[2]].bar(name, max_norm[i], color=clrs[i])
    ax[bars[i]].set_xlabel('')
    ax[bars[i]].set_ylabel(bars_ylabel[i], fontsize=NORMAL_FS)
    ax[bars[i]].set_title('')
    ax[bars[i]].grid(axis='y')
    ax[bars[i]].tick_params(axis='y', labelsize=SMALL_FS)
    ax[bars[i]].tick_params(axis='x', labelsize=NORMAL_FS)

for i, name in enumerate(names):
    ax[plot].plot(cond, norms[i], '.', markersize=6, color=clrs[i])
ax[plot].set_xlabel(plot_xlabel, fontsize=NORMAL_FS)
ax[plot].set_ylabel(plot_ylabel, fontsize=NORMAL_FS)
ax[plot].legend(names, fontsize=NORMAL_FS, framealpha=1.0)
ax[plot].tick_params(axis='both', labelsize=SMALL_FS)
ax[plot].locator_params(axis='x', nbins=PLOT_BINS)
ax[plot].grid(axis='both')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.subplots_adjust(wspace=0.6, hspace=0.9)
fig.suptitle(suptitle, size=NORMAL_FS, fontweight="bold", y=0.94)
fname = str(n) + '_' + str(m) + '_' + str(cnt) + '_' + number[0] + '_' + str(int(np.round(s_mean_mean, 0))) + '_' + destr[0]
print(fname)
plt.show()
