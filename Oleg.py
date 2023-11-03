import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import time

# QR vs PLU vs SVD
n = 100
m = 70
cnt = 100
Id = np.eye(m)
Mat = []
x = []
b = []
cond = []

for i in range(0, cnt):
    # S = [(0.2 + 1/i) for i in range(1, m+1)]  # np.diag(2 + np.random.randn(m))
    # U = np.linalg.qr(np.random.randn(n, m), mode='raw')[0].T
    # Vh = np.linalg.qr(np.random.randn(m, m), mode='raw')[0]
    # Mat.append(U @ np.diag(S) @ Vh)

    Mat.append(np.random.randn(n, m))
    x.append(np.random.randn(m))
    b.append(Mat[i] @ x[i])
    cond.append(np.linalg.cond(Mat[i]))

# PLU
plu_pinv = []
plu_time = []  # need only for decomposition
for i in range(0, cnt):
    start = time.time()
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
    qr_pinv.append(invR @ Q.T)
    end = time.time()
    qr_time.append(end - start)

# SVD
svd_pinv = []
svd_time = []
for i in range(0, cnt):
    start = time.time()
    U, S, Vh = np.linalg.svd(Mat[i], full_matrices=False)
    svd_pinv.append(Vh.T @ np.diag(1 / S) @ U.T)
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
BIG_FS = 20
NORMAL_FS = 10
SMALL_FS = 8
BINS = 28
names = ['PLU', 'QR', 'SVD']
clrs = ['tab:blue', 'tab:orange', 'tab:brown']
hists = ['a', 'b', 'c']
bars = [('d', 'mean time [ms]'), ('e', 'mean norm')]
mean_time = [np.mean(plu_time), np.mean(qr_time), np.mean(svd_time)]
mean_norm = [np.mean(plu_norm), np.mean(qr_norm), np.mean(svd_norm)]
norms = [plu_norm, qr_norm, svd_norm]

mosaic = """
aabbcc
aabbcc
dddeee
dddeee
ffffff
ffffff
ffffff
"""

fig, ax = plt.subplot_mosaic(mosaic)

for i, name in enumerate(names):
    ax[hists[i]].hist(plu_norm, bins=BINS, histtype='bar', color=clrs[i], edgecolor='black')
    ax[hists[i]].set_xlabel('')
    ax[hists[i]].set_ylabel('')
    ax[hists[i]].set_title(names[i] + ' norm hist', fontsize=NORMAL_FS)
    ax[hists[i]].locator_params(axis='both', nbins=6)
    ax[hists[i]].tick_params(axis='both', labelsize=SMALL_FS)
    ax['d'].bar(name, mean_time[i], color=clrs[i])
    ax['e'].bar(name, mean_norm[i], color=clrs[i])
    ax['f'].plot(cond, norms[i], '.', markersize=6, color=clrs[i])

for bar, label in bars:
    ax[bar].set_ylabel(label, fontsize=SMALL_FS)
    ax[bar].grid(axis='y')
    ax[bar].tick_params(axis='y', labelsize=SMALL_FS)

ax['f'].set_xlabel('condition number', fontsize=SMALL_FS)
ax['f'].set_ylabel('norm', fontsize=SMALL_FS)
ax['f'].legend(names, loc=0)
ax['f'].tick_params(axis='both', labelsize=SMALL_FS)
ax['f'].locator_params(axis='x', nbins=16)
ax['f'].grid(axis='both')

plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
plt.subplots_adjust(wspace=0.6, hspace=0.9)
plt.show()
