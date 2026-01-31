import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 1) Naredimo "sliko" A (nek vzorec), in B = A + 10 (povečana svetlost)
H, W = 128, 128
y, x = np.mgrid[0:H, 0:W]
A = (np.sin(2*np.pi*x/16) + np.cos(2*np.pi*y/24))  # neka tekstura
A = (A - A.min()) / (A.max() - A.min()) * 255      # v [0,255]
B = A + 10                                         # svetlost +10

# 2) Zgradimo 2 linearni mešanici X = M * [A; B]
#    (v praksi so to "opazovanja"/mešani signali)
Mmix = np.array([[1.0, 0.7],
                 [0.6, 1.0]])

S = np.vstack([A.ravel(), B.ravel()])              # 2 x (H*W)
X = Mmix @ S                                       # 2 x (H*W)

# 3) ICA poskuša iz X dobiti "neodvisne komponente"
ica = FastICA(n_components=2, whiten='unit-variance', random_state=0, max_iter=2000)
S_hat = ica.fit_transform(X.T).T                   # 2 x (H*W), transponiramo zaradi sklearn oblike
A_hat = ica.mixing_                                # ocenjena mešalna matrika (v whitened smislu)

# 4) Preoblikujemo nazaj v slike
IC1 = S_hat[0].reshape(H, W)
IC2 = S_hat[1].reshape(H, W)

# Pomožna funkcija za lep prikaz (normalizacija)
def norm_img(img):
    img = img.astype(float)
    img -= img.min()
    img /= (img.max() - img.min() + 1e-12)
    return img

# 5) Prikaz
fig, ax = plt.subplots(2, 3, figsize=(10, 6))
ax[0,0].imshow(A, cmap='gray'); ax[0,0].set_title("A (original)"); ax[0,0].axis("off")
ax[0,1].imshow(B, cmap='gray'); ax[0,1].set_title("B = A + 10"); ax[0,1].axis("off")
ax[0,2].imshow(norm_img(X[0].reshape(H,W)), cmap='gray'); ax[0,2].set_title("Mešanica X1"); ax[0,2].axis("off")

ax[1,0].imshow(norm_img(X[1].reshape(H,W)), cmap='gray'); ax[1,0].set_title("Mešanica X2"); ax[1,0].axis("off")
ax[1,1].imshow(norm_img(IC1), cmap='gray'); ax[1,1].set_title("ICA komponenta 1"); ax[1,1].axis("off")
ax[1,2].imshow(norm_img(IC2), cmap='gray'); ax[1,2].set_title("ICA komponenta 2"); ax[1,2].axis("off")

plt.tight_layout()
plt.show()

# 6) Še numerično: korelacije z A in B (da vidimo, ali ICA res loči)
def corr(a, b):
    a = a.ravel().astype(float); b = b.ravel().astype(float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a*b))

print("corr(IC1, A) =", corr(IC1, A), "  corr(IC1, B) =", corr(IC1, B))
print("corr(IC2, A) =", corr(IC2, A), "  corr(IC2, B) =", corr(IC2, B))
