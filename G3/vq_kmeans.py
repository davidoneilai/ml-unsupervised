# Vector Quantization com K-means (pixel vs. blocos 2x2) para 256x256
# Autor: David O'Neil :)  |  Estrutura pensada para apresentação da matéria do Lucas Araujo de AMNS

### Escolha uma imagem colorida qualquer e aplique a estratégia de "vector quantization" com o algoritmo K-means (semelhante ao demonstrado na 
### figura 15.3 do livro https://www.bishopbook.com/). Mostre o gráfico da loss de reconstrução da imagem com K variando de 1 até o valor de máximo de K permitido 
### para garantir uma taxa de compressão acima de 50%. No mesmo gráfico, mostre também a loss de reconstrução ao aplicar a mesma estratégia para clusterizar 
### janelas de 2x2 na imagem (ao invés de cada pixel). Ao comparar os gráficos, justifique qual estratégia de compresssão é melhor.

import argparse
from math import ceil, log2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from datetime import datetime  

# configs
TARGET_H, TARGET_W = 256, 256 # altura e largura da imagem
BITS_PER_CHANNEL = 8 # bits por canal 
CHANNELS = 3 # já que é RGB (3 canais)
BITS_PER_PIXEL = BITS_PER_CHANNEL * CHANNELS  # 24 bits (RGB-8bpc)

# MiniBatchKMeans (mais rápido e estável em 256x256)
# aqui o motivo é basicamente só por questão de performance, por mais que vamos usar apenas uma imagem pequena, o k-means pode demorar
# por mais que a imagem seja pequena, o k-means pode ser computacionalmente caro e eu quero testar com imagens maiores 
KMEANS_KW = dict(
    batch_size=4096,     # podemos ajustar para testes didáticos
    max_iter=60,         # o k-means clássico roda 300 iterações por padrão, mas como vamos usar batch, podemos reduzir já que a convergência é mais rápida
    n_init=8,            # o padrão é 10, vou deixar 8
    random_state=42,
)

# Utilidades
def load_or_make_image(path=None, size=(TARGET_H, TARGET_W)):
    """Carrega imagem RGB e redimensiona"""
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32))**2))

# PSNR (Peak Signal to Noise Ratio) é uma métrica comum para avaliar a qualidade da reconstrução de imagens,
# ela mede quão parecida está a imagem reconstruída em relação à original
# a gente se baseia no MSE, mas transformamos em uma escala logarítmica (dB) para facilitar interpretação
# quanto maior o PSNR, mais parecida a imagem comprimida está da original (menos “ruído” a compressão introduziu)
def psnr(a, b):
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / err)


# Cálculo de K_max (compressão ≥ 50%)
def kmax_for_pixels(N_pixels, min_compression_ratio=0.5):
    """
    Original: N * 24 bits
    Pixel-VQ: N * ceil(log2 K) + K * 24  <=  0.5 * (N*24)
    """
    orig_bits = N_pixels * BITS_PER_PIXEL
    K, K_max = 1, 1
    while True:
        comp_bits = N_pixels * ceil(log2(max(K, 2))) + K * BITS_PER_PIXEL
        if comp_bits <= min_compression_ratio * orig_bits:
            K_max = K
            K += 1
        else:
            break
    return K_max

def kmax_for_patches(N_pixels, patch_h=2, patch_w=2, min_compression_ratio=0.5):
    """
    Original: N * 24 bits
    VQ 2x2: (N/4) * ceil(log2 K) + K * (12*8)  <=  0.5 * (N*24)
    """
    orig_bits = N_pixels * BITS_PER_PIXEL
    patches = (N_pixels // (patch_h * patch_w))
    CODEWORD_BITS = (patch_h * patch_w * CHANNELS) * BITS_PER_CHANNEL  # 12*8=96
    K, K_max = 1, 1
    while True:
        comp_bits = patches * ceil(log2(max(K, 2))) + K * CODEWORD_BITS
        if comp_bits <= min_compression_ratio * orig_bits:
            K_max = K
            K += 1
        else:
            break
    return K_max

def sample_Ks(Kmax, num_points=10):
    """
    Amostra 'num_points' valores inteiros de K uniformemente entre 1 e Kmax (inclui extremos).
    Mantém coisa viável em tempo de execução e mostra a tendência até o limite da taxa.
    """
    if Kmax <= num_points:
        return list(range(1, Kmax + 1))
    grid = np.unique(np.linspace(1, Kmax, num=num_points, dtype=int))
    return grid.tolist()


# Quantização por pixel
def vq_pixels(img, K):
    """Quantização por pixel: K-means em R^3 (RGB). Retorna imagem reconstruída."""
    H, W, _ = img.shape
    X = img.reshape(-1, 3)  # (N, 3)
    kmeans = MiniBatchKMeans(n_clusters=K, **KMEANS_KW)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.astype(np.float32)
    recon = centers[labels].reshape(H, W, 3)
    recon = np.clip(recon, 0.0, 1.0)
    return recon


# Quantização por blocos 2x2
def img_to_patches2x2(img):
    H, W, C = img.shape
    assert H % 2 == 0 and W % 2 == 0
    patches = []
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            patches.append(img[i:i+2, j:j+2, :].reshape(-1))  # (12,)
    return np.array(patches, dtype=np.float32)  # (N/4, 12)

def patches2x2_to_img(patches, H, W):
    recon = np.zeros((H, W, 3), dtype=np.float32)
    idx = 0
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            recon[i:i+2, j:j+2, :] = patches[idx].reshape(2, 2, 3)
            idx += 1
    return np.clip(recon, 0.0, 1.0)

def vq_patches2x2(img, K):
    H, W, _ = img.shape
    P = img_to_patches2x2(img)           # (N/4, 12)
    kmeans = MiniBatchKMeans(n_clusters=K, **KMEANS_KW)
    labels = kmeans.fit_predict(P)
    centers = kmeans.cluster_centers_.astype(np.float32)  # (K, 12)
    recon_patches = centers[labels]      # (N/4, 12)
    recon = patches2x2_to_img(recon_patches, H, W)
    return recon

def make_panel(img, Ks, mode="pixel", title="K-means VQ", save_gif=False, gif_path=None):
    """
    Mostra painel: [K=... imagens quantizadas ... | Original]
    mode: "pixel" (RGB por pixel) ou "patch2x2" (janelas 2x2)
    """
    H, W, _ = img.shape
    recs = []
    for K in Ks:
        if mode == "pixel":
            rec = vq_pixels(img, K)
        elif mode == "patch2x2":
            rec = vq_patches2x2(img, K)
        else:
            raise ValueError("mode deve ser 'pixel' ou 'patch2x2'")
        recs.append((K, rec))

    cols = len(Ks) + 1
    fig, axs = plt.subplots(1, cols, figsize=(3.2*cols, 3.4))
    fig.suptitle(title, fontsize=12)

    # Mostra quantizações
    for j, (K, rec) in enumerate(recs):
        axs[j].imshow(np.clip(rec, 0, 1))
        axs[j].set_title(f"K = {K}")
        axs[j].axis("off")

    # Mostra original no último painel
    axs[-1].imshow(np.clip(img, 0, 1))
    axs[-1].set_title("Original")
    axs[-1].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # (opcional) salva GIF progressivo
    if save_gif:
        try:
            import imageio.v2 as imageio
        except Exception:
            print("imageio não disponível; pulei GIF.")
            return

        frames = []
        # converte para uint8 e monta sequência K→Original
        for _, rec in recs:
            frames.append((np.clip(rec, 0, 1) * 255).astype(np.uint8))
        frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))

        if gif_path is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            gif_path = f"vq_{mode}_{ts}.gif"
        imageio.mimsave(gif_path, frames, duration=0.8)
        print(f"GIF salvo em: {gif_path}")

# Pipeline principal
def main(args):
    
    img = load_or_make_image(args.image, (TARGET_H, TARGET_W))
    Ks_demo = [2, 3, 10]             # ou [2, 4, 8, 16]
    make_panel(img, Ks_demo, mode="pixel", title="K-means (pixel)")
    H, W, _ = img.shape
    N = H * W

    # Descobre K_max sob compressão >= 50%
    Kmax_px = kmax_for_pixels(N, min_compression_ratio=0.5)
    Kmax_pt = kmax_for_patches(N, patch_h=2, patch_w=2, min_compression_ratio=0.5)

    # Amostra Ks (de 1 até Kmax)
    Ks_px = sample_Ks(Kmax_px, num_points=args.k_points)
    Ks_pt = sample_Ks(Kmax_pt, num_points=args.k_points)

    # Curvas de loss
    loss_px, loss_pt = [], []
    for K in Ks_px:
        recon = vq_pixels(img, K)
        loss_px.append(mse(img, recon))
    for K in Ks_pt:
        recon = vq_patches2x2(img, K)
        loss_pt.append(mse(img, recon))

    # aqui vamo plotar Loss x K
    plt.figure(figsize=(8, 5))
    plt.plot(Ks_px, loss_px, marker='o', label=f'Pixel-VQ (K até {Kmax_px})')
    plt.plot(Ks_pt, loss_pt, marker='s', label=f'VQ em janelas 2×2 (K até {Kmax_pt})')
    plt.xlabel('Número de clusters K')
    plt.ylabel('MSE de reconstrução')
    plt.title('Vector Quantization com K-means: Pixel vs. Janelas 2×2 (256×256 RGB)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Escolhe Ks representativos para visualização (¼ do Kmax)
    def closest_K(Ks, target):
        arr = np.array(Ks)
        return int(arr[np.argmin(np.abs(arr - target))])

    K_show_px = closest_K(Ks_px, max(2, Kmax_px // 4))
    K_show_pt = closest_K(Ks_pt, max(2, Kmax_pt // 4))

    recon_px = vq_pixels(img, K_show_px)
    recon_pt = vq_patches2x2(img, K_show_pt)

    # aq vamo plotar Original vs Reconstruções
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(np.clip(img, 0, 1));      axs[0].set_title('Original 256×256'); axs[0].axis('off')
    axs[1].imshow(np.clip(recon_px, 0, 1));  axs[1].set_title(f'Pixel-VQ (K={K_show_px})'); axs[1].axis('off')
    axs[2].imshow(np.clip(recon_pt, 0, 1));  axs[2].set_title(f'VQ 2×2 (K={K_show_pt})');  axs[2].axis('off')
    plt.tight_layout()
    plt.show()

    # Métricas para mostrar na tela 
    print(f"N_pixels = {N}")
    print(f"K_max (Pixel-VQ, ≥50% compressão) = {Kmax_px}")
    print(f"K_max (VQ 2×2, ≥50% compressão)   = {Kmax_pt}")
    print(f"Ks (Pixel-VQ) = {Ks_px}")
    print(f"Ks (VQ 2×2)   = {Ks_pt}")
    print(f"MSE Pixel-VQ @K={Ks_px[-1]} = {loss_px[-1]:.6f}")
    print(f"MSE VQ 2×2   @K={Ks_pt[-1]} = {loss_pt[-1]:.6f}")
    print(f"PSNR Pixel-VQ @K={K_show_px} = {psnr(img, recon_px):.2f} dB")
    print(f"PSNR VQ 2×2   @K={K_show_pt} = {psnr(img, recon_pt):.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Quantization com K-means (pixel vs. 2×2) para 256×256")
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem de entrada")
    parser.add_argument("--k_points", type=int, default=10, help="Quantidade de pontos de K amostrados de 1 até Kmax")
    args = parser.parse_args()
    main(args)
