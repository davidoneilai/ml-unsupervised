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

def sample_Ks(Kmax, num_points=10, use_all_ks=False):
    """
    Amostra 'num_points' valores inteiros de K uniformemente entre 1 e Kmax (inclui extremos).
    Mantém coisa viável em tempo de execução e mostra a tendência até o limite da taxa.
    Se use_all_ks=True, retorna todos os valores de 1 até Kmax.
    """
    if use_all_ks or Kmax <= num_points:
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

    print(f"N_pixels = {N}")
    print(f"K_max (Pixel-VQ, ≥50% compressão) = {Kmax_px}")
    print(f"K_max (VQ 2×2, ≥50% compressão)   = {Kmax_pt}")

    # Usa TODOS os Ks até Kmax para o gráfico principal (como pedido pelo professor)
    # Mas se Kmax for muito grande (>50), usa amostragem para não travar (exceto se --all-ks for usado)
    use_all_px = args.all_ks or Kmax_px <= 50
    use_all_pt = args.all_ks or Kmax_pt <= 50
    
    Ks_px = sample_Ks(Kmax_px, num_points=args.k_points, use_all_ks=use_all_px)
    Ks_pt = sample_Ks(Kmax_pt, num_points=args.k_points, use_all_ks=use_all_pt)

    print(f"Testando {len(Ks_px)} valores de K para Pixel-VQ: {Ks_px[:5]}{'...' if len(Ks_px) > 5 else ''}")
    print(f"Testando {len(Ks_pt)} valores de K para VQ 2×2: {Ks_pt[:5]}{'...' if len(Ks_pt) > 5 else ''}")

    # Curvas de loss (K variando de 1 até Kmax para garantir 50% compressão)
    loss_px, loss_pt = [], []
    
    print("Processando Pixel-VQ...")
    for i, K in enumerate(Ks_px):
        if i % max(1, len(Ks_px)//5) == 0:  
            print(f"  K={K} ({i+1}/{len(Ks_px)})")
        recon = vq_pixels(img, K)
        loss_px.append(mse(img, recon))
    
    print("Processando VQ 2×2...")    
    for i, K in enumerate(Ks_pt):
        if i % max(1, len(Ks_pt)//5) == 0:
            print(f"  K={K} ({i+1}/{len(Ks_pt)})")
        recon = vq_patches2x2(img, K)
        loss_pt.append(mse(img, recon))

    # aqui vamo plotar Loss x K (GRÁFICO PRINCIPAL como pedido pelo luqueta)
    plt.figure(figsize=(10, 6))
    plt.plot(Ks_px, loss_px, marker='o', linewidth=2, markersize=4, 
             label=f'Pixel-VQ (K até {Kmax_px})', color='blue')
    plt.plot(Ks_pt, loss_pt, marker='s', linewidth=2, markersize=4,
             label=f'VQ em janelas 2×2 (K até {Kmax_pt})', color='orange')
    
    plt.xlabel('Número de clusters K')
    plt.ylabel('MSE de reconstrução')
    plt.title('Vector Quantization com K-means: Pixel vs. Janelas 2×2 (256×256 RGB)\n' + 
              'Loss de reconstrução com K variando de 1 até K_max (≥50% compressão)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # vamo colocar as informações sobre compressão no gráfico
    plt.text(0.02, 0.98, f'Taxa de compressão ≥ 50%\nPixel-VQ: K_max = {Kmax_px}\nVQ 2×2: K_max = {Kmax_pt}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
    print("\n" + "="*60)
    print("RESULTADOS FINAIS:")
    print("="*60)
    print(f"Ks testados (Pixel-VQ) = {len(Ks_px)} valores de 1 até {Kmax_px}")
    print(f"Ks testados (VQ 2×2)   = {len(Ks_pt)} valores de 1 até {Kmax_pt}")
    print(f"MSE final Pixel-VQ @K={Ks_px[-1]} = {loss_px[-1]:.6f}")
    print(f"MSE final VQ 2×2   @K={Ks_pt[-1]} = {loss_pt[-1]:.6f}")
    print(f"PSNR Pixel-VQ @K={K_show_px} = {psnr(img, recon_px):.2f} dB")
    print(f"PSNR VQ 2×2   @K={K_show_pt} = {psnr(img, recon_pt):.2f} dB")
    
    # Análise de qual estratégia é melhor
    if loss_px[-1] < loss_pt[-1]:
        melhor = "Pixel-VQ"
        diferenca = ((loss_pt[-1] - loss_px[-1]) / loss_pt[-1]) * 100
    else:
        melhor = "VQ 2×2"
        diferenca = ((loss_px[-1] - loss_pt[-1]) / loss_px[-1]) * 100
    
    print(f"\nCONCLUSÃO: {melhor} teve melhor performance com {diferenca:.1f}% menos erro")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Quantization com K-means (pixel vs. 2×2) para 256×256")
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem de entrada")
    parser.add_argument("--k_points", type=int, default=10, help="Quantidade de pontos de K amostrados (se não usar --all-ks)")
    parser.add_argument("--all-ks", action="store_true", help="Testa TODOS os valores de K de 1 até Kmax (pode ser lento)")
    args = parser.parse_args()
    main(args)
