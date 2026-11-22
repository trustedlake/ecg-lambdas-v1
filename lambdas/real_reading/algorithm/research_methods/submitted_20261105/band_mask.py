#!/usr/bin/env python3
"""Realça as curvas escuras de ECG em um papel de 12 derivações (3x4 + ritmo).

Usa apenas OpenCV (API compatível com OpenCV 5), NumPy e biblioteca padrão.

Exemplo:
    python extrair_ecg.py 50.jpg --saida ecg_mascara_sobreposta.png

Não faz diagnóstico; destina-se a localização/visualização do traçado.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def localizar_linhas_base(binario: np.ndarray, n_linhas: int = 4) -> list[int]:
    """Busca as quatro faixas horizontais com maior concentração de pixels escuros."""
    altura, largura = binario.shape
    esquerda, direita = int(largura * 0.035), int(largura * 0.98)
    proj = binario[:, esquerda:direita].sum(axis=1).astype(np.float32)
    proj = np.convolve(proj, np.ones(3) / 3, mode='same')
    elegiveis = (np.arange(altura) > int(altura * .07)) & (np.arange(altura) < int(altura * .94))
    distancia = int(altura * 0.16)
    bases = []
    for _ in range(n_linhas):
        candidatas = np.where(elegiveis, proj, -1)
        y = int(np.argmax(candidatas))
        if candidatas[y] <= 0:
            raise ValueError('Não foi possível detectar as linhas-base do ECG.')
        bases.append(y)
        elegiveis[max(0, y - distancia): min(altura, y + distancia)] = False
    return sorted(bases)


def extrair_mascara_ecg(imagem: np.ndarray, limiar: int = 140, area_minima: int = 35) -> tuple[np.ndarray, list[int]]:
    """Retorna a máscara binária uint8 (0/255) e as linhas base detectadas.

    Este extrator simples é destinado a registros no layout 3x4 com tira de ritmo.
    A supressão de rótulos usa regiões relativas; cada imagem deve ser validada.
    """
    altura, largura = imagem.shape[:2]
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    preto = (cinza < limiar).astype(np.uint8) * 255
    bases = localizar_linhas_base(preto > 0)

    # Apaga separadores altos entre derivações; não apaga picos R normais.
    altura_separador = max(15, int(altura * .23))
    sep = cv2.morphologyEx(preto, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, altura_separador)))
    preto[sep > 0] = 0

    mascara = np.zeros((altura, largura), dtype=np.uint8)
    x_ini, x_fim = int(largura * .033), int(largura * .98)
    passo = (x_fim - x_ini) / 4

    for indice, base in enumerate(bases):
        cima, baixo = max(0, base - round(altura * .125)), min(altura, base + round(altura * .075))
        banda = np.zeros_like(preto)
        banda[cima:baixo, x_ini:x_fim] = preto[cima:baixo, x_ini:x_fim]

        # Descarta rótulos (I, II, III, aVR, aVL, aVF, V1...V6) sem cortar
        # o trecho horizontal da linha-base abaixo do texto.
        inicios = [x_ini] if indice == 3 else [round(x_ini + k * passo) for k in range(4)]
        for inicio in inicios:
            x_a = max(0, inicio - round(largura * .002))
            x_b = min(largura, inicio + round(largura * .027))
            y_a = max(cima, base - round(altura * .042))
            y_b = max(y_a, base - round(altura * .007))
            banda[y_a:y_b, x_a:x_b] = 0

        # Descarta pequenas partículas isoladas da grade/textura do papel.
        n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(banda, 8)
        for id_componente in range(1, n):
            area = int(estatisticas[id_componente, cv2.CC_STAT_AREA])
            largura_c = int(estatisticas[id_componente, cv2.CC_STAT_WIDTH])
            if area >= area_minima and largura_c >= 5:
                mascara[rotulos == id_componente] = 255

    return mascara, bases


def sobrepor_mascara(imagem: np.ndarray, mascara: np.ndarray, cor_bgr: tuple[int, int, int] = (255, 0, 255),
                    opacidade: float = .85, espessura: int = 2) -> np.ndarray:
    """Aplica a máscara COLORIDA diretamente sobre a imagem original."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (espessura * 2 - 1, espessura * 2 - 1))
    visivel = cv2.dilate(mascara, kernel) if espessura > 1 else mascara
    colorida = imagem.copy()
    colorida[visivel > 0] = cor_bgr
    return cv2.addWeighted(colorida, opacidade, imagem, 1 - opacidade, 0)


def main():
    p = argparse.ArgumentParser(description='Extrai e sobrepõe máscara colorida sobre ECG de 12 derivações.')
    p.add_argument('imagem', help='Caminho do ECG original')
    p.add_argument('--saida', default='ecg_mascara_sobreposta.png', help='PNG com curva colorida sobre imagem original')
    p.add_argument('--mascara', default=None, help='Salvar máscara binária opcional')
    p.add_argument('--limiar', type=int, default=140, help='Intensidade máxima considerada preta (0-255)')
    p.add_argument('--espessura', type=int, default=2, help='Espessura visual do overlay (>=1)')
    args = p.parse_args()

    original = cv2.imread(args.imagem)
    if original is None:
        p.error(f'Não foi possível abrir: {args.imagem}')
    mascara, bases = extrair_mascara_ecg(original, limiar=args.limiar)
    resultado = sobrepor_mascara(original, mascara, espessura=max(1, args.espessura))
    if not cv2.imwrite(str(args.saida), resultado):
        raise IOError(f'Não foi possível salvar {args.saida}')
    if args.mascara:
        if not cv2.imwrite(str(args.mascara), mascara):
            raise IOError(f'Não foi possível salvar {args.mascara}')
    print(f'Faixas detectadas (y): {bases}')
    print(f'Imagem original + máscara colorida: {Path(args.saida).resolve()}')


if __name__ == '__main__':
    main()
