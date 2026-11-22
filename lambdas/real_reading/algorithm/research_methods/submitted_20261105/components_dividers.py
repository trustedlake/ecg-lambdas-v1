#!/usr/bin/env python3
"""Localiza traçados de um ECG 12 derivações (3x4 + tira de ritmo).

Usa somente OpenCV e NumPy (sem IA). Preserva o traçado completo, incluindo
segmentos quase verticais do QRS, por meio de componentes conectados.

Instalação: pip install opencv-python==5.0.0.93 numpy
Uso: python extrair_ecg_opencv5.py 53.jpg --output ecg_colorido.png

Saídas:
  ecg_colorido.png: original com as curvas coloridas;
  ecg_colorido_pontos.csv: cada pixel detectado (derivacao,x_px,y_px);
  ecg_colorido_centros.csv: um ponto representativo por coluna (derivacao,x_px,y_px).

NOTA: Os centros por coluna não são uma digitalização clínica validada. Em trechos
verticais do QRS, use os pontos completos (CSV) ou a máscara, pois um ponto por
coluna necessariamente perde informações verticais.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

LEADS = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
COLORS = [
    (15, 70, 245), (0, 170, 255), (15, 170, 15), (245, 50, 10),
    (200, 15, 210), (0, 160, 170), (245, 100, 70), (20, 105, 230),
    (200, 70, 40), (20, 200, 200), (170, 30, 230), (40, 190, 50),
    (15, 30, 255),
]  # BGR


def group_positions(values):
    """Transforma índices consecutivos em uma posição central por grupo."""
    values = np.asarray(values, dtype=int)
    if len(values) == 0:
        return []
    blocks = np.split(values, np.flatnonzero(np.diff(values) > 1) + 1)
    return [int(round(np.median(b))) for b in blocks]


def detect_layout(gray, threshold):
    """Detecta divisórias verticais e 4 linhas de base por projeções de tinta."""
    h, w = gray.shape
    ink = np.uint8(gray < threshold) * 255

    # Somente traços verticais MUITO longos: não pega espículas do QRS.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(50, round(h * 0.25)))
    )
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vertical_kernel)
    vcount = np.sum(vertical[:int(h * 0.73)] > 0, axis=0)
    x_candidates = np.flatnonzero(vcount > h * 0.28)
    dividers = [x for x in group_positions(x_candidates) if w * 0.17 < x < w * 0.92]
    if len(dividers) != 3:
        raise ValueError(
            f"Esperadas 3 divisórias verticais do formato 3x4; detectadas: {dividers}. "
            "Este layout exige adaptação para outro formato de ECG."
        )

    # Remove as divisórias e a régua de calibração antes de procurar baselines.
    ink[:, :max(20, round(w * 0.035))] = 0
    ink[:, -max(10, round(w * 0.015)):] = 0
    for x in dividers:
        ink[:, max(0, x-7):min(w, x+8)] = 0

    row_hist = (ink > 0).sum(axis=1).astype(np.float32)
    kernel = np.ones(11, np.float32) / 11
    smooth = np.convolve(row_hist, kernel, mode='same')
    smooth[:round(h * .06)] = 0
    smooth[round(h * .96):] = 0
    min_sep = max(60, round(h * .16))
    baselines = []
    for _ in range(4):
        y = int(np.argmax(smooth))
        if smooth[y] < 10:
            break
        baselines.append(y)
        smooth[max(0, y-min_sep):min(h, y+min_sep+1)] = 0
    baselines.sort()
    if len(baselines) != 4:
        raise ValueError(f"Esperadas 4 faixas horizontais; obtidas {baselines}")
    return dividers, baselines


def connected_trace(gray_roi, threshold, min_width_fraction=0.68):
    """Seleciona a componente escura que percorre horizontalmente a ROI.

    A grade é muito mais clara que a curva nesta classe de imagens.
    O critério de extensão horizontal evita selecionar textos e pequenas manchas.
    """
    bw = np.uint8(gray_roi < threshold)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return np.zeros_like(bw), 0.0

    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        if width >= gray_roi.shape[1] * min_width_fraction and area >= width * 1.3:
            candidates.append((width, area, i))

    if not candidates:
        return np.zeros_like(bw), 0.0

    _, _, largest_id = max(candidates)
    curve = np.uint8(labels == largest_id)
    # Proporção de colunas com um ou mais pixels atribuídos ao traçado.
    coverage = float(np.mean(np.any(curve != 0, axis=0)))
    return curve, coverage


def extract_ecg(image, threshold=135, label_margin=23, stroke=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    dividers, baselines = detect_layout(gray, threshold)
    pitch = float(np.median(np.diff(baselines)))
    half_band = int(round(pitch * .40))
    left = max(40, int(w * .032))
    right = min(w - 8, int(w * .985))
    boundaries = [left] + dividers + [right]
    masks = []
    output = image.copy()
    detected = []

    def process(lead, x1, x2, base, color):
        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, int(base-half_band))
        y2 = min(h, int(base+half_band+1))
        if x2 <= x1:
            return
        trace, coverage = connected_trace(gray[y1:y2, x1:x2], threshold)
        if not trace.any():
            print(f"AVISO: {lead} sem curva detectável ({x1}:{x2},{y1}:{y2})")
            return
        ys, xs = np.nonzero(trace)
        xs_global, ys_global = xs + x1, ys + y1
        # Um pouco mais espesso para a conferência visual: apenas os pixels
        # pertencentes ao componente do traçado serão usados como base.
        view_mask = cv2.dilate(trace, np.ones((2*stroke+1, 2*stroke+1), np.uint8))
        overlay = output[y1:y2, x1:x2]
        overlay[view_mask != 0] = (
            0.12 * overlay[view_mask != 0].astype(np.float32)
            + 0.88 * np.asarray(color, np.float32)
        ).astype(np.uint8)
        masks.append((lead, xs_global, ys_global))
        detected.append((lead, round(coverage, 3), len(xs)))

    for r, row in enumerate(LEADS):
        for c, lead in enumerate(row):
            x1 = boundaries[c] + (label_margin if c == 0 else label_margin + 8)
            x2 = boundaries[c+1] - 6
            process(lead, x1, x2, baselines[r], COLORS[r*4+c])

    # Tira de ritmo de 12 derivações: aqui está identificada como III.
    process("III_rhythm", left + label_margin, right - 6, baselines[3], COLORS[12])
    return output, masks, detected, dividers, baselines


def save_csv(output_path, masks):
    base = Path(output_path).with_suffix('')
    pixel_csv = base.with_name(base.name + '_pontos.csv')
    center_csv = base.with_name(base.name + '_centros.csv')
    with pixel_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lead', 'x_px', 'y_px'])
        for lead, xx, yy in masks:
            writer.writerows((lead, int(x), int(y)) for x, y in zip(xx, yy))

    with center_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lead', 'x_px', 'y_px'])
        for lead, xx, yy in masks:
            for x in np.unique(xx):
                ys = yy[xx == x]
                writer.writerow([lead, int(x), round(float(np.median(ys)), 2)])
    return pixel_csv, center_csv


def main():
    p = argparse.ArgumentParser(description='Extrai curvas ECG com OpenCV 5 (layout 3x4 + ritmo)')
    p.add_argument('image', help='Imagem ECG (JPG/PNG)')
    p.add_argument('--output', default='ecg_colorido.png')
    p.add_argument('--threshold', type=int, default=135, help='Limiar do preto (padrão 135)')
    p.add_argument('--label-margin', type=int, default=23, help='Margem para evitar o texto das derivações')
    p.add_argument('--stroke', type=int, default=1, help='Dilatação visual em pixels')
    args = p.parse_args()
    if args.stroke < 0 or args.stroke > 8:
        p.error('--stroke deve estar entre 0 e 8')
    image = cv2.imread(args.image)
    if image is None:
        p.error(f'Não foi possível abrir: {args.image}')
    output, masks, detected, dividers, baselines = extract_ecg(
        image, args.threshold, args.label_margin, args.stroke
    )
    cv2.imwrite(args.output, output)
    pixels, centers = save_csv(args.output, masks)
    print(f'OpenCV {cv2.__version__}; divisórias={dividers}; baselines={baselines}')
    for lead, coverage, count in detected:
        print(f'{lead:>10}: cobertura horizontal {coverage:.1%}, {count} pixels')
    print(f'Salvos: {args.output}, {pixels}, {centers}')


if __name__ == '__main__':
    main()
