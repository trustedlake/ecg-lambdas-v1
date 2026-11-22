#!/usr/bin/env python3
"""Extrai o traco de ECG de uma imagem 12 derivacoes (layout 3x4 + DII longo).

Deps: pip install 'opencv-python>=5,<6' numpy
Uso: python extrair_ecg.py 51.jpg --output-dir resultado_ecg

Gera:
  ecg_tracado_colorido.png: imagem original com traco detectado colorido
  ecg_mascara.png: somente pixels de sinal (sem grade)
  ecg_pontos.csv: amostra por coluna/derivacao (coordenadas em pixels)

O arquivo de mascara e mais fiel do que converter uma coluna vertical do QRS
em um so ponto; o CSV e uma aproximacao de 1 ponto por coluna.
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
COLORS_BGR = [
    [(13, 173, 25), (0, 130, 230), (210, 85, 200), (205, 70, 20)],
    [(13, 173, 25), (0, 130, 230), (210, 85, 200), (205, 70, 20)],
    [(13, 173, 25), (0, 130, 230), (210, 85, 200), (205, 70, 20)],
]


def grouped_centers(indices):
    if len(indices) == 0:
        return []
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [int(round(np.mean(r))) for r in runs]


def find_dividers(dark, width):
    """Identifica as 3 barras verticais separadoras em ECG 3x4."""
    height = dark.shape[0]
    y0, y1 = int(height * 0.07), int(height * 0.68)
    counts = dark[y0:y1].sum(axis=0)
    xs = np.flatnonzero(counts > 0.72 * (y1 - y0))
    xs = xs[(xs > width * 0.12) & (xs < width * 0.90)]
    centers = grouped_centers(xs)
    if len(centers) != 3:
        # Layout convencional: 3 separadores, passos de 1/4 da area util.
        left = int(width * 0.033)
        centers = [int(left + (width - left) * k / 4) for k in (1, 2, 3)]
    return centers


def find_baselines(dark, dividers):
    """Localiza os 4 eixos horizontais pelos histogramas de pixels escuros."""
    height, width = dark.shape
    filtered = dark.copy()
    filtered[:, :int(width * 0.048)] = False  # calibracao / letras iniciais
    for x in dividers:
        filtered[:, max(0, x - 9):min(width, x + 10)] = False
    projections = filtered.sum(axis=1).astype(np.float32)
    smooth = cv2.GaussianBlur(projections.reshape(-1, 1), (1, 9), 0).ravel()
    smooth[:int(height * 0.08)] = 0
    smooth[int(height * 0.94):] = 0
    selected = []
    for _ in range(4):
        y = int(np.argmax(smooth))
        selected.append(y)
        gap = int(height * 0.17)
        smooth[max(0, y - gap):min(height, y + gap)] = 0
    return sorted(selected)


def trace_component(dark_crop):
    """Seleciona o componente conectado que percorre horizontalmente a derivacao.

    Selecionar o componente preserva os segmentos quase verticais do QRS,
    que um algoritmo de mediana por coluna apagaria.
    """
    binary = (dark_crop.astype(np.uint8) * 255)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros_like(binary)
    width = binary.shape[1]
    options = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= 0.35 * width]
    if not options:
        options = list(range(1, n))
    winner = max(options, key=lambda i: (stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_AREA]))
    return np.uint8(labels == winner) * 255


def extract(input_path, output_dir, threshold=110, thickness=3):
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Nao foi possivel abrir: {input_path}")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray < threshold  # elimina grade clara, mantem linha preta

    dividers = find_dividers(dark, width)
    baselines = find_baselines(dark, dividers)
    edges = [int(width * 0.033), *dividers, width]
    signal_mask = np.zeros((height, width), np.uint8)
    painted = image.copy()
    output_rows = []
    report = []

    for row_idx in range(4):
        baseline = baselines[row_idx]
        # Uma faixa independente para cada linha evita captar a linha vizinha.
        half_band = int(height * (0.074 if row_idx < 3 else 0.050))
        y0, y1 = max(0, baseline - half_band), min(height, baseline + half_band + 1)
        col_indices = range(4) if row_idx < 3 else range(1)
        for col_idx in col_indices:
            name = LEADS[row_idx][col_idx] if row_idx < 3 else "II_longo"
            if row_idx == 3:
                # A faixa continua ocupa toda a pagina apos calibracao/rotulo.
                x0, x1 = int(width * 0.047), width - 6
            else:
                # Ignora as letras de identificacao e os separadores verticais.
                x0 = edges[col_idx] + (18 if col_idx == 0 else 26)
                x1 = edges[col_idx + 1] - (4 if col_idx < 3 else 6)
            if x1 <= x0:
                continue
            region = trace_component(dark[y0:y1, x0:x1])
            signal_mask[y0:y1, x0:x1] = cv2.bitwise_or(signal_mask[y0:y1, x0:x1], region)
            color = COLORS_BGR[row_idx][col_idx] if row_idx < 3 else (0, 140, 245)
            thick_mask = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness)))
            crop = painted[y0:y1, x0:x1]
            crop[thick_mask > 0] = color

            valid = 0
            for lx in range(region.shape[1]):
                ys = np.flatnonzero(region[:, lx])
                if ys.size:
                    # Mediana apenas para exportacao; a mascara preserva toda a linha.
                    y_abs = y0 + int(round(np.median(ys)))
                    output_rows.append([name, x0 + lx, y_abs,
                                        y0 + int(ys.min()), y0 + int(ys.max()),
                                        baseline - y_abs])
                    valid += 1
            report.append((name, valid, x1 - x0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "ecg_tracado_colorido.png"), painted)
    cv2.imwrite(str(output_dir / "ecg_mascara.png"), signal_mask)
    with (output_dir / "ecg_pontos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["derivacao", "x_px", "y_central_px", "y_superior_px", "y_inferior_px", "amplitude_px"])
        writer.writerows(output_rows)
    print(f"OpenCV {cv2.__version__} | imagem {width}x{height} | limiar: {threshold}")
    print(f"Separadores x={dividers} | linhas centrais y={baselines}")
    for name, n, total in report:
        print(f"  {name:<9} {n}/{total} colunas com traco ({n / total:.1%})")
    print(f"Arquivos salvos em: {output_dir.resolve()}")
    return painted, signal_mask, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extrai traco de ECG 12 derivacoes com OpenCV 5")
    ap.add_argument("input", type=Path, help="PNG/JPG do ECG")
    ap.add_argument("--output-dir", type=Path, default=Path("resultado_ecg"))
    ap.add_argument("--threshold", type=int, default=110, help="Escuro < limiar = traco (padrao: 110)")
    ap.add_argument("--thickness", type=int, default=3, help="Espessura da marcacao na sobreposicao")
    args = ap.parse_args()
    extract(args.input, args.output_dir, args.threshold, args.thickness)
