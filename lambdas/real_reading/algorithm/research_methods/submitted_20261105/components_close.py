#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrai e destaca o traçado de um ECG de 12 derivações (3x4 + tira longa).
Implementado com APIs compatíveis com OpenCV 5 e NumPy.

Saídas:
  - ecg_colorido.png       : original com cada curva destacada
  - ecg_mascara.png        : máscara binária de TODAS as curvas (sem grade)
  - ecg_derivacoes.npz     : máscaras binárias separadas por derivação
  - ecg_metadados.json     : posição e quantidade de pixels de cada curva

Uso:
  python extrair_ecg_opencv5.py exame.jpg --output resultado
  python extrair_ecg_opencv5.py exame.jpg --output resultado --threshold 125
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LEADS = (
    ("I",   "aVR", "V1", "V4"),
    ("II",  "aVL", "V2", "V5"),
    ("III", "aVF", "V3", "V6"),
)
# Cores BGR. A tira longa de DII fica em amarelo.
COLORS = (
    ((0, 195, 0), (0, 155, 245), (230, 70, 0), (190, 0, 195)),
    ((0, 185, 0), (0, 155, 245), (230, 70, 0), (190, 0, 195)),
    ((0, 185, 0), (0, 155, 245), (230, 70, 0), (190, 0, 195)),
)
RHYTHM_COLOR = (0, 200, 240)


def consecutive_groups(values: np.ndarray) -> list[np.ndarray]:
    """Agrupa índices consecutivos."""
    if values.size == 0:
        return []
    splits = np.where(np.diff(values) > 1)[0] + 1
    return [x for x in np.split(values, splits) if len(x)]


def find_baselines(ink: np.ndarray) -> list[int]:
    """Encontra as 4 linhas de base pelo acúmulo horizontal de tinta."""
    h, w = ink.shape
    left, right = int(w * 0.033), int(w * 0.98)
    density = np.count_nonzero(ink[:, left:right], axis=1)

    candidates = np.flatnonzero(density >= 0.19 * (right - left))
    groups = consecutive_groups(candidates)

    peaks = [
        max((int(y) for y in g), key=lambda y: density[y])
        for g in groups
    ]
    # Rejeita candidatos muito próximos (mesmo traçado dividido em dois picos).
    peaks = sorted(peaks, key=lambda y: density[y], reverse=True)
    chosen = []
    for y in peaks:
        if all(abs(y - p) > h * 0.095 for p in chosen):
            chosen.append(y)
        if len(chosen) == 4:
            break

    # Plano B para digitalizações com traçado mais claro/fino.
    if len(chosen) < 4:
        smoothed = np.convolve(density.astype(float), np.ones(5) / 5, mode="same")
        for y in np.argsort(smoothed)[::-1]:
            y = int(y)
            if all(abs(y - p) > h * 0.095 for p in chosen):
                chosen.append(y)
            if len(chosen) == 4:
                break

    if len(chosen) != 4:
        raise ValueError("Não foi possível identificar as quatro faixas do ECG.")
    return sorted(chosen)


def find_column_separators(ink: np.ndarray, baselines: list[int]) -> list[int]:
    """Localiza os três separadores verticais do formato ECG 3x4."""
    h, w = ink.shape
    top = max(0, baselines[0] - int(h * 0.043))
    bottom = min(h, baselines[2] + int(h * 0.043))
    tall = bottom - top

    density = np.count_nonzero(ink[top:bottom], axis=0)
    candidates = np.flatnonzero(density >= tall * 0.64)
    groups = [
        g for g in consecutive_groups(candidates)
        if 0.12 * w < np.mean(g) < 0.9 * w
    ]
    separators = [
        int(max(g, key=lambda x: density[x])) for g in groups
    ]
    separators.sort()

    # Se a página não tiver linhas separadoras, usa o layout padrão 3x4.
    if len(separators) != 3:
        start = int(w * 0.033)
        spacing = (w - start) / 4
        separators = [round(start + spacing * i) for i in (1, 2, 3)]
    return separators


def largest_trace_component(roi: np.ndarray) -> np.ndarray:
    """
    Seleciona o componente escuro que atravessa horizontalmente o painel.
    Isso preserva TODA a curva, inclusive os picos QRS quase verticais,
    e descarta a grade e as letras isoladas.
    """
    h, w = roi.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)

    eligible = [
        i for i in range(1, n)
        if stats[i, cv2.CC_STAT_WIDTH] >= 0.55 * w
        and stats[i, cv2.CC_STAT_AREA] >= 0.20 * w
    ]

    if not eligible:
        # Pequenas falhas de digitalização podem separar pedaços da curva.
        joined = cv2.morphologyEx(
            roi, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        )
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            joined, connectivity=8
        )
        eligible = [
            i for i in range(1, n)
            if stats[i, cv2.CC_STAT_WIDTH] >= 0.55 * w
        ]

    if not eligible:
        raise ValueError("Não foi encontrado um traçado contínuo neste painel.")

    best = max(eligible, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return np.where(labels == best, 255, 0).astype(np.uint8)


def process_ecg(image: np.ndarray, threshold: int = 125,
                halo: int = 1) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Retorna (imagem_colorida, mascara_global, mascaras_por_derivacao, metadados).
    As máscaras de cada derivação são arrays binários 0/255 no mesmo
    tamanho da imagem original.
    """
    if image is None:
        raise ValueError("Imagem inválida.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    h, w = gray.shape
    baselines = find_baselines(ink)
    separators = find_column_separators(ink, baselines)
    half_band = int(min(np.diff(baselines)) * 0.44)

    # Não inclui a régua de calibração no canto esquerdo.
    left_edge = int(w * 0.032)
    panel_spans = [
        (left_edge, separators[0] - 3),
        (separators[0] + 3, separators[1] - 3),
        (separators[1] + 3, separators[2] - 3),
        (separators[2] + 3, w - 4),
    ]

    output = image.copy()
    all_traces = np.zeros((h, w), np.uint8)
    lead_masks = {}
    metadata = {
        "image_shape": [h, w],
        "opencv_used": cv2.__version__,
        "threshold": threshold,
        "baselines_y": baselines,
        "column_separators_x": separators,
        "leads": [],
    }
    radius = max(0, int(halo))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )

    panels = []
    for row in range(3):
        for col in range(4):
            panels.append((LEADS[row][col], row, panel_spans[col], COLORS[row][col]))
    panels.append(("II_long", 3, (left_edge, w - 4), RHYTHM_COLOR))

    for lead, row, (x0, x1), color in panels:
        baseline = baselines[row]
        y0 = max(0, baseline - half_band)
        y1 = min(h, baseline + half_band)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Recorte inválido para {lead}")

        crop = ink[y0:y1, x0:x1]
        try:
            trace = largest_trace_component(crop)
        except ValueError as exc:
            raise ValueError(f"Falha na derivação {lead}: {exc}") from exc

        # Compõe máscara em coordenadas da imagem original.
        fullmask = np.zeros((h, w), np.uint8)
        fullmask[y0:y1, x0:x1] = trace
        lead_masks[lead] = fullmask
        all_traces = cv2.bitwise_or(all_traces, fullmask)

        # Apenas a visualização recebe espessura adicional.
        colored = cv2.dilate(trace, kernel) if radius else trace
        region = output[y0:y1, x0:x1]
        region[colored != 0] = color

        ys, xs = np.where(trace != 0)
        metadata["leads"].append({
            "name": lead,
            "panel": [x0, y0, x1, y1],
            "trace_bounds": [
                int(x0 + xs.min()), int(y0 + ys.min()),
                int(x0 + xs.max()), int(y0 + ys.max())
            ],
            "trace_pixels": int(np.count_nonzero(trace)),
            "color_bgr": list(color),
        })

    return output, all_traces, lead_masks, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Destaca curvas de ECG com OpenCV 5.")
    parser.add_argument("image", type=Path, help="Imagem JPG/PNG do ECG")
    parser.add_argument("--output", type=Path, default=Path("resultado_ecg"))
    parser.add_argument("--threshold", type=int, default=125)
    parser.add_argument("--halo", type=int, default=1,
                        help="Espessura adicional do destaque, em pixels (0-3)")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Não foi possível ler: {args.image}")
    args.output.mkdir(parents=True, exist_ok=True)

    color, mask, separate, meta = process_ecg(
        image, threshold=args.threshold, halo=args.halo
    )

    cv2.imwrite(str(args.output / "ecg_colorido.png"), color)
    cv2.imwrite(str(args.output / "ecg_mascara.png"), mask)
    np.savez_compressed(args.output / "ecg_derivacoes.npz", **separate)
    (args.output / "ecg_metadados.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OpenCV {cv2.__version__} — {len(separate)} traçados extraídos.")
    for lead in meta["leads"]:
        print(f"  {lead['name']}: {lead['trace_pixels']} pixels")
    print("Arquivos salvos em:", args.output.resolve())


if __name__ == "__main__":
    main()
