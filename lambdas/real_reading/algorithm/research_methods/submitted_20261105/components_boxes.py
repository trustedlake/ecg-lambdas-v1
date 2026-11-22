#!/usr/bin/env python3
"""Extrai traçados de ECG de imagens com layout 12 derivações (3x4) + tira longa.

Utiliza apenas operações de processamento de imagem compatíveis com OpenCV 5.
Não interpreta / diagnostica o ECG. As amostras exportadas estão em pixels.

Uso:
    python extrair_ecg.py 55.jpg -o resultado
    python extrair_ecg.py outro_ecg.png -o resultado --threshold 85
    python extrair_ecg.py outro_ecg.png --layout-json regioes.json

Formato opcional do JSON: {"leads": [{"name":"I", "bbox":[x0,y0,x1,y1]}]}
As coordenadas bbox são pixels e a caixa termina antes de x1,y1.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Valores normalizados para eletrocardiogramas 3x4 + tira de ritmo.
# Esta é uma hipótese de layout, ajustável com --layout-json para outras impressões.
COLUMNS = [(0.035, 0.266), (0.275, 0.506), (0.512, 0.741), (0.748, 0.979)]
ROWS = [(0.104, 0.303), (0.326, 0.557), (0.563, 0.816)]
NAMES = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
PALETTE = [
    (255, 122, 0),    # azul-ciano em BGR
    (40, 165, 255),   # laranja em BGR
    (80, 220, 45),    # verde em BGR
    (190, 30, 245),   # magenta em BGR
]


def default_leads(width: int, height: int) -> list[dict]:
    leads = []
    for row in range(3):
        for col in range(4):
            xa, xb = COLUMNS[col]
            ya, yb = ROWS[row]
            leads.append({
                "name": NAMES[row][col],
                "bbox": [round(xa*width), round(ya*height), round(xb*width), round(yb*height)],
                "row": row,
            })
    leads.append({
        "name": "III_rhythm",
        "bbox": [round(0.035*width), round(0.822*height), round(0.979*width), round(0.952*height)],
        "row": 3,
    })
    return leads


def load_leads(path: Path | None, w: int, h: int) -> list[dict]:
    if path is None:
        return default_leads(w, h)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("leads"), list):
        raise ValueError("JSON precisa conter a lista 'leads'")
    leads = []
    for i, entry in enumerate(data["leads"]):
        x0, y0, x1, y1 = map(int, entry["bbox"])
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"bbox inválida: {entry['bbox']}")
        leads.append({"name": str(entry["name"]), "bbox": [x0, y0, x1, y1], "row": int(entry.get("row", i // 4))})
    return leads


def extract_component(gray: np.ndarray, bbox: list[int], threshold: int) -> tuple[np.ndarray, float, dict]:
    """Detecta a maior estrutura conectada escura dentro de uma derivação.

    Justificativa: nas imagens digitalizadas, o traçado forma uma linha conectada
    horizontalmente, enquanto grade e letras ocupam componentes menores.
    """
    x0, y0, x1, y1 = bbox
    roi = gray[y0:y1, x0:x1]
    # Grade impressa tem tons claros; o ECG costuma ser quase preto.
    binary = (roi < threshold).astype(np.uint8)
    # Ponte microfissuras de 1px sem unir objetos afastados.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), dtype=np.uint8))
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if nlabels < 2:
        return np.zeros_like(binary), 0.0, {"largest_component_pixels": 0}

    # Preferência por componentes que atravessam a região horizontalmente.
    # Letras e barras isoladas não devem ganhar do traçado.
    sizes = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    scores = sizes.astype(float) * (widths / max(1, roi.shape[1])) ** 2
    label_id = int(np.argmax(scores)) + 1
    component = (labels == label_id).astype(np.uint8)
    covered_columns = np.any(component, axis=0)
    coverage = float(np.mean(covered_columns))
    bbox_stats = stats[label_id]
    meta = {"largest_component_pixels": int(bbox_stats[cv2.CC_STAT_AREA]),
            "component_width_px": int(bbox_stats[cv2.CC_STAT_WIDTH]),
            "column_coverage": round(coverage, 4)}
    return component, coverage, meta


def sample_centerline(mask: np.ndarray, x0: int, y0: int) -> list[tuple[int, float, bool]]:
    """1 ponto por coluna x; coordenada y = mediana dos pixels detectados.

    Em QRS quase verticais, pode haver múltiplos pixels no mesmo x; para
    preservar a silhueta exata, utilize a máscara binária (salva em PNG/NPZ).
    """
    nrows, ncols = mask.shape
    result = []
    for xc in range(ncols):
        ys = np.flatnonzero(mask[:, xc])
        if ys.size:
            result.append((x0 + xc, y0 + float(np.median(ys)), True))
        else:
            result.append((x0 + xc, float("nan"), False))
    return result


def run(input_path: Path, output_dir: Path, threshold: int, layout_json: Path | None) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir {input_path}")
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    leads = load_leads(layout_json, width, height)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = img.copy()
    global_mask = np.zeros((height, width), dtype=np.uint8)
    mask_rgb = np.full_like(img, 255)
    records = []
    summary = {"image": str(input_path), "image_size": [width, height],
               "opencv_runtime": cv2.__version__, "threshold": threshold, "leads": []}

    for spec in leads:
        name = spec["name"]
        x0, y0, x1, y1 = spec["bbox"]
        mask, coverage, stats = extract_component(gray, spec["bbox"], threshold)
        if stats["largest_component_pixels"] == 0:
            print(f"[ATENÇÃO] {name}: nenhum componente detectado")
            continue
        if coverage < 0.75:
            print(f"[ATENÇÃO] {name}: apenas {coverage:.1%} das colunas cobertas")
        global_mask[y0:y1, x0:x1] = np.maximum(global_mask[y0:y1, x0:x1], mask * 255)

        color = PALETTE[spec["row"] % len(PALETTE)]
        # Realce apenas nos pixels da linha detectada, sem redesenhar curvas
        # inventadas durante QRSs com partes verticais ou em sobreposição.
        thick = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))) > 0
        sub = overlay[y0:y1, x0:x1]
        base = sub[thick].astype(np.float32)
        sub[thick] = np.clip(0.12 * base + 0.88 * np.array(color, np.float32), 0, 255).astype(np.uint8)

        # Uma cópia da máscara limpa com mesmas cores do overlay.
        colored = mask_rgb[y0:y1, x0:x1]
        colored[thick] = color

        samples = sample_centerline(mask, x0, y0)
        for x, y, detected in samples:
            records.append({"lead": name, "x_px": x,
                            "y_px": f"{y:.2f}" if detected else "",
                            "detected": int(detected)})
        info = {"name": name, "bbox": [x0, y0, x1, y1], **stats}
        summary["leads"].append(info)
        print(f"{name:>10} : {coverage:6.1%} colunas detectadas | {stats['largest_component_pixels']:5d} pixels")

    p_overlay = output_dir / "tracado_colorido.png"
    p_mask = output_dir / "mascara_ecg.png"
    p_centerline = output_dir / "centrolinha.png"
    p_samples = output_dir / "pontos_ecg.csv"
    p_binary = output_dir / "mascara_binaria.png"
    p_json = output_dir / "resumo.json"
    cv2.imwrite(str(p_overlay), overlay)
    cv2.imwrite(str(p_mask), mask_rgb)
    cv2.imwrite(str(p_binary), global_mask)

    # Visualização opcional da aproximação em 1 coordenada y por coluna.
    # Evita ligar lacunas (quando uma letra/rede é removida da máscara).
    centerline = img.copy()
    names_to_rows = {spec["name"]: spec["row"] for spec in leads}
    by_lead = {}
    for rec in records:
        by_lead.setdefault(rec["lead"], []).append(rec)
    for name, pts in by_lead.items():
        color = PALETTE[names_to_rows[name] % len(PALETTE)]
        for a, b in zip(pts[:-1], pts[1:]):
            if a["detected"] and b["detected"]:
                cv2.line(centerline,
                         (int(a["x_px"]), round(float(a["y_px"]))),
                         (int(b["x_px"]), round(float(b["y_px"]))), color, 2, cv2.LINE_AA)
    cv2.imwrite(str(p_centerline), centerline)
    with p_samples.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lead", "x_px", "y_px", "detected"])
        writer.writeheader()
        writer.writerows(records)
    p_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nArquivos salvos em: {output_dir.resolve()}")
    print("AVISO: coordenadas em pixels; não há calibração de tempo nem amplitude.")


def main():
    parser = argparse.ArgumentParser(description="Extrai curvas de ECG usando OpenCV 5")
    parser.add_argument("image", type=Path, help="Imagem JPG/PNG do ECG")
    parser.add_argument("-o", "--output", type=Path, default=Path("resultado_ecg"))
    parser.add_argument("--threshold", type=int, default=85, help="Limiar de preto (0-255, padrão: 85)")
    parser.add_argument("--layout-json", type=Path, help="Regiões (bbox) em JSON; se ausente usa 3x4 + tira")
    args = parser.parse_args()
    if not (1 <= args.threshold <= 254):
        parser.error("--threshold deve ficar entre 1 e 254")
    run(args.image, args.output, args.threshold, args.layout_json)


if __name__ == "__main__":
    main()
