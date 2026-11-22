"""Single Lambda: analyze original ECG or apply auditable lead-specific corrections."""
import json, os, tempfile, uuid, math
from pathlib import Path
import boto3
from PIL import Image, ImageDraw
from ecg.pipeline import process
from ecg.manual_correction import correct
from ecg.config import validate

MAX_INPUT_BYTES = int(os.environ.get('ECG_MAX_INPUT_BYTES', str(15 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get('ECG_MAX_IMAGE_PIXELS', str(24_000_000)))
ARTIFACTS={"result.json":"result","signals.npz":"signals","trace_masks.npz":"trace_masks","visao_geral.png":"visao_geral","working_image.png":"working_image","rejection_preview.png":"rejection_preview","correction_preview.png":"correction_preview"}

def _paint(strokes, lead, size, destination):
    width, height = size
    if not isinstance(strokes, list) or not 1 <= len(strokes) <= 200:
        raise ValueError('Invalid number of paint strokes')
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError('Working image dimensions exceed safe limit')
    total_points = 0
    for stroke in strokes:
        if not isinstance(stroke, dict) or stroke.get('lead') != lead or stroke.get('mode') not in ('include', 'exclude'):
            raise ValueError('Invalid lead or paint mode')
        points = stroke.get('points')
        if not isinstance(points, list) or not 1 <= len(points) <= 3000:
            raise ValueError('Invalid paint stroke points')
        total_points += len(points)
        if total_points > 12_000:
            raise ValueError('Too many painting points')
        for point in points:
            if not isinstance(point,dict) or any(not isinstance(point.get(k),(float,int)) or
                not math.isfinite(point[k]) or not 0 <= point[k] <= 1 for k in ('x','y')):
                raise ValueError('Painting coordinate must be normalized')
    masks = []
    brush = max(3, min(15, round(min(width, height) * 0.005)))
    for mode in ('include', 'exclude'):
        canvas = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for stroke in strokes:
            if stroke['lead'] != lead or stroke['mode'] != mode:
                continue
            pts = [(round(p['x'] * (width - 1)), round(p['y'] * (height - 1))) for p in stroke['points']]
            if len(pts) > 1:
                draw.line(pts, fill=255, width=brush, joint='curve')
            for x, y in pts:
                r = brush // 2
                draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
        file_path = destination / (mode + '.png')
        canvas.save(file_path)
        masks.append(file_path if canvas.getbbox() is not None else None)
    return masks

def _analysis_config(event):
    """Pass calibration stated by the user rather than assuming a training-folder profile."""
    cal = event.get('calibration')
    if not isinstance(cal, dict):
        raise ValueError('Explicit speed and gain are required')
    speed, gain = cal.get('speed_mm_s'), cal.get('gain_mm_mv')
    if type(speed) not in (float,int) or type(gain) not in (float,int) or \
       not math.isfinite(speed) or not math.isfinite(gain) or \
       speed not in (5, 12.5, 25, 50, 100) or gain not in (1, 2, 5, 10, 20):
        raise ValueError('Unsupported ECG speed or gain')
    config_path=Path(os.environ.get('ECG_CONFIG_PATH','configs/report_normal.json'))
    profile=json.loads(config_path.read_text(encoding='utf8'))
    profile['calibration']={**profile['calibration'], 'speed_mm_s':speed, 'gain_mm_mv':gain,
                            'speed_source':'Explicit user selection', 'gain_source':'Explicit user selection'}
    return validate(profile)

def handler(event, context):
    op=event.get('operation')
    if op not in ('analyze','refine'): raise ValueError('Unsupported operation')
    results_bucket=event['results_bucket']; attempt_id=str(uuid.UUID(str(event['attempt_id'])))
    s3=boto3.client('s3')
    with tempfile.TemporaryDirectory(dir='/tmp') as td:
        root=Path(td); out=root/'output'
        if op=='analyze':
            source_key=event.get('imagem_s3_key')
            if not isinstance(source_key, str) or not source_key.strip() or '..' in Path(source_key).parts:
                raise ValueError('Invalid S3 object key')
            suffix=Path(source_key).suffix.lower()
            if suffix not in ('.png','.jpg','.jpeg','.webp'): raise ValueError('Unsupported image format')
            source=root/('input'+suffix)
            info=s3.head_object(Bucket=event['images_bucket'], Key=source_key)
            if info['ContentLength'] > MAX_INPUT_BYTES:
                raise ValueError('ECG image exceeds maximum upload size')
            s3.download_file(event['images_bucket'],source_key,str(source))
            with Image.open(source) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError('ECG image dimensions exceed safe limit')
            metadata=process(source,out,_analysis_config(event))
            revision='initial'
        else:
            lead=event['lead']; strokes=event['strokes']; revision=str(uuid.uuid4())
            previous=root/'previous';previous.mkdir()
            keys=event['parent_artifacts']
            expected={'result':'result.json','signals':'signals.npz','trace_masks':'trace_masks.npz',
                      'working_image':'working_image.png','visao_geral':'visao_geral.png'}
            for field,filename in expected.items():
                key=keys.get(field)
                if not isinstance(key,str) or not key.startswith('ecg-analyser/'+attempt_id+'/') or not key.endswith('/'+filename):
                    # The displayed visao_geral may be the last correction preview.
                    if field != 'visao_geral' or not isinstance(key,str) or not key.startswith('ecg-analyser/'+attempt_id+'/') or not key.endswith('/correction_preview.png'):
                        raise ValueError('Missing or invalid previous artifact: '+field)
                s3.download_file(results_bucket,key,str(previous/filename))
            with Image.open(previous/'working_image.png') as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError('Working image exceeds safe pixel limit')
                positive,negative=_paint(strokes,lead,image.size,root)
            metadata=correct(previous,out,lead,positive,negative)
        artifacts={}
        for filename,field in ARTIFACTS.items():
            p=out/filename
            if p.is_file():
                key=f'ecg-analyser/{attempt_id}/{revision}/{filename}'
                s3.upload_file(str(p),results_bucket,key)
                artifacts[field]=key
        # Corrections don't generate a new general overlay, preserve the previous one.
        if op=='refine': artifacts['visao_geral']=event['parent_artifacts']['visao_geral']
        return {'status':metadata['status'],'warnings':metadata.get('warnings',[]),'grid':metadata.get('grid'),
                'leads':metadata.get('leads',[]),'artifacts':artifacts,
                'manual_correction':metadata.get('manual_correction')}
