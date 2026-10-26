from pathlib import Path
import hashlib,json,time
import cv2
import numpy as np
from PIL import Image
from .config import resolve
from .preprocess import prepare
from .grid import estimate_grid
from .signal import ink_evidence,locate_leads,trace_lead
from .extract import convert
from .contrast_gate import assess_color_separation
from review.overlay import render


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


def code_hash():
    root=Path(__file__).resolve().parent.parent;digest=hashlib.sha256()
    for folder in ['ecg','review']:
        for p in sorted((root/folder).glob('*.py')):digest.update(p.read_bytes())
    return digest.hexdigest()


def process(source,out,config,page=0,resume=False):
    cv2.setNumThreads(1);start=time.monotonic();source=Path(source);out=Path(out)
    if not source.is_file():raise FileNotFoundError(source)
    # An output is a new dedicated directory, never the source directory.
    if out.resolve()==source.parent.resolve():raise ValueError('Output cannot be the input directory')
    resolved=resolve(config,source);cfg_hash=hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()
    signature={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'config_sha256':cfg_hash,'code_sha256':code_hash(),'page':page}
    result=out/'result.json'
    if out.exists() and any(out.iterdir()):
        if resume and result.is_file():
            previous=json.loads(result.read_text(encoding='utf8'))
            if previous['signature']==signature and (
                (previous['status'].startswith('REJECTED_') and (out/'rejection_preview.png').is_file()) or
                all((out/f).is_file() for f in ['visao_geral.png','signals.npz','trace_masks.npz'])):return previous
        marker=out/'run_signature.json'
        if not (resume and marker.is_file() and json.loads(marker.read_text(encoding='utf8'))==signature):
            raise FileExistsError(f'Output exists with a different signature: {out}. Use a new output directory.')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'run_signature.json',signature)
    a,geometry=prepare(source,resolved,page);grid=estimate_grid(a,resolved)
    profile=resolved.get('trace_profile','standard')
    if profile in ('faint_colored_trace', 'affine_faint_trace'):
        from . import signal_tinted as tracer
        if profile == 'affine_faint_trace':
            from . import signal_affine as tracer
        evidence,locator,trace_fn=tracer.ink_evidence,tracer.locate_leads,tracer.trace_lead
    elif profile == 'standard':
        evidence,locator,trace_fn=ink_evidence,locate_leads,trace_lead
    elif profile == 'dark_grid_trace':
        from . import signal_dark as tracer
        raw,excluded=tracer.ink_evidence(a,grid)
        leads=tracer.locate_leads(raw.astype('uint8'),resolved,geometry)
    elif profile == 'chromatic_ratio_paths':
        from . import signal_chromatic_paths as tracer
        from .grid_chromatic import estimate_grid as estimate_chromatic_grid
        grid=estimate_chromatic_grid(a,resolved)
        leads,path_traces,path_ownership=tracer.extract(a,grid,resolved,geometry)
    elif profile == 'dark_grid_paths':
        from . import signal_dark_paths as tracer
        leads,path_traces,path_ownership=tracer.extract(a,grid,resolved,geometry)
    else:
        raise ValueError('Unsupported trace_profile: '+str(profile))
    if profile not in ('dark_grid_trace','dark_grid_paths','chromatic_ratio_paths'):
        raw,contrast=evidence(a);leads=locator(raw,resolved,geometry)
    stroke_support=tracer.stroke_evidence(a)[0] if profile in ('faint_colored_trace', 'affine_faint_trace') else None
    traces=[];arrays={};masks={};records=[]
    for lead in ([] if profile in ('dark_grid_trace','dark_grid_paths','chromatic_ratio_paths') else leads):
        trace=trace_fn(raw,contrast,lead)
        if stroke_support is not None:trace=tracer.recover_stroke_pixels(stroke_support,trace,lead)
        traces.append(trace)
    refinement_names=resolved.get('trace_refinement_leads', [])
    if refinement_names:
        from copy import deepcopy
        proposed=deepcopy(traces)
    ownership=None
    if profile in ('dark_grid_paths','chromatic_ratio_paths'):
        traces,ownership=path_traces,path_ownership
    if profile=='dark_grid_trace':
        traces,ownership=tracer.trace_all(raw,excluded,leads)
    if profile in ('faint_colored_trace', 'affine_faint_trace'):
        from .trace_ownership import separate_traces
        ownership=separate_traces(traces,leads,a.shape)
    # Conservative color gate: reject confirmed physical-color collisions or
    # robustly observed achromatic (gray-only) trace/grid pairs per user policy.
    # This is not a clinical or full waveform-accuracy assessment.
    color_quality=assess_color_separation(a,leads,traces,grid)
    if color_quality['block']:
        from PIL import ImageDraw
        # A separate raw-image preview (no colored extracted trace overlay) for
        # human inspection. The original image is never altered.
        preview=Image.fromarray(a)
        ImageDraw.Draw(preview).rectangle((0,0,preview.width-1,preview.height-1),
                                          outline=(220,35,35),width=6)
        preview.save(out/'rejection_preview.png')
        status='REJECTED_LOW_TRACE_GRID_CONTRAST'
        meta={'schema_version':1,'source':str(source),'signature':signature,
              'configuration':resolved,'geometry':geometry,'grid':grid,
              'leads':[],'status':status,'warnings':[color_quality['reason']],
              'color_contrast':color_quality,
              'elapsed_s':round(time.monotonic()-start,3),
              'validation_scope':'User color policy: high-confidence trace/grid collision OR achromatic trace/grid; not clinical validation',
              'limitations':['Color-based rejection only; gray-only ECGs are rejected even with visible luminance contrast. No clinical validation.',
                             'No physical or pixel-waveform arrays exported after color rejection.']}
        write_json(result,meta)
        return meta
    peak_recovery=None
    if profile == 'standard':
        from .peak_recovery import recover
        peak_recovery=recover(a,leads,traces)
    refinement=None
    neutral_refinement=None
    if resolved.get('neutral_trace_leads'):
        from .signal_neutral import refine as refine_neutral
        neutral_refinement=refine_neutral(a,leads,traces,resolved['neutral_trace_leads'])
    printed_refinement=None
    if resolved.get('printed_trace_leads'):
        from .signal_printed import refine as refine_printed
        printed_refinement=refine_printed(a,leads,traces,resolved['printed_trace_leads'])
    segmented_refinement=None
    if resolved.get('segmented_trace_leads'):
        from .signal_segmented import refine as refine_segmented
        segmented_refinement=refine_segmented(a,leads,traces,resolved['segmented_trace_leads'])
    focal_contrast=None
    if resolved.get('focal_contrast_leads'):
        from .signal_focal_contrast import refine as refine_focal
        focal_contrast=refine_focal(a,leads,traces,resolved['focal_contrast_leads'])
        if resolved.get('focal_peak_ridges'):
            from .signal_peak_ridges import refine as refine_ridges
            focal_contrast['peak_ridges']=refine_ridges(a,leads,traces,resolved['focal_contrast_leads'])
        if resolved.get('focal_contrast_followup_leads'):
            focal_contrast['followup']=refine_focal(a,leads,traces,resolved['focal_contrast_followup_leads'])
    if refinement_names:
        from .trace_refinement import refine
        refinement=refine(a,raw,contrast,grid,leads,proposed,traces,refinement_names)
    if resolved.get('affine_peak_leads'):
        from .signal_affine import refine_peaks
        refine_peaks(a,leads,traces,resolved['affine_peak_leads'])
    if resolved.get('affine_peak_followup_leads'):
        from .signal_affine import refine_peaks
        refine_peaks(a,leads,traces,resolved['affine_peak_followup_leads'],avoid_owned=True)
    excursions=None
    if resolved.get('measured_excursion_directions'):
        from .signal_excursions import refine as refine_excursions
        excursions=refine_excursions(a,leads,traces,resolved['measured_excursion_directions'])
        if resolved.get('measured_excursion_followup'):
            excursions['followup']=refine_excursions(a,leads,traces,resolved['measured_excursion_followup'])
    for lead,trace in zip(leads,traces):
        values,calibrated=convert(trace,grid,resolved['calibration'],resolved.get('review'),lead['name'])
        for key,value in values.items():arrays[lead['name']+'__'+key]=value
        masks[lead['name']]=trace['mask']
        records.append({**lead,'baseline_y_px':trace['baseline_y_px'],'baseline_method':trace['baseline_method'],
                        'mask_origin':trace['mask_origin'],'coverage':trace['coverage'],'unambiguous_fraction':trace['unambiguous_fraction'],
                        'calibrated':calibrated,'samples':len(trace['x_px']),
                        'valid_samples':int(values['valid'].sum()),'local_time_origin':'first exported x pixel; no inferred cross-column synchronization'})
    warnings=[]
    if color_quality['reason'] not in ('TRACE_GRID_COLOR_SEPARATED',):
        warnings.append(color_quality['reason'])
    if geometry.get('orientation',{}).get('uncertain'):warnings.append('ORIENTATION_UNCERTAIN')
    if not grid['valid']:warnings.append('GRID_CALIBRATION_UNCERTAIN')
    if geometry['method'].endswith('fallback'):warnings.append('GEOMETRY_FALLBACK')
    if min(r['coverage'] for r in records)<.90:warnings.append('INCOMPLETE_TRACE')
    if min(r['unambiguous_fraction'] for r in records)<.85:warnings.append('AMBIGUOUS_TRACE')
    if any(resolved['calibration'].get(k) is None for k in ['speed_mm_s','gain_mm_mv']):warnings.append('MISSING_EXPLICIT_CALIBRATION')
    status='REVIEW_REQUIRED' if warnings else 'CONSISTENCY_PASS'
    np.savez_compressed(out/'signals.npz',**arrays);np.savez_compressed(out/'trace_masks.npz',**masks)
    Image.fromarray(a).save(out/'working_image.png')
    render(a,leads,traces,grid,resolved,status).save(out/'visao_geral.png',optimize=False,compress_level=1)
    meta={'schema_version':1,'source':str(source),'signature':signature,'configuration':resolved,'geometry':geometry,'grid':grid,
          'leads':records,'status':status,'warnings':warnings,'color_contrast':color_quality,'elapsed_s':round(time.monotonic()-start,3),
          'validation_scope':'Automated consistency, not waveform accuracy or clinical validation',
          'future_editing':{'grid_controls':grid['controls'],'control_interval_minor_squares':5,
                            'selection':resolved.get('review',{}),'non_destructive':True},
          'limitations':['Contrast-based segmentation can include text/grid or omit trace.',
                        'Baseline is a geometric median, not a clinical isoelectric reference.',
                        'NaN marks gaps; no interpolation, resampling or extrapolation.',
                        'No physical arrays when grid consistency fails; speed/gain only come from explicit input.',
                        'No automatic inference of layout, horizontal speed or vertical gain.']}
    if peak_recovery is not None:meta['peak_recovery']={'records':peak_recovery,'limitation':'Measured narrow strokes only; added columns uncertain; incomplete peaks remain.'}
    if refinement is not None:meta['trace_refinement']=refinement
    if neutral_refinement is not None:meta['neutral_trace_refinement']=neutral_refinement
    if printed_refinement is not None:meta['printed_trace_refinement']=printed_refinement
    if segmented_refinement is not None:meta['segmented_trace_refinement']=segmented_refinement
    if focal_contrast is not None:meta['focal_contrast_refinement']=focal_contrast
    if ownership is not None:meta['trace_ownership']=ownership
    if excursions is not None:meta['measured_excursions']={'leads':excursions,'limitation':'Explicit up/down visual excursion search; cross-row ownership remains heuristic. Added columns are uncertain; no interpolated pixels.'}
    write_json(result,meta)
    return meta
