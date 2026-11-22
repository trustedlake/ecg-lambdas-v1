"""Explicit API/folder configuration; no diagnostic or layout classification."""
from copy import deepcopy
from pathlib import Path
import json, re

STANDARD = [['I','aVR','V1','V4'], ['II','aVL','V2','V5'], ['III','aVF','V3','V6']]
COLORS = dict(zip(['I','II','III','aVR','-aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','II_longo'],
 ['#0878c9','#df4148','#278757','#915bc1','#915bc1','#c17b0a','#008d89','#c83e8e','#7461d6','#7a8b25','#058cae','#b65c31','#52759d','#344b5c']))


def validate(config):
    c=deepcopy(config)
    if not isinstance(c.get('calibration'),dict) or not all(k in c['calibration'] for k in ['speed_mm_s','gain_mm_mv']): raise ValueError('Explicit calibration fields are required; no defaults are inferred')
    if c.get('schema_version') != 1: raise ValueError('schema_version must be 1')
    if c.get('trace_profile','standard') not in ('standard','faint_colored_trace','affine_faint_trace','dark_grid_trace','dark_grid_paths','chromatic_ratio_paths'):
        raise ValueError('Unsupported trace_profile')
    targets=c.get('trace_refinement_leads', [])
    if not isinstance(targets,list) or any(not isinstance(n,str) for n in targets) or len(set(targets))!=len(targets):
        raise ValueError('trace_refinement_leads must be a list of unique lead names')
    if targets and c.get('trace_profile') not in ('faint_colored_trace','affine_faint_trace'):
        raise ValueError('trace_refinement_leads requires a faint trace profile')
    configured=[n for row in c.get('leads',[]) for n in row]+c.get('rhythm_leads',[])
    excursions=c.get('measured_excursion_directions',{})
    excursion_followup=c.get('measured_excursion_followup',{})
    if not isinstance(excursion_followup,dict) or any(n not in configured or d not in ('up','down') for n,d in excursion_followup.items()):
        raise ValueError('Invalid measured_excursion_followup')
    if excursion_followup and not excursions:raise ValueError('Excursion followup requires primary directions')
    if not isinstance(excursions,dict) or any(n not in configured or d not in ('up','down') for n,d in excursions.items()):
        raise ValueError('measured_excursion_directions requires configured leads and up/down directions')
    if excursions and (c.get('trace_profile')!='affine_faint_trace' or c.get('affine_peak_leads') or c.get('affine_peak_followup_leads') or c.get('trace_refinement_leads')):
        raise ValueError('Measured excursions require affine_faint_trace without other peak/refinement passes')
    if any(n not in configured for n in targets):
        raise ValueError('Unknown refinement lead')
    peaks=c.get('affine_peak_leads',[])
    if not isinstance(peaks,list) or any(not isinstance(n,str) for n in peaks) or len(set(peaks))!=len(peaks) or any(n not in configured for n in peaks):
        raise ValueError('affine_peak_leads must contain unique configured lead names')
    if peaks and c.get('trace_profile')!='affine_faint_trace':
        raise ValueError('affine_peak_leads requires affine_faint_trace')
    follow=c.get('affine_peak_followup_leads',[])
    if not isinstance(follow,list) or any(not isinstance(n,str) for n in follow) or len(set(follow))!=len(follow) or any(n not in configured for n in follow):
        raise ValueError('Invalid affine_peak_followup_leads')
    if follow and c.get('trace_profile')!='affine_faint_trace':
        raise ValueError('Follow-up requires affine_faint_trace')
    neutral=c.get('neutral_trace_leads',[])
    if not isinstance(neutral,list) or any(not isinstance(n,str) for n in neutral) or len(set(neutral))!=len(neutral):
        raise ValueError('neutral_trace_leads must be a list of unique lead names')
    if any(n not in configured for n in neutral):raise ValueError('Unknown neutral trace lead')
    if neutral and c.get('trace_profile','standard')!='standard':
        raise ValueError('neutral_trace_leads requires standard profile')
    printed=c.get('printed_trace_leads',[])
    if (not isinstance(printed,list) or any(not isinstance(n,str) for n in printed)
            or len(set(printed))!=len(printed) or any(n not in configured for n in printed)):
        raise ValueError('printed_trace_leads must contain unique configured lead names')
    if printed and (c.get('trace_profile','standard')!='standard' or neutral):
        raise ValueError('printed_trace_leads requires standard profile without neutral refinement')
    focal=c.get('focal_contrast_leads',[])
    if (not isinstance(focal,list) or any(not isinstance(n,str) for n in focal)
            or len(set(focal))!=len(focal) or any(n not in configured for n in focal)):
        raise ValueError('Invalid focal_contrast_leads')
    if focal and (c.get('trace_profile','standard')!='standard' or neutral or printed):
        raise ValueError('Focal contrast requires standard profile without other refinements')
    segmented=c.get('segmented_trace_leads',[])
    if (not isinstance(segmented,list) or any(not isinstance(n,str) for n in segmented)
            or len(set(segmented))!=len(segmented) or any(n not in configured for n in segmented)):
        raise ValueError('Invalid segmented_trace_leads')
    if segmented and (c.get('trace_profile','standard')!='standard' or neutral or printed or focal):
        raise ValueError('Segmented refinement requires standard profile without other refinements')
    if not isinstance(c.get('focal_peak_ridges',False),bool):
        raise ValueError('focal_peak_ridges must be boolean')
    if c.get('focal_peak_ridges') and not focal:
        raise ValueError('focal_peak_ridges requires focal_contrast_leads')
    followup=c.get('focal_contrast_followup_leads',[])
    if (not isinstance(followup,list) or any(not isinstance(n,str) for n in followup)
            or len(set(followup))!=len(followup) or any(n not in configured for n in followup)):
        raise ValueError('Invalid focal_contrast_followup_leads')
    if followup and (not focal or set(followup)&set(focal)):
        raise ValueError('Focal followup requires distinct primary focal leads')
    variants=c.get('variants') or [c]
    for v in variants:
        rows=v.get('leads',c.get('leads'))
        if not rows or not all(isinstance(r,list) and r for r in rows): raise ValueError('leads must be a nonempty matrix')
        if len({len(r) for r in rows})!=1: raise ValueError('Each layout row must have the same column count')
        names=[n for r in rows for n in r]+v.get('rhythm_leads',c.get('rhythm_leads',[]))
        if len(names)!=len(set(names)): raise ValueError('Lead identifiers must be unique; use II_longo for rhythm')
        if not all(n in COLORS for n in names): raise ValueError('Unsupported lead identifier')
    for key in ['speed_mm_s','gain_mm_mv']:
        value=c.get('calibration',{}).get(key)
        if value is not None and (not isinstance(value,(int,float)) or value<=0): raise ValueError(key+' must be positive or null')
    return c


def load_config(filename):
    return validate(json.loads(Path(filename).read_text(encoding='utf-8-sig')))


def resolve(config,source):
    assert_general_processing(config)
    c=deepcopy(config)
    variants=c.pop('variants',[])
    if variants:
        matches=[v for v in variants if re.search(v['filename_regex'],Path(source).name,re.I)]
        if len(matches)!=1: raise ValueError('Filename must select exactly one configured layout variant')
        c.update(matches[0])
    c['rows']=len(c['leads']); c['columns']=len(c['leads'][0])
    c['layout_label']=f"{c['rows']} linhas x {c['columns']} colunas"+(' + '+', '.join(c.get('rhythm_leads',[])) if c.get('rhythm_leads') else '')
    return c


def find_config(source,stop=None):
    p=Path(source).resolve()
    if p.is_file(): p=p.parent
    boundary=Path(stop).resolve() if stop else None
    for folder in [p,*p.parents]:
        f=folder/'ecg_config.json'
        if f.is_file(): return f
        if folder==boundary: break
    raise ValueError('Provide --config or ecg_config.json in the input folder hierarchy')


MANUAL_REFINEMENTS = (
    'trace_refinement_leads', 'affine_peak_leads', 'affine_peak_followup_leads',
    'neutral_trace_leads', 'printed_trace_leads', 'focal_contrast_leads',
    'focal_contrast_followup_leads', 'focal_peak_ridges', 'segmented_trace_leads',
    'measured_excursion_directions', 'measured_excursion_followup',
)

def assert_general_processing(config):
    """Reject historical case-by-case refinement switches at the runtime boundary."""
    selected = [key for key in MANUAL_REFINEMENTS if config.get(key)]
    if selected:
        raise ValueError('Image-specific refinement is retired: ' + ', '.join(selected))
    if config.get('variants'):
        raise ValueError('Supply an explicit layout; filename-based layout selection is retired')
