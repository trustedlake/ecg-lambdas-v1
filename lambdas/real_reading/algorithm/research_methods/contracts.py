"""A strategy receives pixels and explicit geometry, never a source identity."""
from dataclasses import dataclass, field
import numpy as np


class MethodAbstention(ValueError):
    """An expected absence of evidence for this particular strategy."""


@dataclass
class Context:
    image: np.ndarray
    config: dict
    geometry: dict
    grid: dict


@dataclass
class Extraction:
    leads: list
    traces: list
    diagnostics: dict = field(default_factory=dict)

    def validate(self, shape):
        if not self.leads or len(self.leads) != len(self.traces):
            raise ValueError('Missing or mismatched leads/traces')
        if len({x['name'] for x in self.leads}) != len(self.leads):
            raise ValueError('Duplicate lead names')
        claims = np.zeros(shape[:2], np.uint16)
        for trace in self.traces:
            mask = trace['mask']
            x, y = trace['mask_origin']
            h, w = mask.shape
            if mask.dtype != bool or min(x, y) < 0 or x+w > shape[1] or y+h > shape[0]:
                raise ValueError('Invalid mask bounds/type')
            observed = mask.any(0)
            if not np.array_equal(observed, trace['observed']):
                raise ValueError('Mask/sample disagreement')
            for key in ('x_px', 'y_px', 'y_min_px', 'y_max_px', 'valid', 'ambiguous'):
                if np.shape(trace[key]) != (w,):
                    raise ValueError('Invalid sample shape: '+key)
            if not np.isnan(trace['y_px'][~observed]).all():
                raise ValueError('Invented samples in gaps')
            if np.any(trace['valid'] & (~observed | trace['ambiguous'])):
                raise ValueError('Invalid uncertainty flags')
            claims[y:y+h, x:x+w] += mask
        return {'shared_pixels': int((claims > 1).sum()),
                'selected_pixels': int((claims > 0).sum())}
