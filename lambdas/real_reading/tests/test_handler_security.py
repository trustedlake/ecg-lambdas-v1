import json
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from lambda_handler import _paint, _analysis_config

LEAD='II'
def strokes(x=.5,y=.5,mode='include'):
    return [{'lead':LEAD,'mode':mode,'points':[{'x':x,'y':y}]}]

def test_rejects_nonfinite_and_out_of_range_paint():
    with TemporaryDirectory() as td:
        for x in [float('nan'),float('inf'),-0.01,1.01,'0.5',None]:
            with pytest.raises(ValueError): _paint(strokes(x),LEAD,(800,600),Path(td))

def test_rejects_excessive_paint():
    with TemporaryDirectory() as td:
        with pytest.raises(ValueError): _paint(strokes()*201,LEAD,(800,600),Path(td))
        with pytest.raises(ValueError): _paint([{'lead':LEAD,'mode':'include','points':[{'x':.5,'y':.5}]*3001}],LEAD,(800,600),Path(td))

def test_refuses_pixels_above_limit():
    with TemporaryDirectory() as td:
        with pytest.raises(ValueError): _paint(strokes(),LEAD,(80000,6000),Path(td))

def test_rejects_boolean_calibration():
    with pytest.raises(ValueError): _analysis_config({'calibration':{'speed_mm_s':25,'gain_mm_mv':True}})

def test_allowed_strokes_make_mask():
    with TemporaryDirectory() as td:
        a,b=_paint(strokes(),LEAD,(800,600),Path(td))
        assert a is not None and a.is_file() and b is None
