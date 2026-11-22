import numpy as np
import cv2
from PIL import Image
from ecg.reference import prepare_reference
from ecg.extract import convert
from ecg.reference_trace import trace_reference
from research_methods.contracts import Context


def test_reference_keeps_exterior_rgb_and_coordinate_roundtrip(tmp_path):
    a=np.random.default_rng(7).integers(0,256,(400,600,3),dtype=np.uint8)
    path=tmp_path/'photo.png';Image.fromarray(a).save(path)
    image,g=prepare_reference(path,{'max_side':1000},[.2,.2,.8,.8],90)
    assert np.array_equal(image,np.rot90(a,-1))
    forward=np.asarray(g['source_to_working']);inverse=np.asarray(g['working_to_source'])
    assert np.allclose(inverse@forward,np.eye(3))
    assert np.allclose(forward@[0,0,1],[399,0,1])


def test_time_origin_is_independent_of_first_observed_column():
    trace=dict(x_px=np.array([10.,11.,12.]),y_px=np.array([20.,np.nan,21.]),
               valid=np.array([True,False,True]),observed=np.array([True,False,True]),
               ambiguous=np.zeros(3,bool),y_min_px=np.array([20.,np.nan,21.]),
               y_max_px=np.array([20.,np.nan,21.]),baseline_y_px=20.,time_origin_x_px=12.)
    out,_=convert(trace,{'valid':True,'x':{'pixels_per_mm':2},'y':{'pixels_per_mm':2}},
                  {'speed_mm_s':25,'gain_mm_mv':10})
    assert np.allclose(out['time_s'],[-.04,-.02,0])
    assert np.isnan(out['amplitude_mv'][1])


def test_trace_can_leave_all_four_reference_edges():
    raw=np.zeros((400,600),np.uint8)
    points=np.array([[60,220],[150,220],[200,30],[230,220],[300,220],
                     [350,370],[390,220],[550,220]],np.int32)
    cv2.polylines(raw,[points],False,1,1)
    config={'baseline_fractions':[.5],'columns':1,'leads':[['I']],
            'rhythm_leads':[],'signal_x_fraction':[0,1],'label_exclusion_fraction':0}
    result=trace_reference(Context(np.repeat(raw[:,:,None],3,2),config,
                          {'reference_box_px':[100,120,500,320]},{}),raw,raw.astype(float)*50)
    t=result.traces[0];result.validate(raw.shape)
    assert np.nanmin(t['y_px'])<120
    assert np.nanmax(t['y_px'])>320
    assert t['x_px'].min()<100 and t['x_px'].max()>500
    yy,xx=np.where(t['mask']);xx+=t['mask_origin'][0]
    assert raw[yy,xx].all()


def test_crossing_keeps_both_origins():
    raw=np.zeros((400,600),np.uint8)
    upper=np.array([[60,150],[180,150],[240,300],[300,150],[550,150]],np.int32)
    lower=np.array([[60,250],[550,250]],np.int32)
    cv2.polylines(raw,[upper,lower],False,1,1)
    config={'baseline_fractions':[.25,.75],'columns':1,'leads':[['I'],['II']],
            'rhythm_leads':[],'signal_x_fraction':[0,1],'label_exclusion_fraction':0}
    result=trace_reference(Context(np.repeat(raw[:,:,None],3,2),config,
                          {'reference_box_px':[100,100,500,300]},{}),raw,raw.astype(float)*50)
    upper_trace,lower_trace=result.traces
    assert np.nanmax(upper_trace['y_px'])>=299
    assert np.nanmax(lower_trace['y_px'])<=250
    assert lower_trace['observed'].sum()>450
