"""Geometry and ownership checks independent of any ECG identity."""
import cv2
import numpy as np
from PIL import Image
from ecg.preprocess import prepare
from research_methods.common import trace_full_photo
from research_methods.contracts import Context
import pytest


def test_full_photo_roundtrip_and_pixels_outside_calibration(tmp_path):
    rgb=np.full((300,400,3),255,np.uint8)
    rgb[15:25,15:25]=[25,40,170]
    path=tmp_path/'image.png';Image.fromarray(rgb).save(path)
    image,geometry=prepare(path,dict(max_side=400,
        quad_normalized=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]),full_photo=True)
    m=np.array(geometry['source_to_working']);inv=np.array(geometry['working_to_source'])
    corners=np.float32([[[0,0],[399,0],[399,299],[0,299]]])
    projected=cv2.perspectiveTransform(corners,m)
    assert np.all(projected>=0)
    assert np.all(projected[0,:,0]<image.shape[1])
    assert np.all(projected[0,:,1]<image.shape[0])
    np.testing.assert_allclose(cv2.perspectiveTransform(projected,inv),corners,atol=.001)
    assert ((image[:,:,2]>120)&(image[:,:,0]<60)).any()
    assert geometry['calibration_box'][0]>0
    region=[[.3,.3],[.7,.3],[.7,.7],[.3,.7]]
    other,other_geometry=prepare(path,dict(max_side=400,
        quad_normalized=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],
        calibration_region_normalized=region),full_photo=True)
    np.testing.assert_array_equal(image,other)
    assert geometry['source_to_working']==other_geometry['source_to_working']
    assert geometry['calibration_box']!=other_geometry['calibration_box']


def test_grid_phase_is_translated_to_full_canvas():
    from ecg.grid import estimate_grid
    rgb=np.full((450,600,3),255,np.uint8)
    rgb[::25,:,1:]=140;rgb[:,::25,1:]=140
    config=dict(grid_width_mm_hint=120,minor_square_mm=1,major_interval_minor_squares=5)
    geometry=dict(calibration_box=[53,37,553,437],layout_box=[0,0,600,450])
    local=dict(config,grid_width_mm_hint=100)
    expected=estimate_grid(rgb[37:437,53:553],local)
    full=estimate_grid(rgb,config,geometry)
    for axis,shift in [('x',53),('y',37)]:
        assert full[axis]['pixels_per_mm']==expected[axis]['pixels_per_mm']
        assert full[axis]['phase_px']==expected[axis]['phase_px']+shift
        np.testing.assert_allclose(full['controls'][axis],np.array(expected['controls'][axis])+shift)


def _fixture(crossing=False):
    raw=np.zeros((200,180),np.uint8)
    cv2.line(raw,(10,140),(165,140),1,1)
    cv2.line(raw,(75,140),(75,30),1,1)
    if crossing:cv2.line(raw,(10,60),(165,60),1,1)
    else:cv2.line(raw,(100,60),(165,60),1,1)
    leads=[dict(name='a',row=0,trace_x=[10,170],baseline_y_px=60,row_spacing_px=80),
           dict(name='b',row=1,trace_x=[10,170],baseline_y_px=140,row_spacing_px=80)]
    context=Context(np.repeat(raw[:,:,None]*255,3,2),{}, {},{})
    return raw,leads,context


def test_invasion_belongs_to_origin_not_neighbour_band():
    raw,leads,context=_fixture()
    result=trace_full_photo(context,raw,raw,leads)
    assert result.traces[1]['mask'][30,65]
    assert not result.traces[0]['mask'][30,65]
    assert result.validate(raw.shape)['shared_pixels']==0


def test_crossing_with_two_anchors_abstains_without_interpolation():
    raw,leads,context=_fixture(True)
    result=trace_full_photo(context,raw,raw,leads)
    assert result.diagnostics['ownership']['ambiguous_components']>0
    for trace in result.traces:
        assert np.isnan(trace['y_px']).all()
        assert not trace['valid'].any()
        assert not trace['mask'].any()


def test_unrelated_ambiguous_component_does_not_invalidate_other_row():
    raw=np.zeros((200,180),np.uint8)
    for y in [40,80,160]:cv2.line(raw,(10,y),(165,y),1,1)
    cv2.line(raw,(30,40),(30,80),1,1)
    leads=[dict(name=str(i),row=i,trace_x=[10,170],baseline_y_px=y,row_spacing_px=40)
           for i,y in enumerate([40,80,160])]
    result=trace_full_photo(Context(np.zeros((200,180,3),np.uint8),{},{},{}),raw,raw,leads)
    assert not result.traces[2]['ambiguous'].any()
    assert result.traces[2]['valid'][:155].all()


def test_handler_accepts_only_normalized_calibration_coordinates():
    import lambda_v1
    from ecg_contract import ContractError
    region=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]
    event=dict(request_id='00000000-0000-4000-8000-000000000001',layout='3x4+1',
        calibration=dict(speed_mm_s=25,gain_mm_mv=10,region_normalized=region),
        image=dict(bucket='offline',key='image.png'))
    assert lambda_v1._validate(event)[2]['region_normalized']==region
    event['calibration']['region_normalized']=[[False,0]]*4
    with pytest.raises(ContractError):lambda_v1._validate(event)
