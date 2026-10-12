"""Projection geometry checks with synthetic vectors; no model weights."""
import numpy as np
import pytest
from clip_explorer.embedding_space import project_joint_embeddings, point_table


def test_shared_pca_keeps_identical_cross_modal_vectors_together():
    image = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
    combined = np.concatenate([image,image])
    coordinates, variance = project_joint_embeddings(combined)
    np.testing.assert_allclose(coordinates[:3],coordinates[3:],atol=1e-12)
    np.testing.assert_allclose(coordinates.mean(0),0,atol=1e-12)
    assert variance.sum()==pytest.approx(1.)
    # This sample has rank <= 3: PCA retains all of its pairwise distances.
    for i in range(3):
        for j in range(3):
            assert np.linalg.norm(coordinates[i]-coordinates[j])==pytest.approx(np.linalg.norm(image[i]-image[j]))


@pytest.mark.parametrize("dimensions",[2,3])
def test_pca_determinism_and_variance_bounds(dimensions):
    vectors = np.random.default_rng(2021).normal(size=(24,16))
    a, variance = project_joint_embeddings(vectors,dimensions=dimensions)
    b, _ = project_joint_embeddings(vectors,dimensions=dimensions)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(24,dimensions)
    assert 0<variance.sum()<1


@pytest.mark.parametrize("dimensions",[2,3])
def test_tsne_is_seeded_and_finite(dimensions):
    vectors = np.random.default_rng(2021).normal(size=(8,5))
    a, variance = project_joint_embeddings(vectors,'t-SNE',3,dimensions=dimensions)
    b, _ = project_joint_embeddings(vectors,'t-SNE',3,dimensions=dimensions)
    assert variance is None
    assert a.shape==(8,dimensions)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a,b)


def test_tsne_rejects_invalid_neighborhood_size():
    with pytest.raises(ValueError,match='perplexity'):
        project_joint_embeddings(np.eye(4),'t-SNE',4)


def test_point_metadata_stays_aligned_with_modalities_and_query():
    records=[{'id':'cat','label':'cat','category':'Animals'},{'id':'car','label':'car','category':'Vehicles'}]
    coordinates=np.arange(15).reshape(5,3)
    frame=point_table(records,['a cat','a car'],coordinates,'a pet')
    assert frame.Modality.tolist()==['Image','Image','Text','Text','Query']
    assert frame.ID.tolist()==['cat','car','cat','car','custom_query']
    np.testing.assert_array_equal(frame[['X','Y','Z']].values,coordinates)


def test_2d_pca_is_the_same_first_two_principal_axes():
    vectors=np.random.default_rng(2021).normal(size=(16,8))
    two,var2=project_joint_embeddings(vectors,dimensions=2)
    three,var3=project_joint_embeddings(vectors,dimensions=3)
    np.testing.assert_allclose(two,three[:,:2],atol=1e-12)
    np.testing.assert_allclose(var2,var3[:2],atol=1e-12)


def test_2d_table_and_figure_do_not_invent_a_third_coordinate():
    from clip_explorer.embedding_space import embedding_figure
    records=[{'id':'cat','label':'cat','category':'Animals'},{'id':'car','label':'car','category':'Vehicles'}]
    frame=point_table(records,['a cat','a car'],np.arange(8).reshape(4,2))
    assert 'Z' not in frame
    plot=embedding_figure(frame,'PCA',2)
    assert all(trace.type=='scatter' for trace in plot.data)
    assert plot.layout.yaxis.scaleanchor=='x'
    assert plot.layout.xaxis.title.text=='PC 1'


@pytest.mark.parametrize('dimensions',[2,3])
def test_thumbnail_view_keeps_exact_coordinates_and_safe_captions(dimensions,monkeypatch):
    import json
    import re
    from PIL import Image
    from clip_explorer import thumbnail_space
    monkeypatch.setattr(thumbnail_space,'open_asset',lambda record:Image.new('RGB',(20,10),'red'))
    records=[{'id':'cat','label':'cat','category':'Animals'}, {'id':'car','label':'car','category':'Vehicles'}]
    coords=np.arange(4*dimensions,dtype=float).reshape(4,dimensions)/7
    frame=point_table(records,['a cat','a car'],coords)
    frame.loc[2,'Description']='</script><script>alert(1)</script>'
    html=thumbnail_space.thumbnail_document(frame,records,'PCA',False)
    raw=re.search(r'const data=(.*);\n',html).group(1)
    assert '</script>' not in raw
    payload=json.loads(raw)
    assert payload['dimensions']==dimensions
    assert payload['connect'] is False
    assert payload['points'][2]['Description']==frame.loc[2,'Description']
    for row,expected in zip(payload['points'],coords):
        np.testing.assert_array_equal([row[k] for k in ['X','Y','Z'][:dimensions]],expected)
        assert ('Z' in row)==(dimensions==3)
    assert set(payload['photos'])=={'cat','car'}
    assert all(src.startswith('data:image/png;base64,') for src in payload['photos'].values())
    assert '<script src=' not in html  # No runtime CDN dependency.
    assert payload['credits'][0]['id']=='cat'
