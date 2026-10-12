import math
import pytest
import torch
import torch.nn.functional as F
from clip_explorer.math_ops import (
    normalize_embeddings, cosine_similarity, compute_logits, compute_probabilities,
    contrastive_loss, temperature_logits, directional_probabilities,
    ensemble_embeddings, retrieval_ranking, similarity_margins, format_prompts,
)


def test_normalization_preserves_direction():
    x = torch.tensor([[3.,4.],[-5.,12.]])
    result = normalize_embeddings(x)
    assert torch.allclose(result.norm(dim=-1),torch.ones(2))
    assert torch.allclose(result[0],torch.tensor([.6,.8]))


@pytest.mark.parametrize('values',[torch.zeros(1,2),torch.tensor([[float('nan'),1.]])])
def test_undefined_embeddings_rejected(values):
    with pytest.raises(ValueError): normalize_embeddings(values)


def test_cosines_include_orthogonal_and_opposite_vectors():
    result = cosine_similarity(torch.tensor([[2.,0.],[0.,3.]]),torch.tensor([[7.,0.],[-4.,0.],[0.,5.]]))
    assert torch.allclose(result,torch.tensor([[1.,-1.,0.],[0.,0.,1.]]))


def test_learned_scale_is_exponentiated():
    embeddings = torch.eye(2)
    assert torch.allclose(compute_logits(embeddings,embeddings,math.log(17)),17*embeddings)


def test_uniform_contrastive_loss_equals_log_batch_size():
    li,lt,total = contrastive_loss(torch.zeros(4,4))
    assert li.item()==pytest.approx(math.log(4))
    assert lt.item()==pytest.approx(math.log(4))
    assert total.item()==pytest.approx(math.log(4))


def test_asymmetric_matrix_uses_both_directions():
    logits = torch.tensor([[3.,1.,0.],[2.,4.,1.],[1.,2.,5.]])
    li,lt,total = contrastive_loss(logits)
    row_loss = -logits.log_softmax(1).diag().mean()
    column_loss = -logits.log_softmax(0).diag().mean()
    assert torch.allclose(li,row_loss)
    assert torch.allclose(lt,column_loss)
    assert torch.allclose(total,(row_loss+column_loss)/2)


def test_caption_rotation_changes_targets_not_loss():
    logits = torch.tensor([[6.,2.,1.],[0.,4.,3.],[1.,2.,5.]])
    permutation = torch.tensor([1,2,0])
    targets = torch.tensor([2,0,1])
    original = contrastive_loss(logits)[2]
    shuffled = contrastive_loss(logits[:,permutation],targets)[2]
    assert torch.allclose(original,shuffled)
    assert contrastive_loss(logits[:,permutation])[2]>original


def test_positive_alignment_reduces_loss():
    low = contrastive_loss(torch.eye(4)*2)[2]
    high = contrastive_loss(torch.eye(4)*6)[2]
    assert high<low


def test_temperature_sharpens_distribution():
    scores = torch.tensor([[.7,.2],[.1,.6]])
    warm = compute_probabilities(temperature_logits(scores,1))
    cold = compute_probabilities(temperature_logits(scores,.1))
    assert cold[0,0]>warm[0,0]
    assert torch.allclose(temperature_logits(scores,.5),scores*2)


@pytest.mark.parametrize('tau',[0,-1,float('inf'),float('nan')])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError): temperature_logits(torch.eye(2),tau)


def test_row_and_column_probabilities_use_different_axes():
    logits = torch.tensor([[2.,7.,1.],[4.,3.,6.],[1.,5.,0.]])
    rows,columns = directional_probabilities(logits)
    assert torch.allclose(rows.sum(1),torch.ones(3))
    assert torch.allclose(columns.sum(0),torch.ones(3))
    assert torch.allclose(compute_probabilities(logits),rows)
    assert not torch.allclose(columns,rows)


def test_candidate_set_changes_probability_without_changing_cosine():
    first = compute_probabilities(torch.tensor([[3.,1.]]))[0,0]
    expanded = compute_probabilities(torch.tensor([[3.,1.,2.]]))[0,0]
    assert expanded<first


def test_ensemble_normalizes_before_averaging():
    # Unequal vector lengths would incorrectly favor the second prompt if averaged raw.
    prompts = torch.tensor([[[1.,0.],[0.,100.]],[[2.,0.],[4.,0.]]])
    result = ensemble_embeddings(prompts)
    assert torch.allclose(result.norm(dim=-1),torch.ones(2))
    assert torch.allclose(result[0],torch.tensor([1.,1.])/math.sqrt(2))
    assert torch.allclose(result[1],torch.tensor([1.,0.]))


def test_retrieval_is_cosine_ranked_and_stable_on_ties():
    gallery = torch.tensor([[0.,4.],[3.,0.],[9.,0.],[-2.,0.]])
    indices,scores = retrieval_ranking(gallery,torch.tensor([[7.,0.]]),3)
    assert indices.tolist()==[1,2,0]
    assert scores.tolist()==[1.,1.,0.]


def test_margin_signs_and_tie():
    result = similarity_margins(torch.tensor([.4,.2,.3]),torch.tensor([.2,.4,.3]))
    assert torch.allclose(result,torch.tensor([.2,-.2,0.]))


@pytest.mark.parametrize('template',['no placeholder','{} {}','{name}','{:s}','{} {bad}'])
def test_malformed_prompt_template_is_actionable(template):
    with pytest.raises(ValueError): format_prompts(['dog'],template)


def test_valid_prompt_template():
    assert format_prompts(['dog','cat'],'a photo of a {}')==['a photo of a dog','a photo of a cat']
