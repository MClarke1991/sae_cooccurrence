import pytest
import torch
from PIBBSS.normalised_cooc_functions import apply_activation_threshold

def test_apply_activation_threshold_basic():
    feature_acts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    threshold = 2.5
    internal_threshold = torch.tensor([0.1, 0.2, 0.3])
    
    result = apply_activation_threshold(feature_acts, threshold, internal_threshold)
    
    expected = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    assert torch.all(torch.isclose(result, expected))

def test_apply_activation_threshold_all_inactive():
    feature_acts = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    threshold = 4.0
    internal_threshold = torch.tensor([0.1, 0.2, 0.3])
    
    result = apply_activation_threshold(feature_acts, threshold, internal_threshold)
    
    expected = torch.zeros_like(feature_acts)
    assert torch.all(torch.isclose(result, expected))

def test_apply_activation_threshold_all_active():
    feature_acts = torch.tensor([[5.0, 6.0, 7.0], [5.5, 6.5, 7.5]])
    threshold = 1.0
    internal_threshold = torch.tensor([0.1, 0.2, 0.3])
    
    result = apply_activation_threshold(feature_acts, threshold, internal_threshold)
    
    expected = torch.ones_like(feature_acts)
    assert torch.all(torch.isclose(result, expected))

def test_apply_activation_threshold_edge_case():
    feature_acts = torch.tensor([[2.6, 3.2, 3.8], [2.5, 3.1, 3.7]])
    threshold = 2.5
    internal_threshold = torch.tensor([0.1, 0.2, 0.3])
    
    result = apply_activation_threshold(feature_acts, threshold, internal_threshold)
    
    expected = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    assert torch.all(torch.isclose(result, expected))

def test_apply_activation_threshold_invalid_input():
    feature_acts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    threshold = 2.5
    
    with pytest.raises(ValueError, match="internal_threshold must be a 1D tensor"):
        apply_activation_threshold(feature_acts, threshold, torch.tensor([[0.1, 0.2, 0.3]]))
    
    with pytest.raises(ValueError, match="feature_acts must be a 2D tensor with shape"):
        apply_activation_threshold(feature_acts, threshold, torch.tensor([0.1, 0.2]))

def test_apply_activation_threshold_different_shapes():
    feature_acts = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    threshold = 3.5
    internal_threshold = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    result = apply_activation_threshold(feature_acts, threshold, internal_threshold)
    
    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    assert torch.all(torch.isclose(result, expected))

def test_apply_activation_threshold_float_precision():
    feature_acts = torch.tensor([[2.99999, 3.00001], [3.49999, 3.50001]])
    threshold = 3.0
    internal_threshold = torch.tensor([0.0, 0.0])
    
    result = apply_activation_threshold(feature_acts, threshold, internal_threshold)
    
    expected = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    assert torch.all(torch.isclose(result, expected))