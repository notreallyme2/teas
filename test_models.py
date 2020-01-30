import numpy as np
from collections import OrderedDict
from torch import tensor, FloatTensor
from models.linear import MiniLinearMLP

def test_mini_linear_mlp():
    """Test the MiniLinearMLP model"""
    mini_lmlp = MiniLinearMLP(input_dim=5, output_dim=1)
    # create an arbitrary input and weight matrix
    X = FloatTensor([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5]])
    state_dict = OrderedDict([(
        'weight', tensor([[-0.1453, -0.3046,  0.3570, -0.0946,  0.2181]])),
        ('bias', tensor([-0.3412])
        )])

    mini_lmlp.input_output.load_state_dict(state_dict, strict=True)
    y1, y2 = mini_lmlp(X)
    assert(y1 == tensor(0.6874))
    assert(y2 == tensor(0.6874))