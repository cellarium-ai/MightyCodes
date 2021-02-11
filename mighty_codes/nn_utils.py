import torch
from typing import List, Union, Tuple


ACTIVATION_MAP = {
    'sigmoid': torch.nn.Sigmoid(),
    'relu': torch.nn.ReLU(inplace=True),
    'softplus': torch.nn.Softplus(),
    'exp': torch.exp,
    'elu': torch.nn.ELU(inplace=True),
    'selu': torch.nn.SELU(inplace=True)
}


def generate_dense_nnet(dense_nnet_specs: List[Union[str, Tuple[int, int]]]) -> torch.nn.Module:
    assert len(dense_nnet_specs) > 0
    blocks = []
    for block_spec in dense_nnet_specs:
        if isinstance(block_spec, Tuple):
            blocks.append(torch.nn.Linear(block_spec[0], block_spec[1]))
        elif isinstance(block_spec, str):
            blocks.append(ACTIVATION_MAP[block_spec])
        else:
            raise ValueError(f'Invalid entry in dense nnet specifications: {block_spec}')
    return torch.nn.Sequential(*blocks)


def assert_nn_specs_io_features(
        nn_specs: List[Union[str, Tuple[int, int]]],
        in_features: int,
        out_features: int):
    
    # assert at least one tuple in nnet specs
    assert any(isinstance(block_specs, Tuple) for block_specs in nn_specs)
            
    first_dense_specs = None
    last_dense_specs = None
    for block_specs in nn_specs:
        if isinstance(block_specs, Tuple):
            if first_dense_specs is None:
                first_dense_specs = block_specs
            last_dense_specs = block_specs
            
    assert first_dense_specs[0] == in_features
    assert last_dense_specs[-1] == out_features
    
