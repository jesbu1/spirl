from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import ProgramDataset


data_spec = AttrDict(
    dataset_class=ProgramDataset,
    n_actions=4 + 1,
    state_dim=16,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    res=32,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 100
