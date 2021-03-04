import os

from spirl.models.prl_bc_mdl import BCMdl
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.prl import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator
from spirl.components.logger import Logger


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': BCMdl,
    'logger': Logger,
    'data_dir': os.path.join(os.environ['DATA_DIR'], 'karel_dataset_stmt_cond_act_20d_trace_1.0_prob_2_trials_200_35k_7.5k_7.5k'),
    'epoch_cycles_train': 5,
    'evaluator': TopOfNSequenceEvaluator,
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    input_res=data_spec.res,
    n_input_frames=1,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = 1 + 1 + (model_config.n_input_frames - 1)
