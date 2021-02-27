from contextlib import contextmanager
import torch
import torch.nn as nn
import copy

from spirl.components.base_model import BaseModel
from spirl.utils.general_utils import AttrDict, ParamDict
#from spirl.utils.pytorch_utils import RemoveSpatial, ResizeSpatial
from spirl.modules.variational_inference import ProbabilisticModel, MultivariateGaussian

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class BCMdl(BaseModel):
    """Simple recurrent forward predictor network with image encoder and decoder."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        #self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self.device = self._hp.device

        self.build_network()

    def _default_hparams(self):
        # put new parameters in here:
        return super()._default_hparams().overwrite(ParamDict({
            'use_convs': False,
            'device': None,
            'state_dim': 1,             # dimensionality of the state space
            'action_dim': 1,            # dimensionality of the action space
            'nz_mid': 128,              # number of dimensions for internal feature spaces
            'nz_vae': 5,              # number of dimensions for internal feature spaces
            'n_processing_layers': 5,   # number of layers in MLPs
            'output_type': 'gauss',     # distribution type for learned prior, ['gauss', 'gmm', 'flow']
            'n_gmm_prior_components': 5,    # number of Gaussian components for GMM learned prior
            'beta': 0.1,
        }))

    def build_network(self):
        assert not self._hp.use_convs   # currently only supports non-image inputs
        assert self._hp.output_type == 'gauss'  # currently only support unimodal output
        #self.net = Predictor(self._hp, input_size=self._hp.state_dim, output_size=self._hp.action_dim * 2)
        self.p = nn.Sequential(
            nn.LSTM(input_size=self._hp.action_dim, hidden_dim=self._hp.nz_mid, num_layers=1),
            SelectItem(0),
            nn.Linear(self._hp.nz_mid, self._hp.action_dim)
        )
        self.q = nn.Sequential(
            nn.LSTM(input_size=self._hp.action_dim, hidden_dim=self._hp.nz_mid, num_layers=1, bidirectional=True),
            SelectItem(0),
            nn.Linear(self._hp.nz_mid, self._hp.nz_vae * 2)
        )

    def forward(self, inputs, use_learned_prior=False):
        """
        forward pass at training time
        """ 
        output = AttrDict()
        output.q = self._run_inference(self._net_inputs(inputs))
        
        sampled_latent = output.q.rsample()

        output.pred_act = self._compute_output_dist(sampled_latent) # outputs logits
        return output

    def _run_inference(self, inputs):
        z = self.q(inputs)
        normal = torch.distributions.Normal(loc=z[..., :self._hp.nz_vae], scale=torch.clamp(z[..., self._hp.nz_vae:], min=-10, max=2).exp())
        return normal


    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        losses.nll = - torch.distributions.Categortical(model_output.pred_act).log_prob(self._regression_targets(inputs)) * inputs.valid_mask #NLL()(model_output.pred_act, self._regression_targets(inputs))

        fixed_prior = torch.distributions.Normal(loc=torch.zeros_like(model_output.q), scale=torch.ones_like(model_output.q))
        losses.kl_loss = AttrDict(value=torch.distributions.kl.kl_divergence(model_output.q, fixed_prior), weight=self._hp.beta)

        losses.total = self._compute_total_loss(losses)
        return losses

    def _compute_output_dist(self, inputs):
        return torch.distributions.Categorical(self.net(inputs))

    def _net_inputs(self, inputs):
        return inputs.actions

    def _regression_targets(self, inputs):
        return inputs.actions

    def compute_learned_prior(self, inputs, first_only=False):
        """Used in BC prior regularized RL policies."""
        assert first_only is True       # do not currently support ensembles for BC model
        if len(inputs.shape) == 1:
            return self._compute_output_dist(inputs[None])[0]
        else:
            return self._compute_output_dist(inputs)

    @property
    def resolution(self):
        return 64  # return dummy resolution, images are not used by this model

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass
