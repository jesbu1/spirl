from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from spirl.components.base_model import BaseModel
from spirl.utils.pytorch_utils import map2np, ten2ar, RemoveSpatial, ResizeSpatial, map2torch, find_tensor
from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.modules.recurrent_modules import RecurrentPredictorTeacherEnforced, RecurrentPredictor
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
            'nz_mid_lstm': 128,              # number of dimensions for internal feature spaces
            'n_lstm_layers': 1,              # number of dimensions for internal feature spaces
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
        #self.p = nn.Sequential(
        #    nn.LSTM(input_size=self._hp.nz_vae, hidden_size=self._hp.nz_mid, num_layers=2),
        #    SelectItem(0),
        #    nn.Linear(self._hp.nz_mid, self._hp.action_dim + 1)
        #)
        self.p = RecurrentPredictorTeacherEnforced(self._hp,
        #TODO: groudn truth actions are incorrect
        #self.p = RecurrentPredictor(self._hp,
                                          input_size=self._hp.action_dim+1+self._hp.nz_vae,
                                          output_size=self._hp.action_dim + 1)
        self.decoder_input_initalizer = self._build_decoder_initializer(size=self._hp.action_dim + 1)
        self.decoder_hidden_initalizer = self._build_decoder_initializer(size=self.p.cell.get_state_size())
        self.q = nn.Sequential(
            nn.LSTM(input_size=self._hp.action_dim + 1, hidden_size=self._hp.nz_mid, num_layers=2, bidirectional=True),
            SelectItem(0),
            nn.Linear(self._hp.nz_mid * 2, self._hp.nz_vae * 2)
        )

    def forward(self, inputs, use_learned_prior=False):
        """
        forward pass at training time
        """ 
        output = AttrDict()
        inputs.observations = inputs.actions # for seamless evaluation
        output.q = self._run_inference(F.one_hot(self._net_inputs(inputs), num_classes=self._hp.action_dim+1).squeeze(-2).float())
        
        sampled_latent = output.q.rsample()
        
        output.pred_act = self._compute_output_dist(sampled_latent, cond_inputs=inputs.actions[:, 0], steps=inputs.actions.shape[1], gt_output=F.one_hot(inputs.actions.squeeze(-1), num_classes=self._hp.action_dim + 1).float()) # outputs logits
        #output.pred_act = self._compute_output_dist(sampled_latent, cond_inputs=inputs.actions[:, 0], steps=inputs.actions.shape[1]) # outputs logits
        output.reconstruction = output.pred_act.sample()
        return output
    
    def decode(self, z, cond_inputs, steps):
        """Runs forward pass of decoder given skill embedding.
        :arg z: skill embedding
        :arg cond_inputs: info that decoder is conditioned on
        :arg steps: number of steps decoder is rolled out
        """
        pred_act = self._compute_output_dist(z, cond_inputs=cond_inputs, steps=steps)
        reconstruction = torch.argmax(pred_act.logits, -1)
        stop_time = reconstruction == self._hp.action_dim
        for t in range(steps):
            if stop_time[0, t].item():
                break
        reconstruction = reconstruction[:, :t]
        return reconstruction
    
    def _run_inference(self, inputs):
        z = self.q(inputs)[:, -1]
        normal = torch.distributions.Normal(loc=z[..., :self._hp.nz_vae], scale=torch.clamp(z[..., self._hp.nz_vae:], min=-10, max=2).exp())
        return normal


    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        losses.nll = AttrDict(value=-model_output.pred_act.log_prob(self._regression_targets(inputs).squeeze(-1)).mean(), weight=1.0) #NLL()(model_output.pred_act, self._regression_targets(inputs))

        fixed_prior = torch.distributions.Normal(loc=torch.zeros_like(model_output.q.loc), scale=torch.ones_like(model_output.q.loc))
        losses.kl_loss = AttrDict(value=torch.distributions.kl.kl_divergence(model_output.q, fixed_prior).mean(), weight=self._hp.beta)

        losses.total = self._compute_total_loss(losses)
        return losses

    def _compute_output_dist(self, z, cond_inputs, steps, gt_output=None):
        lstm_init_input = self.decoder_input_initalizer(cond_inputs)
        lstm_init_hidden = self.decoder_hidden_initalizer(cond_inputs)
        decoder_pred = self.p(lstm_initial_inputs=AttrDict(x_t=lstm_init_input),
                                lstm_static_inputs=AttrDict(z=z),
                                steps=steps,
                                lstm_hidden_init=lstm_init_hidden,
                                lstm_gt_output=gt_output).pred
        return torch.distributions.Categorical(logits=decoder_pred)

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
        return 64       # return dummy resolution, images are not used by this model

    @property
    def latent_dim(self):
        return self._hp.nz_vae

    @property
    def state_dim(self):
        return self._hp.state_dim

    @property
    def prior_input_size(self):
        return self.state_dim

    @property
    def n_rollout_steps(self):
        return self._hp.n_rollout_steps

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass

    def _build_decoder_initializer(self, size):
        class FixedTrainableInitializer(nn.Module):
            def __init__(self, hp):
                super().__init__()
                self._hp = hp
                self.val = torch.zeros((1, size), requires_grad=True, device=self._hp.device)

            def forward(self, state):
                return self.val.repeat(find_tensor(state).shape[0], 1 )
                #return self.val.repeat(state.shape[0], 1)
        return FixedTrainableInitializer(self._hp)
