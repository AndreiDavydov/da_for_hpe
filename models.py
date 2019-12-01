import torch as th
import torch.nn as nn
from resnet.resnet import resnet50

PATH_RESNET50_PRETRAINED = './resnet/resnet50.pth'

SEED = 0

class DamNet(nn.Module):
    ''' Produces DamNet (based on the DAMN paper)
    with the ResNet-50 as a backbone.
    As written in the original paper, this model
    "encodes sample X from domain d into a feature 
    vector Z = f(X, d)".

    # named_children of ResNet50: 
    ## layer0: "conv1", "bn1", "relu", "maxpool" 
    ## "layer1", "layer2", "layer3", "layer4"
    ## "avgpool", "fc"
    '''
    def __init__(self, num_domains=2, pretrained=True, z_features=256, debug=False):
        super(DamNet, self).__init__()

        self.num_domains = num_domains
        self.num_flows = num_domains 
        self.z_features = z_features # dimension of latent vector Z
        # can vary from unit to unit, here it is fixed

        ########### for checking correct ResNet50 classification quality.
        self.debug = debug
        ###########

        th.manual_seed(SEED)

        flows = []
        for _ in range(self.num_flows): 
            flow = resnet50(pretrained=False)
            if pretrained:
                flow.load_state_dict(th.load(PATH_RESNET50_PRETRAINED)) 
            flow = self._compose_flow(flow)
            flows.append(flow)                
                                                                        
        self.units_names = flow.keys()                           
        self.units = self._compose_units(flows)  

        # in case of sigmoid == 1/2, plasts would be equal 0.000,
        # it would zero gradients!  
        self.gate_init_val = th.tensor(1. / (self.num_flows + 1.)).float()
        self.gates_ws, self.plasts_ws = self._init_gates()

    def get_gates(self):
        gates = {}
        for key in self.gates_ws.keys(): 
            gates_ws = self.gates_ws[key].data.view(-1)
            plasts_ws = self.plasts_ws[key].data.view(-1)
            
            gates[key] = th.sigmoid(gates_ws*plasts_ws)
        return gates
        
    def _init_gates(self):
        th.manual_seed(SEED)
        gates_ws, plasts_ws = nn.ParameterDict(), nn.ParameterDict() 
        # HOW TO DO IT CORRECTLY, ParamDict of ParamDicts?
        for domain_idx in range(self.num_domains):
            key = 'domain_'+str(domain_idx)
            for unit_name in self.units_names:
                # Not clear how to init plasticities correctly. 
                # So far all gates are equal at the first epoch.
                key_unit = key+'_'+unit_name

                sqrt_log = th.sqrt(th.abs(th.log(1. / self.gate_init_val - 1.))) # if pi == g
                gates_ws[key_unit] = nn.Parameter(
                    sqrt_log + th.randn((self.num_flows, 1,1,1,1)), requires_grad=True)
                plasts_ws[key_unit] = self._compute_plast(gates_ws[key_unit].data)

        return gates_ws, plasts_ws

    def _compute_plast(self, gate):

        log = th.log(1. / self.gate_init_val - 1.)
        return nn.Parameter(-log / gate).requires_grad_(False)

    def _compose_flow(self, flow, arch_name='resnet'):
        ''' Takes the original backbone network and transforms it in 
        the "flow" structure, where after each unit the gate transformation 
        is applied.
        ''' 
        if arch_name == 'resnet':
            d = dict(flow.named_children())

            # combines first four transformations in one layer
            # ("conv1" in DamNet notation for ResNet-50)
            d['layer0'] =[]
            for key in ['conv1', 'bn1', 'relu', 'maxpool']: 
                d['layer0'].append(d[key])
                del d[key]
            d['layer0'] = nn.Sequential(*d['layer0'])

            # Composes correctly ordered dictionary of flow layers
            ordered_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', \
                            'avgpool', 'fc']
            flow = th.nn.ModuleDict()
            for key in ordered_keys:
                flow[key] = d[key]

            if not self.debug: # "debug" mode serves for debugging ResNet50
                flow['fc'] = nn.Linear(2048, self.z_features) # otherwise it would give 1000 classes

            return flow

    def _compose_units(self, flows):
        ''' Takes flows prepared by _compose_units and 
        rearranges theirs modules to be run unit by unit.
        '''
        units = nn.ModuleDict()
        for key in flows[0].keys(): 
            unit = th.nn.ModuleList()
            for flow in flows:
                unit.append(flow[key])
            units[key] = unit

        return units

    def _aggregation(self, outs, gates_ws, plasts_ws):
        ''' Performs gate transformation, denoted in the paper 
        as the "aggregation operator". for the given 
        outputs of some computational unit and corresponding 
        gates weights (denoted as "gates_ws" in the signature).
        It is also suppossed that the size of "outs" is
        ( N_flows x B x C x H x W ).

        params:
        outs      - stack of outputs of some computational unit
        gates_ws  - "g" coefs, trainable parameters
        plasts_ws - "pi" coefs, plasticity parameters. 
                    Increasing over time accordint to the schedule.
        '''
        if outs.ndim != 5:
            # case of resnet "fc" layer, dim is decreased
            return outs.mean(dim=0)

        # correct 5-ndim layer case
        gates = th.sigmoid(gates_ws*plasts_ws)
        return (outs * gates).sum(dim=0)

    def forward(self, x, domain_idx):
        '''
        Computes output of the DamNet. 
        x - input vector, batch of images.
        domain_idx - some index (or key) of the domain, 
                     to be able to choose what gates to learn.
        '''
        key = 'domain_'+str(domain_idx) \
                if isinstance(domain_idx, int) else domain_idx

        assert (int(key[-1]) <= self.num_domains-1), \
            'domain index is incorrect, must be from {}, but {} was given.'.\
            format(list(range(self.num_domains)), int(key[-1]))

        for unit_name in self.units_names:
            unit = self.units[unit_name]

            if unit_name == 'fc':
                x = th.flatten(x,1)

            outs = []

            for flow_ind in range(len(unit)):
                out = unit[flow_ind](x)
                outs.append(out)
            outs = th.stack(outs)

            x = self._aggregation(outs, self.gates_ws[key+'_'+unit_name], 
                            self.plasts_ws[key+'_'+unit_name])
        return x


class GradientReversalFunction(th.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    
    The implementation is taken from 
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    Thanks to the author.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        # lambda_ = grad_output.new_tensor(lambda_)
        dx = -lambda_ * grad_output
        return dx, None

class GradientReversal(th.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    ''' Produces Domain Classifier network (F_d) that
    predicts a D-dimensional vector of domain probabilities d_hat
    given the feature vector Z from the DamNet.
    d_hat = F_d(R(Z)), where R - the gradient reversal pseudo-function
    from the DANN paper.
    '''
    def __init__(self, z_features=256, hidden_features=256,
                    num_domains=2, lambd=0.):

        super(DomainClassifier, self).__init__()

        self.num_domains = num_domains
        self.lambd = nn.Parameter(th.tensor(float(lambd))).requires_grad_(False)

        self.gradreverse = GradientReversal(self.lambd)

        fc1 = nn.Linear(z_features, hidden_features) 
        fc2 = nn.Linear(hidden_features, hidden_features)
        fc3 = nn.Linear(hidden_features, self.num_domains) 
        self.layers = nn.Sequential(*[
                fc1, nn.ReLU(inplace=True),
                fc2, nn.ReLU(inplace=True),
                fc3])

    def forward(self, z):
        z = self.gradreverse(z)
        z = self.layers(z)
        z = th.sigmoid(z)
        return z


class PoseRegressor(nn.Module):
    ''' Produces Pose given feature vector Z.
    '''
    def __init__(self, z_features=256, hidden_features=256, num_joints=15, joint_dim=2):
        super(PoseRegressor, self).__init__()

        self.num_joints = num_joints
        self.joint_dim = joint_dim
        fc1 = nn.Linear(z_features, hidden_features) 
        fc2 = nn.Linear(hidden_features, self.num_joints*self.joint_dim)
        self.layers = nn.Sequential(*[
                fc1, nn.ReLU(inplace=True),
                fc2])

    def forward(self, z):
        z = self.layers(z)
        z = z.view(-1, self.num_joints, self.joint_dim)
        z = nn.functional.hardtanh(z, min_val=0, max_val=224) # only for resnet50!!!
        return z


################################################


class WholeNet(nn.Module):
    ''' Combines the whole pipeline in one architecture with
    one joint "forward" function and schedulers for hyperparameters.
    '''
    def __init__(self, num_domains=2, pretrained=True, 
                        z_features=256, hidden_features=256, num_joints=15, joint_dim=2,
                        do_da=True, return_z=False, debug=False):
        super(WholeNet, self).__init__()

        self.num_domains = num_domains
        self.do_da = do_da # "do Domain Adaptation" flag 
        self.return_z = return_z # "return latent z" flag

        if debug:
            z_featues = 1000 # it is done violently to correspond to the ResNet50 evaluation

        # "fe" - feature extractor, "pr" - pose regressor, "dd" - domain discriminator
        self.fe = DamNet(self.num_domains, pretrained, z_features=z_features, debug=debug)
        self.pr = PoseRegressor(z_features, hidden_features, num_joints, joint_dim)

        if do_da:
            lambd_init = 0.
            self.dd = DomainClassifier(z_features, hidden_features, num_domains, lambd_init)

    def update_plasts(self, progress=None, base=1.05):
        ''' Does update of plasticities parameters given training progress.
        '''
        for key in self.fe.plasts_ws:
            self.fe.plasts_ws[key].data *= base

    def update_lambd(self, progress, gamma=10):
        ''' Does update of lambda parameter in GRL given the training progress.
        progress - float in the range [0,1]
        '''
        if self.do_da:
            self.dd.lambd.data = 2.*th.sigmoid(th.tensor(gamma*progress)) - 1.
            self.dd.gradreverse = GradientReversal(lambda_=self.dd.lambd)

    def forward(self, x, domain_idx, is_val=False):
        out = [None, None, None]

        z = self.fe(x, domain_idx)

        if domain_idx == 0 or is_val: 
            p = self.pr(z)
            out[0] = p
        
        if self.do_da:
            d = self.dd(z)
            out[1] = d

        if self.return_z:
            out[2] = z

        return out




























