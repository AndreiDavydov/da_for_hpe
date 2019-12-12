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
    def __init__(self, num_domains=2, pretrained=True, z_features=256, use_heatmaps=True, freeze=True, debug=False):
        super(DamNet, self).__init__()

        self.num_domains = num_domains
        self.num_flows = num_domains 
        self.z_features = z_features # dimension of latent vector Z
        # can vary from unit to unit, here it is fixed
        self.freeze = freeze
        self.use_heatmaps = use_heatmaps

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
 
        self.plast_init_val = 1e-3
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
        if self.num_domains == 1:
            return gates_ws, plasts_ws

        for domain_idx in range(self.num_domains):
            key = 'domain_'+str(domain_idx)
            for unit_name in self.units_names:
                key_unit = key+'_'+unit_name

                plast = self.plast_init_val * th.ones((self.num_flows, 1,1,1,1))
                gate = - th.log(th.tensor(self.num_flows - 1.)) / plast.data

                plasts_ws[key_unit] = nn.Parameter(plast.float()).requires_grad_(False)
                gates_ws[key_unit] = nn.Parameter(gate.float()).requires_grad_(True)

        return gates_ws, plasts_ws

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
            ordered_keys = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
            flow = th.nn.ModuleDict()
            for key in ordered_keys:
                flow[key] = d[key]

            if not self.debug: # "debug" mode serves for debugging ResNet50
                if self.use_heatmaps:
                    del flow['avgpool']
                    del flow['fc']
                    
                else: # use joint vectors
                    flow['fc'] = nn.Linear(2048, self.z_features) # otherwise it would give 1000 classes

            if self.freeze: # assume that ALL layers are freezed
                flow['layer0'].requires_grad_(False)
                flow['layer1'].requires_grad_(False)
                flow['layer2'].requires_grad_(False)
                flow['layer3'].requires_grad_(False)
                flow['layer4'].requires_grad_(False)

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

    def forward(self, x, domain_idx=0):
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

            if len(unit) == 1: # only one flow is trained
                x = unit[0](x)
            else:
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
                fc2, nn.ReLU(inplace=True)])

    def forward(self, z):
        z = self.layers(z)
        z = z.view(-1, self.num_joints, self.joint_dim)
        z = z*224
        z = nn.functional.hardtanh(z, min_val=0, max_val=224) # only for resnet50 with input size 224!!!
        return z


class DeconvHead(nn.Module):
    def __init__(self, in_channels=2048, num_layers=3, num_filters=256, kernel_size=4, conv_kernel_size=1, num_joints=15, depth_dim=1,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filters))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # self.features.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x

################################################


class WholeNet(nn.Module):
    ''' Combines the whole pipeline in one architecture with
    one joint "forward" function and schedulers for hyperparameters.
    '''
    def __init__(self, num_domains=2, pretrained=True, 
                        z_features=256, hidden_features=256, num_joints=15, joint_dim=2,
                        do_da=True, return_z=False, use_heatmaps=True, debug=False):
        super(WholeNet, self).__init__()

        self.num_domains = num_domains
        self.do_da = do_da # "do Domain Adaptation" flag 
        self.return_z = return_z # "return latent z" flag

        if debug:
            z_featues = 1000 # it is done violently to correspond to the ResNet50 evaluation

        # "fe" - feature extractor, "pr" - pose regressor, "dd" - domain discriminator
        self.fe = DamNet(self.num_domains, pretrained, z_features=z_features, use_heatmaps=use_heatmaps, debug=debug)
        if use_heatmaps:
            self.pr = DeconvHead()
        else:
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

    def forward(self, x, domain_idx=0, is_val=False):
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






























