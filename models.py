import torch as th
import torch.nn as nn
from resnet.resnet import resnet50

PATH_RESNET50_PRETRAINED = './resnet/resnet50.pth'

class DamNet(nn.Module):
    ''' Produces DAMNet (based on the DAMN paper)
    with the ResNet-50 as a backbone.
    As written in the original paper, this model
    "encodes sample X from domain d into a feature 
    vector Z = f(X, d)".

    # named_children of ResNet50: 
    ## layer0: "conv1", "bn1", "relu", "maxpool" 
    ## "layer1", "layer2", "layer3", "layer4"
    ## "avgpool", "fc"
    '''
    def __init__(self, num_domains=2, pretrained=True, seed=0, debug=False):
        super(DamNet, self).__init__()

        th.manual_seed(seed)
        self.num_domains = num_domains
        self.num_flows = num_domains 
        # can vary from unit to unit, here it is fixed

        ########### for checking correct ResNet50 classification quality.
        self.debug = debug
        ###########

        flows = []
        for _ in range(self.num_flows): 
            flow = resnet50(pretrained=pretrained)
            if pretrained:
                flow.load_state_dict(th.load(PATH_RESNET50_PRETRAINED)) 
            flow = self._compose_flow(flow)
            flows.append(flow)                
                                                                        
        self.units_names = flow.keys()                           
        self.units = self._compose_units(flows)   

        self.gates_ws, self.plasts_ws = self._init_gates()

    def _init_gates(self):
        gates_ws, plasts_ws = nn.ParameterDict(), nn.ParameterDict() 
        # HOW TO DO IT CORRECTLY, ParamDict of ParamDicts?
        for domain_ind in range(self.num_domains):
            key = 'domain_'+str(domain_ind)
            for unit_name in self.units_names:
                # Not clear how to init plasticities correctly. 
                # So far all gates are equal at the first epoch.
                key_unit = key+'_'+unit_name
                gates_ws[key_unit] = nn.Parameter(
                    th.randn((self.num_flows, 1,1,1,1)), requires_grad=True)
                plasts_ws[key_unit] = nn.Parameter(-th.log(
                    th.tensor(1/(1/self.num_flows) - 1))/ \
                        (gates_ws[key_unit].detach()))

        return gates_ws, plasts_ws

    def _compose_flow(self, flow, arch_name='resnet'):
        ''' Takes the original backbone network and transforms it in 
        the "flow" structure, where after each unit the gate transformation 
        is applied.
        ''' 
        if arch_name == 'resnet':
            d = dict(flow.named_children())

            # combines first four transformations in one layer
            # ("conv1" in DAMNet notation for ResNet-50)
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

            # if not self.debug:
            #     del flow['avgpool']
            #     del flow['fc']

            if not self.debug:
                flow['fc'] = nn.Linear(2048, 256) # instead it would give 1000 classes

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

    def forward(self, x, domain_ind):
        '''
        Computes output of the DAMNet. 
        x - input vector, batch of images.
        domain_ind - some index (or key) of the domain, 
                     to be able to choose what gates to learn.
        '''
        key = 'domain_'+str(domain_ind) \
                if isinstance(domain_ind, int) else domain_ind

        assert (int(key[-1]) <= self.num_domains-1), \
            'domain index is incorrect, must be from {}, but {} was given.'.\
            format(list(range(self.num_domains)), int(key[-1]))

        for unit_name in self.units_names:
            unit = self.units[unit_name]

            if unit_name == 'fc':
                x = th.flatten(x,1)

            outs = []
            # print(unit_name)
            for flow_ind in range(len(unit)):
                out = unit[flow_ind](x)
                outs.append(out)
            outs = th.stack(outs)

            x = self._aggregation(outs, self.gates_ws[key+'_'+unit_name], 
                            self.plasts_ws[key+'_'+unit_name])
            # print(x.shape)
        return x


class GradientReversalFunction(th.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    
    The implementation is taken from 
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    Thanks to author.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        lambda_ = grad_output.new_tensor(lambda_)
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
    given the feature vector Z from the DAMNet.
    d_hat = F_d(R(Z)), where R - the gradient reversal pseudo-function
    from the DANN paper.
    '''
    def __init__(self, in_features=256, hidden_features=256,
                    num_domains=2, lambd=1,seed=0):
        super(DomainClassifier, self).__init__()
        th.manual_seed(0)
        self.num_domains = num_domains

        self.gradreverse = GradientReversal(lambd)

        fc1 = nn.Linear(in_features, hidden_features) 
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
    def __init__(self, in_features=256, hidden_features=256, out_features=15*2):
        super(PoseRegressor, self).__init__()

        fc1 = nn.Linear(in_features, hidden_features) 
        fc2 = nn.Linear(hidden_features, out_features)
        self.layers = nn.Sequential(*[
                fc1, nn.ReLU(inplace=True),
                fc2])

    def forward(self, z):
        z = self.layers(z)
        z = z.view(-1, 15,2)
        return z




























