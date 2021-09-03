import torch
import time
import math
import copy
import numpy as np
import numbers
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Module, Parameter
from fairseq.modules import LayerNorm


def init_weight(weight, init="standard", scale_fan=False, prune_rate=0.5, nonlinearity='relu', mode="fan_in"):

    if init == "signed_constant":

        fan = nn.init._calculate_correct_fan(weight, mode)
        if scale_fan:
            fan = fan * (1 - prune_rate)
        gain = nn.init.calculate_gain(nonlinearity)
        std = gain / math.sqrt(fan)
        weight.data = weight.data.sign() * std

    elif init == "unsigned_constant":

        fan = nn.init._calculate_correct_fan(weight, mode)
        if scale_fan:
            fan = fan * (1 - prune_rate)

        gain = nn.init.calculate_gain(nonlinearity)
        std = gain / math.sqrt(fan)
        weight.data = torch.ones_like(weight.data) * std

    elif init == "kaiming_normal":

        if scale_fan:
            fan = nn.init._calculate_correct_fan(weight, mode)
            fan = fan * (1 - prune_rate)
            gain = nn.init.calculate_gain(nonlinearity)
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                weight.data.normal_(0, std)
        else:
            nn.init.kaiming_normal_(
                weight, mode=mode, nonlinearity=nonlinearity
            )

    elif init == "kaiming_uniform":
        nn.init.kaiming_uniform_(
            weight, mode=mode, nonlinearity=nonlinearity
        )
    elif init == "xavier_normal":
        nn.init.xavier_normal_(weight)
    elif init == "xavier_constant":

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        weight.data = weight.data.sign() * std

    elif init == "standard":

        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    else:
        raise ValueError(f"{init} is not an initialization option!")

# this part is copied from movement pruning
class ThresholdBinarizer(autograd.Function):
    """
    Thresholdd binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > tau`
    where `tau` is a real value threshold.
    Implementation is inspired from:
        https://github.com/arunmallya/piggyback
        Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
        Arun Mallya, Dillon Davis, Svetlana Lazebnik
    """

    @staticmethod
    def forward(ctx, inputs, threshold, sigmoid, lower_bound=0.005):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The threshold value (in R).
            sigmoid (`bool`)
                If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
                In this case, `threshold` should be a value between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        nb_elems = inputs.numel()
        nb_min = int(lower_bound * nb_elems) + 1
        if sigmoid:
            mask = (torch.sigmoid(inputs) > threshold).type(inputs.type())
        else:
            mask = (inputs > threshold).type(inputs.type())
        if lower_bound > 0. and mask.sum() < nb_min:
            # We limit the pruning so that at least 0.5% (half a percent) of the weights are remaining
            k_threshold = inputs.flatten().kthvalue(max(nb_elems - nb_min, 1)).values
            mask = (inputs > k_threshold).type(inputs.type())
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None, None


class ThresholdBinarizer_V2(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, threshold):
        
        mask = (inputs.abs() > threshold).type(inputs.type()) * (inputs.sign()).type(inputs.type())
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        #
        ori_size = out.size()

        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # return out
        return flat_out.view(ori_size)

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

def get_mask(scores, prune_ratio, method_name, lower_bound=0.005):
    if method_name == "super_mask":
        # this one is based on magnitude
        subnet = GetSubnet.apply(scores.abs(), prune_ratio)
    elif method_name == "topk":
        # this one is based on value
        subnet = GetSubnet.apply(scores, prune_ratio)
    elif method_name == 'threshold':
        #TODO:  Note that for now, the prune ratio is not directly related to pruned ratio
        # We need to discuss that
        subnet = ThresholdBinarizer.apply(scores, prune_ratio, False, lower_bound)
    elif method_name == 'sigmoid_threshold':
        #TODO:  Note that for now, the prune ratio is not directly related to pruned ratio
        # We need to discuss that
        subnet = ThresholdBinarizer.apply(scores, prune_ratio, True, lower_bound)
    elif method_name == 'signed_threshold':
        subnet = ThresholdBinarizer_V2.apply(scores, prune_ratio)
    else:
        raise Exception(f"We did not support {method_name} yet!")
    return subnet

def get_dynamic_scale_mask(current_mask, method='row'):
    if method == "row":
        reduce_dim = 0
    elif method == "col":
        reduce_dim = 1

    mask_sum = current_mask.sum(reduce_dim, keepdim=True)
    mask_size = current_mask.size(0)
    mask_rate = torch.sqrt(mask_size / mask_sum)

    return mask_rate * current_mask

class MaskedLayerNorm(nn.LayerNorm):
    def __init__(self, dim):
        super(MaskedLayerNorm, self).__init__(dim, elementwise_affine=False)

class LayerNorm_no_learn(nn.LayerNorm):
    def __init__(self, dim):
        super(LayerNorm_no_learn, self).__init__(dim, elementwise_affine=True)

        self.weight.requires_grad = False
        self.bias.requires_grad = False

class self_normal_LayerNorm(nn.Module):
    # the weight and bias are learned
    def __init__(self, normalized_shape, eps=1e-5, bias_need=True, weight_need=True):
        super(self_normal_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.bias_need = bias_need
        self.weight_need = weight_need

        # if self.weight_need:
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        # else:
            # self.register_parameter('weight', None)
        
        # if self.bias_need:
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        # else:
            # self.register_parameter('bias', None)

        self.reset_parameters()
        if not self.weight_need:
            self.weight.requires_grad = False
        if not self.bias_need:
            self.bias.requires_grad = False

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'weight_need={weight_need}, bias_need={bias_need}, ' \
            ' '.format(**self.__dict__)



class self_mask_LayerNorm(nn.Module):
    # the weight and bias are shifted based on std and mean
    # I hard-code a lot of things for this function. 
    # TODO: need to make the function better
    def __init__(self, normalized_shape, eps=1e-5, bias_need=True, weight_need=True):
        super(self_mask_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.bias_need = bias_need
        self.weight_need = weight_need

        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.weight.requires_grad = False
        if self.weight_need:
            self.weight_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            nn.init.normal_(self.weight_scores, 0, 0.1)



        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.bias.requires_grad = False        
        if self.bias_need:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            nn.init.normal_(self.bias_scores, 0, 0.1)

        

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.ones_(self.weight)

        if self.bias_need:
            nn.init.normal_(self.bias, 0, 0.1) 
        else:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        
        if self.weight_need:
            subnet_weight = get_mask(self.weight_scores, 0.0, 'threshold', 0.)
            weight = self.weight * (subnet_weight + 1.) # either * 1 or * 2
        else:
            weight = self.weight

        if self.bias_need:
            subnet_bias = get_mask(self.bias_scores, 0.0, 'threshold', 0.)
            bias = self.bias * (2 * subnet_bias - 1.) # either * 1 or * -1
        else:
            bias = self.bias

        # print(self.weight)
        # print(subnet_weight)
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'weight_need={weight_need}, bias_need={bias_need}, ' \
            ' '.format(**self.__dict__)

class self_mask_LayerNorm_V2(nn.Module):
    # the weight and bias are shifted based on std and mean
    # I hard-code a lot of things for this function. 
    # TODO: need to make the function better
    def __init__(self, normalized_shape, eps=1e-5, bias_need=True, weight_need=True):
        super(self_mask_LayerNorm_V2, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.bias_need = bias_need
        self.weight_need = weight_need

        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.weight.requires_grad = False
        if self.weight_need:
            self.weight_scores = nn.Parameter(torch.Tensor(self.weight.size()))
            nn.init.normal_(self.weight_scores, 0, 0.1)



        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.bias.requires_grad = False        
        if self.bias_need:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            nn.init.normal_(self.bias_scores, 0, 0.1)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.ones_(self.weight)

        if self.bias_need:
            nn.init.normal_(self.bias, 0, 0.1) 
        else:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        
        if self.weight_need:
            subnet_weight = get_mask(self.weight_scores, 0.0, 'threshold', 0.)
            weight = self.weight * (subnet_weight + 1.) # either * 1 or * 2
        else:
            weight = self.weight

        if self.bias_need:
            subnet_bias = get_mask(self.bias_scores, 0.5, 'signed_threshold')
            bias = self.bias * subnet_bias # either * 1 or * -1 or * 0
        else:
            bias = self.bias

        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'weight_need={weight_need}, bias_need={bias_need}, ' \
            ' '.format(**self.__dict__)


# no weight and bias for this layer
def MaskedLayerNorm_select(dim, mask_layernorm_type, eps=1e-5):
    if mask_layernorm_type == 'masked_layernorm':
        return MaskedLayerNorm(dim)
    elif mask_layernorm_type == 'normal_layernorm':
        return LayerNorm(dim)
    elif mask_layernorm_type == 'normal_layernorm_no_learn': # this is designed for fine-tuning task!
        return LayerNorm_no_learn(dim)
    elif mask_layernorm_type == 'normal_bias_layernorm':
        return self_normal_LayerNorm(dim, bias_need=True, weight_need=False)
    elif mask_layernorm_type == 'normal_weight_layernorm':
        return self_normal_LayerNorm(dim, bias_need=False, weight_need=True)
    elif mask_layernorm_type == 'masked_bias_layernorm':
        return self_mask_LayerNorm(dim, bias_need=True, weight_need=False)
    elif mask_layernorm_type == 'masked_weight_layernorm':
        return self_mask_LayerNorm(dim, bias_need=False, weight_need=True)
    elif mask_layernorm_type == 'masked_affine_layernorm':
        return self_mask_LayerNorm(dim, bias_need=True, weight_need=True)
    elif mask_layernorm_type == 'masked_bias_layernorm_v2':
        return self_mask_LayerNorm_V2(dim, bias_need=True, weight_need=False)
    elif mask_layernorm_type == 'masked_affine_layernorm_v2':
        return self_mask_LayerNorm_V2(dim, bias_need=True, weight_need=True)
    else:
        raise Exception("We do not support this {mask_layernorm_type} yet")

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, prune_ratio=0.5, prune_method="super_mask", mask_init="standard", mask_constant=0.5,
                    init="standard", nonlinearity="relu", scale_fan=False, dynamic_scaling=False, group=1, group_type='row'):
        super(MaskedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.need_bias = bias
        self.prune_ratio = prune_ratio
        self.prune_method = prune_method
        self.mask_init = mask_init
        self.mask_constant = mask_constant
        self.init = init
        self.nonlinearity = nonlinearity
        self.scale_fan = scale_fan
        self.dynamic_scaling = dynamic_scaling
        if self.dynamic_scaling:
            assert self.scale_fan == False

        self.weight = Parameter(torch.Tensor(out_features, in_features))

        if not bias:
            self.register_parameter('bias', None)
        else:
            self.bias = Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)

        # initialize the scores
        weight_size = self.weight.size()
        self.scores_group = group
        self.scores_group_type = group_type
        if group != 1:
            if 'row' in group_type:
                weight_size = torch.Size([weight_size[0]//group, weight_size[1]])
            elif 'col' in group_type:
                weight_size = torch.Size([weight_size[0], weight_size[1]//group])

        
        self.scores = nn.Parameter(torch.Tensor(weight_size))
        if 1. > self.prune_ratio > 0.:
            if mask_init == "standard":
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            elif mask_init == "constant":
                nn.init.constant_(self.scores, val=(mask_constant*1.0) )
                # raise Exception(f"We did not support {mask_init} yet!")
            elif mask_init == "constant_noisy":
                nn.init.constant_(self.scores, val=(mask_constant*1.0) )
                self.scores.data = self.scores.data + torch.randn_like(self.scores.data) * 0.0001
            else:
                raise Exception(f"We did not support {mask_init} yet!")
        else:
            nn.init.constant_(self.scores, val=1.0 )
            self.scores.requires_grad = False


        # NOTE: initialize the weights like this.
        if nonlinearity == 'gelu':
            nonlinearity = 'relu'
        init_weight(self.weight, init, scale_fan, prune_ratio, nonlinearity)


        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        # Used for fast inference
        self.training_or_inference = "train" # either train or eval
        self.current_mask = None


    def change_mode(self, mode):
        assert mode == "train" or mode == "eval"
        self.training_or_inference = mode

    def get_mask(self):
        if self.scores_group != 1:
            if 'row' in self.scores_group_type:
                scores = self.scores.repeat(self.scores_group, 1)
            elif 'col' in self.scores_group_type:
                scores = self.scores.repeat(1, self.scores_group)
        else:
            scores = self.scores
        if self.training_or_inference == "train":
            # reset mask to be none
            self.current_mask = None
            subnet = get_mask(scores, self.prune_ratio, self.prune_method)
            if self.dynamic_scaling:
                subnet = get_dynamic_scale_mask(subnet)
            return subnet
        elif self.training_or_inference == "eval":
            if self.current_mask is None:
                self.current_mask = get_mask(scores, self.prune_ratio, self.prune_method)
                if self.dynamic_scaling:
                    self.current_mask = get_dynamic_scale_mask(self.current_mask)
            return self.current_mask

    def forward(self, x):
        if self.scores_group != 1:
            if 'row' in self.scores_group_type:
                scores = self.scores.repeat(self.scores_group, 1)
            elif 'col' in self.scores_group_type:
                scores = self.scores.repeat(1, self.scores_group)
        else:
            scores = self.scores
        if self.training_or_inference == "train":
            # reset mask to be none
            self.current_mask = None
            subnet = get_mask(scores, self.prune_ratio, self.prune_method)
            if self.dynamic_scaling:
                subnet = get_dynamic_scale_mask(subnet)
            w = self.weight * subnet
        elif self.training_or_inference == "eval":
            if self.current_mask is None:
                self.current_mask = get_mask(scores, self.prune_ratio, self.prune_method)
                if self.dynamic_scaling:
                    self.current_mask = get_dynamic_scale_mask(self.current_mask)

            w = self.weight * self.current_mask

        if self.prune_ratio == 1.0:
            # print("pass the test!")
            assert torch.all(torch.eq(w, self.weight))

        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, need_bias={self.need_bias}," \
               f"prune_ratio={self.prune_ratio}, prune_method={self.prune_method}, mask_init={self.mask_init}," \
               f"mask_constant={self.mask_constant}, init={self.init}, nonlinearity={self.nonlinearity}," \
               f"scale_fan={self.scale_fan}"


class MaskedEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None, prune_ratio=0.5, prune_method="super_mask",
                 mask_init="standard", mask_constant=0.5,
                 init="standard", scale_fan=False, init_embedding_seperate=False, dynamic_scaling=False):
        super(MaskedEmbedding, self).__init__()

        self.prune_ratio = prune_ratio
        self.prune_method = prune_method
        self.mask_init = mask_init
        self.mask_constant = mask_constant
        self.init = init
        self.scale_fan = scale_fan
        self.dynamic_scaling = dynamic_scaling
        if self.dynamic_scaling:
            assert scale_fan == False

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))

            # NOTE: initialize the weights like this.
            if not init_embedding_seperate:
                init_weight(self.weight, init, scale_fan, prune_ratio, 'relu')
            # TODO: is padding token trainable or not? If it is trainable, we do not need to
            # re-init it.
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

        self.sparse = sparse

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if 1. > self.prune_ratio > 0.:
            if mask_init == "standard":
                nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            elif mask_init == "constant":
                nn.init.constant_(self.scores, val=(mask_constant*1.0) )
                # raise Exception(f"We did not support {mask_init} yet!")
            elif mask_init == "constant_noisy":
                nn.init.constant_(self.scores, val=(mask_constant*1.0) )
                self.scores.data = self.scores.data + torch.randn_like(self.scores.data) * 0.0001
            else:
                raise Exception(f"We did not support {mask_init} yet!")
        else:
            nn.init.constant_(self.scores, val=1.0 )
            self.scores.requires_grad = False


        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

        # Used for fast inference
        self.training_or_inference = "train" # either train or eval
        self.current_mask = None

    def reset_parameters(self):
        # init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def change_mode(self, mode):
        assert mode == "train" or mode == "eval"
        self.training_or_inference = mode

    def forward(self, input):
        if self.training_or_inference == "train":
            # reset mask to be none
            self.current_mask = None
            subnet = get_mask(self.scores, self.prune_ratio, self.prune_method)
            if self.dynamic_scaling:
                subnet = get_dynamic_scale_mask(subnet)
            w = self.weight * subnet
        elif self.training_or_inference == "eval":
            if self.current_mask is None:
                self.current_mask = get_mask(self.scores, self.prune_ratio, self.prune_method)
                if self.dynamic_scaling:
                    self.current_mask = get_dynamic_scale_mask(self.current_mask)
            w = self.weight * self.current_mask

        return F.embedding(
            input, w, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)



    def extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}," \
               f"prune_ratio={self.prune_ratio}, prune_method={self.prune_method}, mask_init={self.mask_init}," \
               f"mask_constant={self.mask_constant}, init={self.init}, " \
               f"scale_fan={self.scale_fan}"
