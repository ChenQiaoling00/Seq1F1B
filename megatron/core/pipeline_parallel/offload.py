import contextlib
import torch


from megatron import get_args


class ActivationGroup:
    def __init__(self,tensors):

        # 这种排序策略的目的是首先根据张量的内存连续性进行排序，
        # 非连续内存的张量排在前面。
        # 其次，在内存连续性相同的情况下，根据张量的元素数量进行排序，元素数量多的张量排在后面。
        # 这样的排序策略通常用于优化计算，
        # 因为处理非连续内存的张量可能需要更多的注意力（如特殊处理或优化），
        # 而大的张量处理起来计算量更大，可能希望它们在某些处理流程中后进行处理。
        self.tensors=sorted(tensors,key=lambda t: (not t.x.is_contiguous(),-t.shape.numel()))
        self.offload_ratio = get_args().kaimm_offload_activation_ratio



class TensorWrap:
    def __init__(self,x):
        self.x=x
        self.shape=x.shape
        self.dtype=x.dtype
        self.device=x.device
        self.base=None


groups=dict()

@contextlib.contextmanager
def record(key):
    offload_ratio = get_args().kaimm_offload_activation_ratio
    if offload_ratio == 0:
        yield
        groups[key] = ActivationGroup([])
        return
    
    tensors=list()


    def pack_hook(x):
        tensor_wrap=TensorWrap(x)
        is_parameter

    
    def unpack_hook(tensor_pack):


    with torch.autograd.graph.saved_tensors_hooks(pack_hook,unpack_hook)

