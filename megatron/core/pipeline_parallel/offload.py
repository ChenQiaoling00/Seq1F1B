import contextlib
import torch
import math

from megatron import get_args
import torch.distributed as dist
from megatron.core import parallel_state
from megatron import print_rank_0

_CPU_BUFFER_POOL=list()
_GPU_BUFFER_POOL=dict()
_MEMCPY_STREAM = {}
_PARTIAL_TENSOR=list()

def get_memcpy_stream(key):
    global _MEMCPY_STREAM  # 明确指出这里使用的是全局变量
    if key not in _MEMCPY_STREAM:
        _MEMCPY_STREAM[key] = torch.cuda.Stream()  # 为每个独特的key创建一个新的CUDA流
    return _MEMCPY_STREAM[key]

def get_persistent_gpu_buffer(key,size):
    if key not in _GPU_BUFFER_POOL or _GPU_BUFFER_POOL[key].numel() < size:
        _GPU_BUFFER_POOL[key]=None
        _GPU_BUFFER_POOL[key]=torch.empty(size,dtype=torch.uint8,device='cuda')
        _GPU_BUFFER_POOL[key].ref_cnt = 0
    return _GPU_BUFFER_POOL[key][:size]
    
def recyle_cpu_buffer(buffer):
    _CPU_BUFFER_POOL.append(buffer._base)

def get_cpu_buffer(size):
    best_i = -1
    
    for i,buffer in enumerate(_CPU_BUFFER_POOL):
        if buffer.numel()>= size:
            if best_i == -1 or buffer.numel() < _CPU_BUFFER_POOL[best_i].numel():
                best_i=i
    if best_i != -1:
        return _CPU_BUFFER_POOL.pop(best_i)[:size]
    if _CPU_BUFFER_POOL:
        _CPU_BUFFER_POOL.pop()
    import wrap_gemm_cuda
    buffer = wrap_gemm_cuda.wrap_cuda_malloc_host(size)
    return buffer[:size]
    
def copy2d_(dst,src):
    assert dst.dtype == src.dtype, "mismatch dtype"
    if not dst.is_contiguous():
        raise NotImplementedError(f'unsupported dst shape {dst.shape} stride {dst.stride()}')
    
    shape = src.shape
    stride = src.stride()
    
    if stride[-1] ==1 and all(stride[i] == shape[i+1]*stride[i+1] for i in range(len(shape)-2)):
        import wrap_gemm_cuda
        dw=src.dtype.itemsize
        cudaMemcpyDefault = 4
        # 此函数参数包括目标内存地址、每行数据大小、源内存地址、源数据行跨度、复制的宽度和高度，以及使用的 CUDA stream。
        wrap_gemm_cuda.wrap_cuda_memcpy_2d_async(dst.data_ptr(), shape[-1] * dw, src.data_ptr(), stride[-2] * dw,
                                            shape[-1] * dw, shape[:-1].numel(), cudaMemcpyDefault,
                                            torch.cuda.current_stream().cuda_stream)
    else:
        raise NotImplementedError(f'unsupported dst shape {dst.shape} stride {dst.stride()}')
    

class ActivationGroup:
    
    def __init__(self,tensors):

        # 这种排序策略的目的是首先根据张量的内存连续性进行排序，
        # 非连续内存的张量排在前面。
        # 其次，在内存连续性相同的情况下，根据张量的元素数量进行排序，元素数量多的张量排在后面。
        # 这样的排序策略通常用于优化计算，
        # 因为处理非连续内存的张量可能需要更多的注意力（如特殊处理或优化），
        # 而大的张量处理起来计算量更大，可能希望它们在某些处理流程中后进行处理。
        # 首先按tensor是否是非连续的来排序，非连续的tensor优先。
        # 如果两个tensor的连续性相同，则按元素数量多少来排序，元素数量多的tensor优先。
        
        self.tensors = sorted(tensors, key=lambda t: (not t.x.is_contiguous(), -t.shape.numel()))
        self.offload_ratio = get_args().kaimm_offload_activation_ratio
        if self.offload_ratio > .5:
            self.tensors = self.tensors[::-1]  # workaround: avoid offloading half FC1 output

    def offload_prologue(self,key,use_bucket):
        if not self.tensors:
            return None,None,None
        self.map=list()
        top=0
        
        for i, tensor in enumerate(self.tensors):
            duplicate_flag=False
            if tensor.x.is_contiguous():
                for j, pre_tensor in enumerate(self.tensors[:i]):
                    if tensor.x.data_ptr() == pre_tensor.x.data_ptr() and pre_tensor.x.is_contiguous() and tensor.shape.numel() == pre_tensor.shape.numel() and tensor.device == pre_tensor.device:
                        begin_idx,end_idx,_0=self.map[j]
                        duplicate_flag=True
                        self.map.append((begin_idx,end_idx,duplicate_flag))
                        break
                    
            if not duplicate_flag:
                n=tensor.shape.numel()*tensor.dtype.itemsize
                self.map.append((top,top+n,duplicate_flag))
                top+=n
        
        MiB=2**20
        offload_size=(int(math.ceil(top * self.offload_ratio))+MiB - 1) // MiB * MiB # 来确保任何计算结果都会向上取整到最近的 MiB 的倍数
        def cal_zero_perc(x):
            # 计算 tensor 中 0 的个数
            num_zeros = (x == 0).sum().item()
            num_elements_le_0_05 = (x <= 0.005).sum().item()
            # 计算总元素数
            total_elements = x.numel()

            # 计算 0 的百分比
            num_elements_le_0_05_percentage = (num_elements_le_0_05 / total_elements) * 100
            
            return num_elements_le_0_05_percentage,total_elements
        
        def tensor_memory_size(tensor):
            """
            计算一个 tensor 占用的内存大小，并根据大小自动选择单位。

            参数:
            tensor (torch.Tensor): 需要计算内存的 tensor。

            返回:
            str: tensor 占用的内存大小，自动选择合适的单位（B, MiB, GB）。
            """
            total_elements = tensor.numel()  # 获取 tensor 的总元素数量
            element_size = tensor.element_size()  # 获取每个元素的大小（字节）
            total_size_bytes = total_elements * element_size  # 总字节大小
            
            if total_size_bytes < 1024:
                return f"{total_size_bytes} B"
            elif total_size_bytes < 1024 ** 2:
                return f"{total_size_bytes / 1024:.2f} KiB"
            elif total_size_bytes < 1024 ** 3:
                return f"{total_size_bytes / (1024 ** 2):.2f} MiB"
            else:
                return f"{total_size_bytes / (1024 ** 3):.2f} GB" 
        
        if use_bucket:
            buffer=get_persistent_gpu_buffer("offload",offload_size)
        else:
            buffer=None # not use bucket
        
        buffer_fake=get_persistent_gpu_buffer("offload_fake",offload_size)
        copy_tasks=[] #restore those task need to be copied async
        partially_offloaded_base=set() #restore those partially offloaded tensor's base\
        partial_tensor={}
        
        for tensor ,(begin_idx,end_idx,duplicate_flag) in zip(self.tensors,self.map):
            assert tensor.x.device.type=="cuda"
   
            print_rank_0(f'origin:::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{tensor.x.shape},tensor memory allocated:{tensor_memory_size(tensor.x)}')
            # zero_percentage,total_elements = cal_zero_perc(tensor.x)
    
            pp_rank_id=parallel_state.get_pipeline_model_parallel_rank()
            key_id = key
            if pp_rank_id not in partial_tensor:
                partial_tensor[pp_rank_id] = {}

            # 如果 key_id 不存在，先初始化为一个空列表
            if key_id not in partial_tensor[pp_rank_id]:
                partial_tensor[pp_rank_id][key_id] = []
            # partial_tensor[pp_rank_id][key_id].append((tensor.x.shape,zero_percentage))
                
            # partial_tensor.append((parallel_state.get_pipeline_model_parallel_rank(),key,tensor.x.shape,zero_percentage))
        
            # print_rank_0(f'zero_percentage tensor::::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},zero:{zero_percentage},total_elements:{total_elements}')
            
            if end_idx <= offload_size:
        
                if not duplicate_flag:
                    if tensor.x._base is not None:
                        partially_offloaded_base.add(tensor.x._base)
                        if use_bucket:
                            buffer[begin_idx:end_idx].view(tensor.dtype).view(tensor.shape).copy_(tensor.x)
                        else:
                            copy_tasks.append((begin_idx,end_idx,tensor.x))
                    else:
                        print_rank_0(f'tensor x base is none, rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{tensor.x.shape}')
                else:
                    print_rank_0(f'duplicate_flag, rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{tensor.x.shape}')
                tensor.x=None
            elif begin_idx < offload_size:
       
                if not duplicate_flag:
                    if tensor.x._base is not None:
                        partially_offloaded_base.add(tensor.x._base)
                    linear_data = tensor.x.contiguous().view(-1).view(torch.uint8)
                    if use_bucket:
                        buffer[begin_idx:].view(tensor.dtype).view(tensor.shape).copy_(linear_data[:offload_size-begin_idx])
                    else:
                        copy_tasks.append((begin_idx,offload_size,linear_data[:offload_size-begin_idx]))
                    self.remained_not_offloaded=linear_data[offload_size-begin_idx:].clone()
                tensor.x=None
            elif tensor.x._base in partially_offloaded_base:
                if duplicate_flag:
                    raise NotImplementedError("does not support partially offload duplicate tensors")
                tensor.x=tensor.x.clone()
        self.buffer_cpu=get_cpu_buffer(offload_size)
        stream=get_memcpy_stream("offload")

        
        # 这种等待是必要的，特别是当你需要确保在开始新的内存复制之前，所有依赖于当前 GPU 结果的计算都已完成。这样可以防止数据竞争和潜在的错误。
        stream.wait_stream(torch.cuda.current_stream()) 
        # _PARTIAL_TENSOR.append(partial_tensor)
        # print_rank_0(f'partial_tensor::::{_PARTIAL_TENSOR}')
        # if len(_PARTIAL_TENSOR) == 400:
        #     with open(f'list32k005_{pp_rank_id}.txt', 'w') as f:
        #         for item in _PARTIAL_TENSOR:
        #             f.write(f"{item}\n")
                    
        with torch.cuda.stream(stream):
            if use_bucket:
                print(f'offload::::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},offload_size:{tensor_memory_size(buffer)}')
                self.buffer_cpu.copy_(buffer_fake,non_blocking=True)
            else:
                for begin_idx, end_idx, x in copy_tasks:
                    print(f'offload::::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{x.shape},tensor memory allocated:{tensor_memory_size(x)}')
                    if x.is_contiguous():
                        self.buffer_cpu[begin_idx:end_idx].view(x.dtype).view(x.shape).copy_(x, non_blocking=True)
                    else:
                        copy2d_(self.buffer_cpu[begin_idx:end_idx].view(x.dtype).view(x.shape), x)
        # torch.cuda.synchronize()

        return stream,buffer,copy_tasks
        
    def offload_epilogue(self,stream,buffer,copy_tasks):
        if not self.tensors:
            return
        # torch.cuda.current_stream().wait_stream(stream)
        
    def onload_prologue(self,key):
        
        if not self.tensors:
            return None,None
        stream_key='onload'
        buffer_key=stream_key
        stream = get_memcpy_stream(stream_key)
        buffer = get_persistent_gpu_buffer(buffer_key,self.buffer_cpu.numel())
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            buffer.copy_(self.buffer_cpu,non_blocking=True)
        return stream,buffer,key
        
    def onload_epilogue(self,stream,buffer,key):
        if not self.tensors:
            return
        # torch.cuda.current_stream().wait_stream(stream)
    
        recyle_cpu_buffer(self.buffer_cpu)
        self.buffer_cpu=None
        offload_size=buffer.numel()
        duplicate_tensor=dict()
        for tensor,(begin_idx,end_idx,duplicate_flag) in zip(self.tensors,self.map):
            if end_idx <= offload_size:
                tensor.x=buffer[begin_idx:end_idx].view(tensor.dtype).view(tensor.shape)
                tensor.x=tensor.x.clone()
                print(f'onload::::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{tensor.x.shape}',flush=True)
            elif begin_idx < offload_size:
                if not duplicate_flag:
                    tensor.x = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                    linear_data = tensor.x.view(-1).view(buffer.dtype)
                    linear_data[:offload_size - begin_idx].copy_(buffer[begin_idx:])
                    linear_data[offload_size - begin_idx:].copy_(self.remained_not_offloaded)
                    self.remained_not_offloaded = None
                    duplicate_tensor[begin_idx, end_idx] = linear_data
                else:
                    tensor.x = duplicate_tensor[begin_idx, end_idx].view(tensor.dtype).view(tensor.shape)
                # print(f'2222222222222::::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{tensor.x.shape}',flush=True)
        del self.tensors
        del self.map
        

class ForwardLeftBackwardRightFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,left,right):
        return left
    
    @staticmethod
    def backward(ctx,grad_output):
        return None,grad_output

class ForwardEmptyBackwardIdentityFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,x):
        return torch.empty((),dtype=x.dtype,device=x.device).expand_as(x)

    @staticmethod
    def backward(ctx,grad):
        return grad

class TensorWrap:
    def __init__(self,x):
        self.x=x
        self.shape=x.shape
        self.dtype=x.dtype
        self.device=x.device
        self.base=None
        
class TensorPack:
    def __init__(self,tensor_wrap):
        self.tensor_wrap = tensor_wrap
        
    def get(self):
        return self.tensor_wrap.x   
    
    def __del__(self):
        self.tensor_wrap.x = None
        if self.tensor_wrap.base is not None:
            self.tensor_wrap.base.ref_cnt -=1
    
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
        
        is_parameter=isinstance(x,torch.nn.Parameter)
        is_too_small=x.numel()*x.element_size() < 1024*1024
        is_rope_freqs=x.dim()==4 and x.shape[1] == 1 and x.shape[2] ==1
        print(f'ALLLLLLLLL tensor::::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{x.shape}',flush=True)

        if not is_parameter and not is_too_small and not is_rope_freqs:
            tensors.append(tensor_wrap)
        # else:
        #     print_rank_0(f'not pack tensor::::::rank:{parallel_state.get_pipeline_model_parallel_rank()},key:{key},x:{x.shape}')
        return TensorPack(tensor_wrap)

    def unpack_hook(tensor_pack):
        try:
            x= tensor_pack.get()
            print(f'TTTTTTTTTTTTTTTTTTTTTTTrank:{dist.get_rank()}, key:{key},x:{x.shape}',flush=True)
        except:
            # import pdb;pdb.set_trace()
            print(f'FFFFFFFFFFFFFFFFFFFFFFrank:{dist.get_rank()}, key:{key}, x:{tensor_pack}',flush=True)
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack_hook,unpack_hook):
        yield
    
    groups[key]=ActivationGroup(tensors)
    
@contextlib.contextmanager    
def offload_async(key):
    group=groups[key]
    # before_mem = torch.cuda.memory_allocated()
    # print_rank_0(f"Memory before operation: {before_mem} bytes")
    args=group.offload_prologue(key,use_bucket=True)
    # after_mem = torch.cuda.memory_allocated()
    # print_rank_0(f"Memory after operation: {after_mem} bytes")
    # print_rank_0(f"Memory allocated by operation: {(before_mem-after_mem)/before_mem * 100} \%")
    yield
    group.offload_epilogue(*args)
 
@contextlib.contextmanager   
def onload_async(key):
    group=groups[key]
    args=group.onload_prologue(key)
    yield
    group.onload_epilogue(*args)

def get_forward_tensor_and_backward_handle(x):
    backward_handle = torch.empty((), dtype=x.dtype, device=x.device).expand_as(x)
    backward_handle.requires_grad_(x.requires_grad)
    x.requires_grad_(False)
    x = ForwardLeftBackwardRightFunction.apply(x, backward_handle)
    return x, backward_handle
    

def forward_empty_backward_indentity(x):
    
    return ForwardEmptyBackwardIdentityFunction.apply(x)
