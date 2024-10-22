import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from itertools import combinations_with_replacement, product

def round_down(x, m):
    return int(x / m) * m

class SeqTFlops:
    def __init__(self, hidden_size, num_heads, ffn_ratio, vocab_size, num_layers,num_pp_stages,num_seq_partitions):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        self.ffn_ratio = ffn_ratio
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_pp_stages = num_pp_stages
        self.num_seq_partitions = num_seq_partitions

    def get_ffn_tflops(self, seq_len):
        # FFN consists of two linear layers with activation in between
        ffn_hidden_size = self.hidden_size * self.ffn_ratio  # Typically 4 * h

        # First linear layer: [seq_len, h] x [h, 4h] -> [seq_len, 4h]
        first_layer_flops = 2 * seq_len * self.hidden_size * ffn_hidden_size

        # Activation function (e.g., GELU): [seq_len, 4h]
        activation_flops = seq_len * ffn_hidden_size  # Approximate as 1 FLOP per element

        # Second linear layer: [seq_len, 4h] x [4h, h] -> [seq_len, h]
        second_layer_flops = 2 * seq_len * ffn_hidden_size * self.hidden_size

        # Total FFN FLOPs for forward pass
        total_ffn_flops = first_layer_flops + activation_flops + second_layer_flops

        return total_ffn_flops / 1e12  # Convert to TFLOPs

    def get_emb_tflops(self, seqlen):
        embed_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        # 输出投影计算
        emb_proj_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        return embed_tflops / 1e12, emb_proj_tflops / 1e12  # Convert to TFLOPs
    
    def get_origin_attention_tflops(self, seqlen, causal=False):
        scale = 0.5 if causal else 1
                # Linear projections for Q, K, V: [seq_len, h] x [h, h] -> [seq_len, h]
        qkv_flops = 3 * 2 * seqlen * self.hidden_size * self.hidden_size

        # Attention scores: Q x K^T -> [seq_len, prefix_len]
        attn_scores_flops = 2 * seqlen * seqlen * self.hidden_size *scale
        attn_softmax_tflops = 3 * seqlen * seqlen * self.num_heads + 2 * seqlen * seqlen * self.num_heads * self.dim_head
        attn_softmax_tflops *= scale
        attn_o_proj_tflops = 2 * seqlen * self.hidden_size * (self.dim_head * self.num_heads)
        # Total Attention FLOPs for forward pass
        total_attn_flops = qkv_flops + attn_scores_flops + attn_softmax_tflops + attn_o_proj_tflops
        return total_attn_flops / 1e12  # Convert to TFLOPs
    
    def get_prefix_attention_tflops(self, seq_len, prefix_len,causal=False):
        attn_part = seq_len * prefix_len * (self.dim_head * 4 + 3 ) \
            * self.num_heads + seq_len * 8 * self.hidden_size \
            * self.num_heads * self.dim_head - seq_len ** 2 * (4 * self.dim_head + 3) \
            * self.num_heads / 2
        
        return attn_part / 1e12
    
    def get_prefix_layer_tflops(self, seq_len, prefix_len, causal=False):
        attn_tflops = self.get_prefix_attention_tflops(seq_len, prefix_len, causal)
        ffn_tflops = self.get_ffn_tflops(seq_len)
        return attn_tflops + ffn_tflops
    def get_origin_layer_tflops(self, seq_len, causal=False):
        attn_tflops = self.get_origin_attention_tflops(seq_len, causal)
        ffn_tflops = self.get_ffn_tflops(seq_len)
        return attn_tflops + ffn_tflops
    def get_stage0_prefix_tflops(self, seq_len,prefix_len):
        layer_tflops=self.get_prefix_layer_tflops(seq_len,prefix_len)*self.num_layers/self.num_pp_stages
        embed_tflops, _ = self.get_emb_tflops(seq_len)
        return layer_tflops + embed_tflops

    

class PipelineParallelism:
    def __init__(self, config, total_layers, total_seqlen, tflops_capacity):
        self.config = config
        self.total_layers = total_layers
        self.total_seqlen = total_seqlen
        self.tflops_capacity = tflops_capacity  # Already in TFLOPs
        # self.solver = Solver(total_seqlen, config)

    def partition_layers(self, num_pp_stages):
        # Total components: input embedding + transformer layers + final layernorm + output layer
        total_units = 1 + self.total_layers + 1 + 1
        
        # Initialize partitions
        partitions = [0] * num_pp_stages
        
        # Distribute transformer layers
        layers_per_stage = self.total_layers // num_pp_stages
        remainder = self.total_layers % num_pp_stages
        for i in range(num_pp_stages):
            partitions[i] += layers_per_stage
            if i < remainder:
                partitions[i] += 1
        
        # Add input embedding to the first stage
        partitions[0] += 1
        
        # Add final layernorm and output layer to the last stage
        partitions[-1] += 2
        
        return partitions
    
    def calculate_stage_tflops(self, stage_idx, num_stages, num_layers, seq_len, prefix_len):
        # Base TFLOPs for transformer layers in this stage
        stage_tflops = self.config.get_prefix_layer_tflops(seq_len, prefix_len) * num_layers 

        # First stage: Add embedding layer
        if stage_idx == 0:
            embed_tflops, _ = self.config.get_emb_tflops(seq_len)
            stage_tflops += embed_tflops

        # Last stage: Add final layernorm and output projection
        if stage_idx == num_stages - 1:
            # Add TFLOPs for final layernorm (approximation)
            final_ln_tflops = seq_len * self.config.hidden_size / 1e12
            stage_tflops += final_ln_tflops

            # Add TFLOPs for output projection
            _, emb_proj_tflops = self.config.get_emb_tflops(seq_len)
            stage_tflops += emb_proj_tflops

        return stage_tflops
    
    def calculate_pipeline_latency(self, layer_partitions, seq_partitions):
        num_stages = len(layer_partitions)
        num_subsequences = len(seq_partitions)
        
        stage_start_times = [0] * num_stages
        schedule = []
        
        for subseq_idx in range(num_subsequences):
            for stage_idx in range(num_stages):
                if stage_idx == 0:
                    start_time = stage_start_times[stage_idx]
                else:
                    start_time = max(stage_start_times[stage_idx], stage_start_times[stage_idx-1])
                
                seq_len = seq_partitions[subseq_idx]
                num_layers = layer_partitions[stage_idx]
                prefix = sum(seq_partitions[:subseq_idx+1])
                
                # Use calculate_stage_tflops to get accurate TFLOPs for this stage
                tflops = self.calculate_stage_tflops(stage_idx, num_stages, num_layers, seq_len, prefix)
                process_time = tflops / self.tflops_capacity
                
                stage_start_times[stage_idx] = start_time + process_time
                
                schedule.append((stage_idx, subseq_idx, start_time, process_time, tflops))
        
        return stage_start_times[-1], schedule
    
    def get_all_partitions(self, num_pp_stages, num_seq_partitions,uniform=False):
    
        if uniform:
            seq_partitions = [self.total_seqlen // num_seq_partitions] * num_seq_partitions
        else: 
            seq_partitions=Solver(self.total_seqlen,num_pp_stages,num_seq_partitions,self.config).solve_partition()
        layer_partitions = self.partition_layers(num_pp_stages)
        
        for i in range(self.total_seqlen % num_seq_partitions):
            seq_partitions[i] += 1
        total_latency, schedule = self.calculate_pipeline_latency(layer_partitions, seq_partitions)
        return total_latency, schedule

def plot_pp_schedule(schedule, num_pp_stages, num_seq_partitions):
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, num_seq_partitions))

    for stage, subseq, start_time, duration, tflops in schedule:
        bar = ax.barh(stage, duration, left=start_time, height=0.5, 
                color=colors[subseq], alpha=0.8, 
                label=f'Subseq {subseq+1}' if stage == 0 else "")
        
        # 在每个 chunk 上添加 TFLOPs 标注
        ax.text(start_time + duration/2, stage, f'{tflops:.2f}T',
                ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(range(num_pp_stages))
    ax.set_yticklabels([f'Stage {i+1}' for i in range(num_pp_stages)])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Pipeline Parallelism Schedule with TFLOPs')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()



class Solver:
    def __init__(self, total_seqlen, num_stages,num_seq_partitions, config):
        self.total_seqlen = total_seqlen
        self.num_stages = num_stages
        self.num_seq_partitions = num_seq_partitions
        self.config = config
        self.stage0_tflops=self.config.get_origin_layer_tflops(self.total_seqlen,causal=True)*self.config.num_layers/self.num_stages+self.config.get_emb_tflops(self.total_seqlen)[0]
        self.stage_average_tflops=self.stage0_tflops/num_seq_partitions


    def solve_partition(self):
        res=[]
        prefix=self.total_seqlen
        for i in range(1,self.num_seq_partitions):
            seqlen=symbols('seqlen')
            tflops=self.config.get_stage0_prefix_tflops(seqlen,prefix)
            eq=Eq(tflops,self.stage_average_tflops)
            sol=solve(eq,seqlen)
            sol=round_down(int(sol[0]),1)
            res.insert(0,int(sol))
            prefix-=int(sol)
        res.insert(0,prefix)
        return res
            

# 使用示例
num_pp_stages=8
num_seq_partitions=16
config = SeqTFlops(hidden_size=8192, num_heads=8, ffn_ratio=4, vocab_size=50000, num_layers=60,num_pp_stages=8,
num_seq_partitions=16)
pp = PipelineParallelism(config, config.num_layers, total_seqlen=131072, tflops_capacity=312)

uni_total_latency, uni_schedule = pp.get_all_partitions(num_pp_stages, num_seq_partitions,uniform=True)
balanced_total_latency, balanced_schedule = pp.get_all_partitions(num_pp_stages, num_seq_partitions,uniform=False)
print(f"Uniform latency for 2 PP stages and 4 sequence partitions: {uni_total_latency:.6f} seconds")
print(f"Balanced latency for 2 PP stages and 4 sequence partitions: {balanced_total_latency:.6f} seconds")
print(f"improve:{(uni_total_latency-balanced_total_latency)/uni_total_latency*100:.2f}%")
plot_pp_schedule(uni_schedule, num_pp_stages, num_seq_partitions)
plot_pp_schedule(balanced_schedule, num_pp_stages, num_seq_partitions)
