import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from itertools import combinations_with_replacement, product

def round_down(x, m):
    return int(x / m) * m

class SeqTFlops:
    def __init__(self, hidden_size, num_heads, ffn_ratio, vocab_size, num_layers):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim_head = hidden_size // num_heads
        self.ffn_ratio = ffn_ratio
        self.vocab_size = vocab_size
        self.num_layers = num_layers

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
        # 嵌入层计算量可以忽略
        embed_tflops = 0
        # 输出投影计算
        emb_proj_tflops = 2 * seqlen * self.hidden_size * self.vocab_size
        return embed_tflops / 1e12, emb_proj_tflops / 1e12  # Convert to TFLOPs

    def get_prefix_tflops(self, seqlen, prefix,causal=False):
        scale = 0.5 if causal else 1
      
        attn_tflops = self.get_attention_tflops(seqlen, prefix) * scale
        ffn_tflops = self.get_ffn_tflops(seqlen)
        embed_tflops, emb_proj_tflops = self.get_emb_tflops(seqlen)
        # 总TFLOPS，包括所有层和前向、后向传播
        tf = attn_tflops + ffn_tflops
        # print(f'seqlen:{seqlen},prefix:{prefix},total_tflops:{tf:.4f},attn_tflops:{attn_tflops:.4f},ffn_tflops:{ffn_tflops:.4f},embed_tflops:{embed_tflops:.4f},emb_proj_tflops:{emb_proj_tflops:.4f}')
        return tf  # Already in TFLOPs

    def get_seq_tflops(self, seqlen, causal):
        return self.get_prefix_tflops(seqlen, seqlen,causal)

    def get_attention_tflops(self, seq_len, prefix_len,causal=False):
    
        # Compute FLOPs for multi-head attention

        # Linear projections for Q, K, V: [seq_len, h] x [h, h] -> [seq_len, h]
        qkv_flops = 3 * 2 * seq_len * self.hidden_size * self.hidden_size

        # Attention scores: Q x K^T -> [seq_len, prefix_len]
        attn_scores_flops = 2 * seq_len * prefix_len * self.hidden_size 
        # attn_scores_flops = 2 * seq_len * seq_len * self.hidden_size * scale + 2* (prefix_len-seq_len)* (prefix_len-seq_len) * self.hidden_size
        # Softmax: [seq_len, prefix_len]
        softmax_flops = 5 * seq_len * prefix_len  # Approximate as 5 FLOPs per element
        # softmax_flops = 5 * seq_len * seq_len * scale  + 5 * (prefix_len-seq_len) * (prefix_len-seq_len)
        # Attention weighted sum: [seq_len, prefix_len] x [prefix_len, h] -> [seq_len, h]
        attn_flops = 2 * seq_len * prefix_len * self.hidden_size
        # attn_flops = 2 * seq_len * seq_len * self.hidden_size* scale + 2 * (prefix_len-seq_len) * (prefix_len-seq_len) * self.hidden_size

        # Output projection: [seq_len, h] x [h, h] -> [seq_len, h]
        output_proj_flops = 2 * seq_len * self.hidden_size * self.hidden_size

        # Total Attention FLOPs for forward pass
        total_attn_flops = qkv_flops + attn_scores_flops + softmax_flops + attn_flops + output_proj_flops


        return total_attn_flops / 1e12  # Convert to TFLOPs


class Solver:
    def __init__(self, total_seqlen, config, causal=True):
        self.total_seqlen = total_seqlen 
        self.config = config
        self.total_tflops = config.get_seq_tflops(total_seqlen, causal)
        
    def solve_partition(self, num_splits, tp_size=1):
        res = []
        prefix = self.total_seqlen
        for i in range(1, num_splits):
            seqlen = symbols('seqlen')
            tflops = self.config.get_prefix_tflops(seqlen, prefix)
            eq = Eq(tflops, self.total_tflops / num_splits)
            sol = solve(eq, seqlen)
            sol = round_down(int(sol[0]),tp_size)
            
            res.insert(0, int(sol))
            prefix -= int(sol)
        res.insert(0, prefix)
        return res
  

class PipelineParallelism:
    def __init__(self, config, total_layers, total_seqlen, tflops_capacity):
        self.config = config
        self.total_layers = total_layers
        self.total_seqlen = total_seqlen
        self.tflops_capacity = tflops_capacity  # Already in TFLOPs
        self.solver = Solver(total_seqlen, config)

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

    def estimate_memory_usage(self, num_layers, seq_len):
        return (self.config.hidden_size * seq_len * 4 * num_layers) / (1024 ** 3)  # in GB

    def calculate_stage_tflops(self, stage_idx, num_stages, num_layers, seq_len, prefix_len):
        # Base TFLOPs for transformer layers in this stage
        stage_tflops = self.config.get_prefix_tflops(seq_len, prefix_len) * num_layers 

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

    def optimize_partition(self, num_pp_stages, seq_partition):
        layer_partitions = self.partition_layers(num_pp_stages)
        tflops_per_stage = []
        for i, num_layers in enumerate(layer_partitions):
            start_seq = sum(seq_partition[:i])
            end_seq = sum(seq_partition[:i+1])
            stage_tflops = self.calculate_stage_tflops(i, num_pp_stages, num_layers, end_seq - start_seq, end_seq)
            tflops_per_stage.append(stage_tflops)
        
        memory_per_stage = [self.estimate_memory_usage(layers, seq_len) 
                            for layers, seq_len in zip(layer_partitions, seq_partition)]
        
        total_latency, _ = self.calculate_pipeline_latency(layer_partitions, seq_partition)

        return layer_partitions, tflops_per_stage, memory_per_stage, seq_partition, total_latency

    def get_uniform_partition(self, num_pp_stages, num_seq_partitions):

        layer_partitions = self.partition_layers(num_pp_stages)
        seq_partitions = [self.total_seqlen // num_seq_partitions] * num_seq_partitions
        for i in range(self.total_seqlen % num_seq_partitions):
            seq_partitions[i] += 1
        
        tflops_per_stage = []
        for i, num_layers in enumerate(layer_partitions):
            start_seq = sum(seq_partitions[:i])
            end_seq = sum(seq_partitions[:i+1])
            stage_tflops = self.calculate_stage_tflops(i, num_pp_stages, num_layers, end_seq - start_seq, end_seq)
            tflops_per_stage.append(stage_tflops)
        
        memory_per_stage = [self.estimate_memory_usage(layers, seq_len) 
                            for layers, seq_len in zip(layer_partitions, seq_partitions)]
        
        total_latency, _ = self.calculate_pipeline_latency(layer_partitions, seq_partitions)
        
        return layer_partitions, tflops_per_stage, memory_per_stage, seq_partitions, total_latency
    

    def search_optimal_configuration(self, num_pp_stages, num_seq_partitions, use_solver=False):
        print("\nStarting search for optimal configuration...")
        print(f"PP stages: {num_pp_stages}, Sequence partitions: {num_seq_partitions}")
        print("Partition Sequence | Latency (s) | Total TFLOPS per Subsequence | Attention TFLOPS per Subsequence")
        print("-" * 160)

        best_config = None
        best_results = None
        best_partition = None
        use_solver = True
        if use_solver:
            # Use Solver.solve_partition method to get the partition
            partition = self.solver.solve_partition(num_seq_partitions)
            results = self.optimize_partition(num_pp_stages, partition)
            
            total_tflops_per_subseq = [f"{self.config.get_prefix_tflops(seq_len, sum(partition[:i+1])):.4f}" for i, seq_len in enumerate(partition)]
            attn_tflops_per_subseq = [f"{self.config.get_attention_tflops(seq_len, sum(partition[:i+1])):.4f}" for i, seq_len in enumerate(partition)]
            
            print(f"{partition} | {results[4]:11.6f} | {total_tflops_per_subseq} | {attn_tflops_per_subseq}")
            
            best_config = (num_pp_stages, num_seq_partitions)
            best_results = results
            best_partition = partition
        else:
            # Original search method
            step = 128
            min_partition_size = step
            max_partition_size = self.total_seqlen - (num_seq_partitions - 1) * min_partition_size
            possible_lengths = range(min_partition_size, max_partition_size + 1, step)

            def generate_partitions(total, n, min_size=512, step=512):
                max_size = total - (n - 1) * min_size
                possible_lengths = range(max_size, min_size - 1, -step)  # 从大到小的可能长度
                
                valid_partitions = []
                for combo in combinations_with_replacement(possible_lengths, n):
                    if sum(combo) == total:
                        valid_partitions.append(tuple(sorted(combo, reverse=True)))  # 对每个组合进行从大到小排序
                
                return valid_partitions

            all_partitions = generate_partitions(self.total_seqlen, num_seq_partitions, min_partition_size, step)
            total_iterations = sum(1 for _ in combinations_with_replacement(possible_lengths, num_seq_partitions))
            pbar = tqdm(total=len(all_partitions), desc="Searching configurations")
            for partition in all_partitions:
                if sum(partition) == self.total_seqlen:
                    results = self.optimize_partition(num_pp_stages, partition)
                    
                    total_tflops_per_subseq = [f"{self.config.get_prefix_tflops(seq_len, sum(partition[:i+1])):.4f}" for i, seq_len in enumerate(partition)]
                    attn_tflops_per_subseq = [f"{self.config.get_attention_tflops(seq_len, sum(partition[:i+1])):.4f}" for i, seq_len in enumerate(partition)]
                    
                    print(f"{partition} | {results[4]:11.6f} | {total_tflops_per_subseq} | {attn_tflops_per_subseq}")
                    
                    if best_results is None or results[4] < best_results[4]:
                        best_config = (num_pp_stages, num_seq_partitions)
                        best_results = results
                        best_partition = partition

                pbar.update(1)

            pbar.close()

        if best_partition:
            print("\nBest configuration:")
            best_total_tflops_per_subseq = [f"{self.config.get_prefix_tflops(seq_len, sum(best_partition[:i+1])):.4f}" for i, seq_len in enumerate(best_partition)]
            best_attn_tflops_per_subseq = [f"{self.config.get_attention_tflops(seq_len, sum(best_partition[:i+1])):.4f}" for i, seq_len in enumerate(best_partition)]
            print(f"{best_results[4]:11.6f} | {best_partition} | {best_total_tflops_per_subseq} | {best_attn_tflops_per_subseq}")
        else:
            print("\nNo valid configuration found.")

        return best_config, best_results

    def visualize_comparison(self, best_config, best_results, uniform_results):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 24))
        
        self._visualize_schedule(ax1, best_results, best_config, "Optimal Configuration")
        self._visualize_schedule(ax2, uniform_results, best_config, "Uniform Partition Configuration")
        # self._visualize_schedule(ax3, uniform_results, best_config, "Uniform Partition Configuration")
        
        plt.tight_layout()
        plt.savefig('schedule.png')
        plt.show()

    def _visualize_schedule(self, ax, results, config, title):
        layer_partitions, tflops_per_stage, memory_per_stage, seq_partitions, total_latency = results
        _, schedule = self.calculate_pipeline_latency(layer_partitions, seq_partitions)
        
        num_stages, num_subsequences = config
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, num_subsequences))
        
        for stage, subseq, start, duration, tflops in schedule:
            ax.barh(stage, duration, left=start, height=0.5, 
                    align='center', color=colors[subseq], alpha=0.8)
            ax.text(start + duration/2, stage, f'S-{seq_partitions[subseq]//1024}K', 
                    ha='center', va='center', color='black', fontweight='bold', fontsize=10)
        
        ax.set_yticks(range(num_stages))
        ax.set_yticklabels([f'Stage {i}' for i in range(num_stages)], fontsize=12)
        ax.invert_yaxis()
        
        ax.set_xlabel('Time (seconds)', fontsize=14)
        ax.set_title(f'{title}\nEnd-to-end Latency: {total_latency:.6f} seconds', fontsize=16)

    def print_detailed_info(self, config, results, config_type=""):
        layer_partitions, tflops_per_stage, memory_per_stage, seq_partitions, total_latency = results
        print(f"\n{config_type} Configuration Details:")
        print(f"Number of PP stages: {config[0]}")
        print(f"Number of sequence partitions: {len(seq_partitions)}")
        print(f"Layer distribution: {layer_partitions}")
        print(f"TFLOPs per stage: {[f'{tflops:.4f} TFLOPs' for tflops in tflops_per_stage]}")
        print(f"Memory per stage: {[f'{mem:.2f} GB' for mem in memory_per_stage]}")
        print(f"Total latency: {total_latency:.6f} seconds")

def calculate_tflops_for_subsequence(pp, seq_len, num_layers, is_first_stage, is_last_stage):
    # 计算Transformer层的TFLOPs
    transformer_tflops = pp.config.get_tflops(seq_len) * num_layers

    # 如果是第一个stage，添加embedding层的计算量
    if is_first_stage:
        embedding_tflops = 2 * seq_len * pp.config.hidden_size * pp.config.vocab_size / 1e12
        transformer_tflops += embedding_tflops

    # 如果是最后一个stage，添加额外一层的计算
    if is_last_stage:
        transformer_tflops += pp.config.get_tflops(seq_len)

    return transformer_tflops

def visualize_computation_blocks(pp, best_config, best_results):
    num_stages, num_partitions = best_config
    layer_partitions, _, _, seq_partitions, _ = best_results

    # 计算每个块的TFLOPs
    tflops_blocks = np.zeros((num_stages, num_partitions))
    for stage in range(num_stages):
        num_layers = layer_partitions[stage]
        is_first_stage = (stage == 0)
        is_last_stage = (stage == num_stages - 1)
        
        for partition in range(num_partitions):
            seq_len = seq_partitions[partition]
            tflops = calculate_tflops_for_subsequence(pp, seq_len, num_layers, is_first_stage, is_last_stage)
            tflops_blocks[stage, partition] = tflops

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.imshow(tflops_blocks, cmap='YlOrRd')

    # 设置坐标轴
    ax.set_xticks(np.arange(num_partitions))
    ax.set_yticks(np.arange(num_stages))
    ax.set_xticklabels([f'Seq {i+1}\n({seq_partitions[i]})' for i in range(num_partitions)])
    ax.set_yticklabels([f'Stage {i+1}\n({layer_partitions[i]} layers)' for i in range(num_stages)])

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('TFLOPs', rotation=-90, va="bottom")

    # 在每个单元格中添加数值
    for i in range(num_stages):
        for j in range(num_partitions):
            text = ax.text(j, i, f'{tflops_blocks[i, j]:.2f}',
                           ha="center", va="center", color="black")

    ax.set_title("Computation Distribution (TFLOPs) Across Stages and Sequence Partitions")
    fig.tight_layout()
    plt.show()

# 调用函数以创建可视化


if __name__ == "__main__":
    # Update the config initialization to use NewSeqTFlops
    config = SeqTFlops(hidden_size=8192, num_heads=80, ffn_ratio=4, vocab_size=50000, num_layers=60)
    # config = SeqTFlops(num_layers=60, hidden_size=8192, num_heads=80, dim_head=102, vocab_size=50000)
    total_seqlen = 16384
    tflops_capacity = 312  # Already in TFLOPs

    pp = PipelineParallelism(config, config.num_layers, total_seqlen, tflops_capacity)

    max_pp_stages = 2
    max_seq_partitions = 4

    print(f"Starting search with max_pp_stages={max_pp_stages} and max_seq_partitions={max_seq_partitions}")
    best_config, best_results = pp.search_optimal_configuration(max_pp_stages, max_seq_partitions)
    print("Search completed.")

    print(f"\nBest configuration: {best_config[0]} PP stages, {best_config[1]} sequence partitions")
    print(f"Optimal latency: {best_results[4]:.6f} seconds")
    print(f"Final partition sequence: {best_results[3]}")

    uniform_results = pp.get_uniform_partition(best_config[0], best_config[1])

    print(f"\nUniform partition latency: {uniform_results[4]:.6f} seconds")
    print(f"Improvement over uniform partition: {(uniform_results[4] - best_results[4]) / uniform_results[4] * 100:.2f}%")

    # Print the uniform partition sequence
    print(f"\nUniform partition sequence: {uniform_results[3]}")

    pp.print_detailed_info(best_config, best_results, "Optimal")
 
    pp.print_detailed_info(best_config, uniform_results, "Uniform Partition")

    pp.visualize_comparison(best_config, best_results, uniform_results)

    # Print the final partition sequence
    visualize_computation_blocks(pp, best_config, best_results)


