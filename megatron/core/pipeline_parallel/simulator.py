class SeqTFlops:
    def __init__(self,hidden_size,num_heads,ffn_ratio,vocab_size,num_layers):
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.ffn_ratio=ffn_ratio
        self.vocab_size=vocab_size
        self.num_layers=num_layers
    
    def get_ffn_tflops(self,seq_len):
        ffn_hidden_size=self.hidden_size*self.ffn_ratio
        
        # first layer:[seqlen,h] x [h,4h]--> [seqlen,4h]
        first_layer_flops=2*seq_len*self.hidden_size*ffn_hidden_size
        
        # activation layer 
        activation_flops=seq_len*ffn_hidden_size
        
        # second layer:[seqlen,4h]*[4h,h]
        second_layer_flops=2*seq_len*self.hidden_size*ffn_hidden_size
        
        total_ffn_tflops=first_layer_flops+second_layer_flops+activation_flops
        
        return total_ffn_tflops / 1e12
    
    def get_emb_tflops(self,seq_len):
        emb_proj_tflops=2*seq_len*self.hidden_size*self.vocab_size
        return emb_proj_tflops / 1e12
    
    def get_attention_tflops(self,seq_len,prefix_len):
        
        # linear projection for Q K V:[seq_len,h]x[h,h] -> [seq_len,h]
        qkv_flops=3*2*seq_len*self.hidden_size*self.hidden_size
        
        # QKV -> [seqlen,prefix_len]
        attn_score_flops=2*seq_len*prefix_len*self.hidden_size
        
        # softmax [seqlen,prefix_len]
        softmax_flops= 5*seqlen*prefix_len
        
        # attention weight sum: [seqlen,prefix_len] x [prefix_len,h] -> [seqlen,h]
        attn_flops=2*seq_len*prefix_len*self.hidden_size
        
        # output projection: [seqlen,h] x [h,h] -> [seqlen,h]
        output_project_flops=2*seq_len*self.hidden_size*self.hidden_size
        
        total_attn_flops=qkv_flops+attn_flops+softmax_flops+attn_score_flops+output_project_flops
        
        # causal model
        if seq_len == prefix_len:
            total_attn_flops /=2
            
        return total_attn_flops / 1e12
    
    def get_prefix_tflops(self,seq_len,prefix):
        attn_flops=self.get_attention_tflops(seq_len,prefix)
        ffn_tflops=self.get_ffn_tflops(seq_len)
        
        tf=attn_flops+ffn_tflops
        
        return tf
    
    def get_seq_tflops(self,seq_len,causal=True):
        return self.get_prefix_tflops(seq_len,seqlen)
    
    
class PipelineParallism:
    def __init__(self,config,total_layer):
        self.config=config
        self.total_layer=total_layer
    
    def partition_layers(self,num_pp_stage):
        partitions=[0]*num_pp_stage
        layers_per_stage=self.total_layer // num_pp_stage
        for i in range(num_pp_stage):
            partitions[i]+=layers_per_stage
        partitions[0]+=1
        partitions[-1]+=2
        return partitions
            
    def partition_sequence(self,num_seq_partitions):
        seq_per_partition=self.total_seqlen//num_seq_partitions
        partitions = [seq_per_partition]*num_seq_partitions
        return partitions

if __name__=="__main__":
    config=SeqTFlops(hidden_size=4096,num_heads=8,ffn_ratio=4,vocab_size=50000,num_layers=32)
    total_seqlen=8192
    tflops_capacity=312
    
    
    num_pp_stages=4
    num_seq_partitions=4
    
    pp=PipelineParallism(config,config.num_layers)
    
    layer_partitions=pp.partition_layers(num_pp_stage)
    seq_partitions=pp.partition_sequence(num_seq_partitions)