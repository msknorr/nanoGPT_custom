"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def like(tensor: torch.Tensor):
    return {'dtype': tensor.dtype, 'device': tensor.device}


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


from rotary_embedding_torch import RotaryEmbedding

class SelfAttention(nn.Module):

    def __init__(self, config, causal, local=False):
        super().__init__()
        


        n_embd = config.n_emb_local if local else config.n_embd
        assert n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))
        self.is_causal = causal

        if config.pos_enc == 'rope':
            self.rotary_emb = RotaryEmbedding(dim = n_embd // config.n_head)


    def forward(self, x, xa=None, is_causal=False):
        B, T, C = x.size()
        #print("self attn:", B, T, C)
        if xa is not None:  # cor cross attention
            q = self.c_attn(x)[:, :, :self.n_embd]
            k, v = self.c_attn(xa)[:, :, self.n_embd:].chunk(2, dim=2)
        else:
            q, k, v = self.c_attn(x).chunk(3, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if hasattr(self, 'rotary_emb'):
            #seq_len = k.shape[-2]
           # cos, sin = self.rotary_emb(v, seq_len)
           # k, v = apply_rotary_pos_emb(q, k, cos, sin, position_ids=None)
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.is_causal:
            mask = self.causal_mask[:,:,:T,:T]  # Ensure the mask is the right size for the input
            att = att.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):

    def __init__(self, config, local=False):
        super().__init__()
        n_emb = config.n_emb_local if local else config.n_embd
    
        self.c_fc    = nn.Linear(n_emb, 4 * n_emb, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_emb, n_emb, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
   



class Block(nn.Module):

    def __init__(self, config, causal=False, local=False):
        super().__init__()
        n_embd = config.n_emb_local if local else config.n_embd
        print("n_embd block:", n_embd)
        self.ln_1 = LayerNorm(n_embd, bias=config.bias)
        self.attn = SelfAttention(config, causal=causal, local=local)
        self.ln_2 = LayerNorm(n_embd, bias=config.bias)
        self.mlp = MLP(config, local=local)
        self.causal = causal    


    def forward(self, x, xa=None):
        # LLama block https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        
        residual = x
        x = self.ln_1(x)

        
        x = self.attn(x, xa=xa, is_causal=self.causal)
        x = residual + x

        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual
        return x



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_layer_local: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_emb_local: int = 200
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pos_enc: str = 'learnt' # 'learnt' or None
    moe: bool = False
    rope_theta: float = 10000.0
    intermediate_layer_loss: bool = False 
    #global_context_size: int = 5
    patch_method: str = 'utf8'



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        print("Found blocksize:", config.block_size)
        self.config = config

        assert config.n_layer % 3 == 0
        n_initial_layers = config.n_layer_local//2

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb_local),
            wpe = nn.Embedding(config.block_size, config.n_emb_local),
            
            drop = nn.Dropout(config.dropout),
            h_initial = nn.ModuleList([Block(config, causal=True, local=True) for _ in range(n_initial_layers)]),
            h_global = nn.ModuleList([Block(config, causal=True, local=False) for _ in range(config.n_layer)]),
            h_final = nn.ModuleList([Block(config, causal=True, local=True) for _ in range(n_initial_layers)]),
            ln_f = LayerNorm(config.n_emb_local, bias=config.bias),
        ))
        self.global_context_size = config.block_size // 6  # 6 = default patchsize
        self.wpe_global = nn.Parameter(torch.randn(self.global_context_size, config.n_embd))

        #if config.intermediate_layer_loss:
        #    self.intermediate_heads = nn.ModuleList([nn.Linear(config.n_emb_local, config.vocab_size, bias=False) for _ in range(config.n_layer - 1)])  # last one is covered as per usual
        #    reduction_factor = 0.7  # the higher the moe weight to early layers
        #    self.intermediate_head_scale = list(reversed([reduction_factor**(i+1) for i in range(config.n_layer - 1)]))  # +1 because final head is covered below
        #    print("Using intermediate heads with scales:", self.intermediate_head_scale)

        self.lm_head = nn.Linear(config.n_emb_local, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)
       
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))  # apply special scaled init to the residual projections, per GPT-2 paper
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))



    def forward(self, idx, targets=None):

        x = self.transformer.wte(idx)
        
        B, Tx, d = x.shape  # Tx = number of tokens in the sequence
        assert Tx <= self.config.block_size
        
        pos = torch.arange(0, Tx, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = x + pos_emb

        for block in self.transformer.h_initial:
            x = block(x) #, log=log)
 

        D = self.config.n_embd  # die große size
        T = self.config.block_size  # das gleiche wie Tx (oben)
        TG = self.global_context_size
        P = T // TG

        # if patch metho != perioic:
        global_T = torch.full((B,), -1)
        max_global_T = min(TG, Tx)  # context size des großen, aber nur wenn kleiner als context size des kleinen
        global_ts = torch.full((B, max_global_T), Tx-1, device=idx.device)  # Batch*max_global_T tensor mit Tx-1 gefüllt
        #print("global_ts:", global_ts)

        stop_gen = []
        def set_global_ts(use_global):
            for b, use_global0 in enumerate(use_global):
                global_ts0, = use_global0.nonzero(as_tuple=True)
                if len(global_ts0) > TG:
                    if targets is not None:
                        targets[b, global_ts0[TG]:] = -1
                    else:
                        stop_gen.append((b, global_ts0[TG]))
                global_ts0 = global_ts0[:TG]
                global_T[b] = len(global_ts0)
                assert global_T[b] <= max_global_T
                global_ts[b, :global_T[b]] = global_ts0

        
            
        use_global = (  # find space tokens --> bool
            (idx < ord('0')) |
            ((ord('9') < idx) & (idx < ord('A'))) | 
            ((ord('Z') < idx) & (idx < ord('a'))) |
            ((ord('z') < idx) & (idx < 0b1000_0000)) |
            (0b1100_0000 <= idx)
        )
        
        use_global[:, 1:] &= use_global[:, :-1].bitwise_not()  # True only if it was originally True and the preceding cell in the same row was False -> start positions??
        use_global |= idx == -42 #c.BOS
        set_global_ts(use_global)

        y = x.gather(1, global_ts[:, :, None].expand(B, max_global_T, d))  # wir nehmen für die große embed size da wo unsere globalen tokens sind
        y = torch.cat([torch.zeros(B, max_global_T, D-d, **like(x)), y], -1)  # die last dim (feature dim) wird aufgefüllt, sodass kleine dim -> großedim
        y = y + self.wpe_global[:max_global_T]


        for block in self.transformer.h_global:
            y = block(y) #, log=log)
        

       # print("x.shape", x.shape, "global_ts.shape", global_ts.shape, "global_T.shape", global_T.shape, "y.shape", y.shape)
        assert x.shape[0] == global_ts.shape[0] == global_T.shape[0] == y.shape[0]
       
        x = torch.stack([x0.index_add(0, ts[:Ty0], y0[:Ty0, -d:]) for x0, ts, Ty0, y0 in zip(x, global_ts, global_T, y) ])  # zurückrechnen auf small block size


        del y
        for block in self.transformer.h_final:
            x = block(x) #, log=log)

        x = self.transformer.ln_f(x)

        losses = []
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            _loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) #+ recon_losses
            losses.append(_loss)
            bpc = _loss / torch.log(torch.tensor(2.0))
            loss = torch.stack(losses).mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            bpc = None
        

        return logits, loss, bpc
    




        awdwad
        #_____________________________________________________________________________
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        if self.config.pos_enc == 'learnt':
            x = self.transformer.drop(tok_emb + pos_emb)
        elif self.config.pos_enc == None or self.config.pos_enc == 'rope':
            x = self.transformer.drop(tok_emb)
        else:
            raise ValueError("pos_enc must be 'learnt' or None")
        

        
        losses = []
        for ii, block in enumerate(self.transformer.h):
            x = block(x)

            if hasattr(self, 'intermediate_heads') and ii < len(self.intermediate_heads):
                logits = self.intermediate_heads[ii](x) if hasattr(self, 'intermediate_heads') else None
                _loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) * self.intermediate_head_scale[ii]
                losses.append(_loss)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            _loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) #+ recon_losses
            losses.append(_loss)
            bpc = _loss / torch.log(torch.tensor(2.0))
            loss = torch.stack(losses).mean()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            bpc = None
        

        return logits, loss, bpc





#####################################################################################################################################################
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx








"""
class NoisyTopkRouter(nn.Module):
    # https://github.com/AviSoori1x/makeMoE/blob/main/makeMoE.py
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices"""

"""class Expert(nn.Module):
   #An MLP is a simple linear layer followed by a non-linearity i.e. each Expert

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.1),  # as per the original implementation: https://github.com/AviSoori1x/makeMoE/blob/main/makeMoE.py
        )

    def forward(self, x):
        return self.net(x)
    
class SparseMoE(nn.Module):
    # https://github.com/AviSoori1x/makeMoE/blob/main/makeMoE.py
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
    
    def forward(self, x):
    # Assuming x has shape [batch_size, seq_len, n_embd]
        batch_size, seq_len, _ = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Flatten the batch and sequence dimensions to treat each token independently
        flat_x = x.view(-1, x.size(-1))  
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)

        updates = torch.zeros_like(flat_x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                updates.index_add_(0, limited_indices, weighted_output)

        # Reshape updates to match the original dimensions of x
        final_output += updates.view(batch_size, seq_len, -1)

        return final_output"""





"""class ReconBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(config.n_embd) * 0.01)
        self.norm = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SuperCausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, mask_prop, x, xa):
        assert torch.isnan(x).sum() == 0
        # x.shape = (12, 64, 504)
        x_orig = x
        
        mask = torch.rand(x.shape[0], x.shape[1]) < mask_prop
        mask[:, 0] = False

        x[mask] = self.mask_token

        x = x + self.attn(x, xa)
        x = x + self.mlp(self.norm(x))

        x = x[mask].reshape(-1, x.shape[-1])
        x_orig = x_orig[mask].reshape(-1, x_orig.shape[-1])

        cos_sim_loss = 1 - F.cosine_similarity(x, x_orig, dim=-1).mean()

        #print(cos_sim_loss)
        return cos_sim_loss"""

