import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask
import torch.nn.functional as F
from model.mdm_multiheadattention import multi_head_attention_forward
F.multi_head_attention_forward = multi_head_attention_forward
# from model.mdm_multiheadattention import MDM_MultiheadAttention as MultiheadAttention
from typing import Optional, Any, Union, Callable
from torch import Tensor, LongTensor

# derived and partially replicated from torch/nn/modules/transformer.py (pytorch 2.3.1, pytorch-cuda 12.1)

class MDM_TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, **kwargs):
        super().__init__(decoder_layer, num_layers, norm)
        self.get_feat_idx = kwargs.get('get_feat_idx', {})       


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, mode: Optional[str] = None) -> Tensor:
        output = tgt
        get_feat = {}
        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for layer_i, mod in enumerate(self.layers):
            layer_i_str = f'layer{layer_i:02d}'
            cur_mode = mode
            if layer_i not in self.get_feat_idx.get('layer', []):
                cur_mode = None
            output, layer_get_feat = mod(output, memory, tgt_mask=tgt_mask,
                                         memory_mask=memory_mask,
                                         tgt_key_padding_mask=tgt_key_padding_mask,
                                         memory_key_padding_mask=memory_key_padding_mask,
				                         tgt_is_causal=tgt_is_causal,
				                         memory_is_causal=memory_is_causal,
                                         mode=cur_mode)
            get_feat[layer_i_str] = layer_get_feat

        if self.norm is not None:
            output = self.norm(output)

        return output, get_feat

class MDM_TransformerDecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, **kwargs) -> None:
        super(MDM_TransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,	
                                                          layer_norm_eps, batch_first, norm_first, bias, device, dtype)
        
        self.transfer_idx = kwargs.get('transfer_idx', {})       
        self.nhead = nhead
        

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        mode: Optional[str] = None
    ) -> Tensor:

        self.layer_get_feat = {}
        x = tgt  # hml: n_frames, n_samples, n_features
        if self.norm_first:
            raise NotImplementedError('[norm_first] is not supported at the moment')
        else:
            # self attn 
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal, mode))
            if mode == 'get': self.layer_get_feat['sa_frame_res'] = x.clone()
            
            # cross attn 
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal, mode=mode))
            if mode == 'get': self.layer_get_feat['ca_frame_text_res'] = x.clone()
                     
            # feed forward
            x = self.norm3(x + self._ff_block(x, mode))
            if mode == 'get': self.layer_get_feat['ff_res'] = x.clone()

        return x, self.layer_get_feat

    def _self_attn_wrap(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor] = None, 
                        is_causal: bool = False, mode: Optional[str] = None) -> Tensor:
        if mode == 'transfer':
            n_tokens, _, n_features = x.shape
            leader_idx, follower_idx, out_idx = [self.transfer_idx[i] for i in ['leader', 'follower', 'out']]
            assert key_padding_mask.shape[0] == len(follower_idx)
            key_padding_mask = key_padding_mask.reshape(len(out_idx), -1)
            follower_mult = len(follower_idx) // len(out_idx)
            follower_idx_mult = LongTensor(follower_idx).reshape(-1, follower_mult)
            q_all, k_all, v_all = x, x, x
            q = torch.zeros((n_tokens, len(out_idx), n_features), device=x.device, dtype=x.dtype)
            k, v = [torch.zeros((n_tokens * follower_mult, len(out_idx), n_features), device=x.device, dtype=x.dtype) for _ in range(2)]
            for cur_follower_idx, cur_out_idx in zip(follower_idx_mult, out_idx):
                q[:, cur_out_idx-out_idx[0]] = q_all[:, leader_idx]
                k[:, cur_out_idx-out_idx[0]] = k_all[:, cur_follower_idx].reshape(-1, n_features)
                v[:, cur_out_idx-out_idx[0]] = v_all[:, cur_follower_idx].reshape(-1, n_features)
        else:
            q, k, v = x, x, x

        # average_attn_weights should be False because we sometimes want to extract attention weights
        x, attn_weights = self.self_attn(q, k, v, 
					                     attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask,
                                         is_causal=is_causal,
                                         need_weights=(mode=='get'), 
					   average_attn_weights=False)
        return x, attn_weights

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False, mode: Optional[str] = None) -> Tensor:
        if mode != 'transfer':
            x, attn_weights = self._self_attn_wrap(x, attn_mask, key_padding_mask, is_causal, mode)      
        else:
            # sa for 'leader' and 'follower' features
            leader_follower_idx = [self.transfer_idx['leader']] + self.transfer_idx['follower']
            assert attn_mask is None and key_padding_mask is not None  # a lazy way to refrain from code like "attn_mask[:, idx] if attn_mask is not None else None"
            # regular self attention for 'leader' and 'follower' 
            x_leader_follower, _ = self._self_attn_wrap(x[:, leader_follower_idx], attn_mask, 
                                                          key_padding_mask=key_padding_mask[leader_follower_idx], is_causal=is_causal, mode=None)
            # cross attention between 'leader' (q) and 'follower' (k,v), and the result is put into 'out'
            x_out, _ = self._self_attn_wrap(x, attn_mask, key_padding_mask=key_padding_mask[self.transfer_idx['follower']], is_causal=is_causal, mode=mode)
            x = torch.cat([x_leader_follower, x_out], dim=1)
        if mode == 'get': 
            self.layer_get_feat['sa_frame'] = x.clone()  # .cpu()
            self.layer_get_feat['sa_frame_atn'] = attn_weights['atn'].clone()
            # n_samples [x n_heads] x n_frames x n_feat[//n_heads]  ==>  n_frames x (n_samples[*n_heads]) x n_feat[//n_heads]
            qkv_shape = attn_weights['q'].shape
            for attn_weight in ['q', 'k', 'v']:
                self.layer_get_feat[f'sa_frame_{attn_weight}'] = attn_weights[attn_weight].movedim(-2, 0).reshape(qkv_shape[-2], -1, qkv_shape[-1]).clone()
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False, mode: Optional[str] = None) -> Tensor:
        is_get = (mode=='get')
        x, attn_weights = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=is_get, 
				  average_attn_weights=False)
        if is_get:  
            qk = attn_weights['atn']    
            # permute qk to match x order, so outer methods that use features treat qk same as other features
            qk = qk.permute(-2, 0, 1, 3)  # ==> n_frames, n_samples, n_heads, n_features
            self.layer_get_feat['ca_frame_text_atn'] = qk.clone()
            self.layer_get_feat['ca_frame_text'] = x.clone()
        return self.dropout2(x)

    def _ff_block(self, x: Tensor, mode: Optional[str] = None) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if mode == 'get': self.layer_get_feat['ff'] = x.clone()
        return self.dropout3(x)
