import numpy as np
import torch
import torch.nn as nn
import clip
from model.BERT_encoder import load_bert
from model.rotation2xyz import Rotation2xyz
from model.mdm_transformer import MDM_TransformerDecoderLayer, MDM_TransformerDecoder
from utils.misc import recursive_op2

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"  # for debug


class MDM(nn.Module):
    def __init__(self, njoints, nfeats, translation, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec='cls_none_cross_tcond', clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.njoints = njoints
        self.nfeats = nfeats
        self.dataset = dataset

        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.llm_dim = clip_dim #TODO: deprecate

        self.input_feats = self.nfeats
        self.input_feats *= self.njoints

        self.attention_map = {}  # in use for visualization
        self.attention_lookup = 'layer{:02d}_step{:03d}'
        self.get_dict = {}  # for PnP features savings

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'text')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.emb_before_mask = kargs.get('emb_before_mask', False)
        self.diffusion_steps = kargs.get('diffusion_steps', 1000)
        self.arch = arch
        self.input_process = InputProcess(self.input_feats, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.get_feat_idx = kargs.get('get_feat_idx', dict())
        self.transfer_idx = kargs.get('transfer_idx', dict())

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif 'trans_dec' in self.arch:
            print("TRANS_DEC init")
            seqTransDecoderLayer = MDM_TransformerDecoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=activation,
                                                               transfer_idx=self.transfer_idx)
            self.seqTransDecoder = MDM_TransformerDecoder(seqTransDecoderLayer,
                                                          num_layers=self.num_layers,
                                                          get_feat_idx=self.get_feat_idx)
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, PositionalEncoding(self.latent_dim, self.dropout,
                                                                                   max_len=max(1000, self.diffusion_steps)))

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')
                
                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')
                
                if self.text_encoder_type == "clip":
                    print('Loading CLIP...')
                    self.clip_model = self.load_and_freeze_clip(clip_version)
                    self.encode_text = self.clip_encode_text
                elif self.text_encoder_type == 'bert':
                    assert 'trans_dec' in self.arch
                    print("Loading BERT...")
                    bert_model_path = 'distilbert/distilbert-base-uncased'
                    self.bert_model = load_bert(bert_model_path)
                    self.encode_text = self.bert_encode_text
                    self.llm_dim = 768
                    if self.trans_dec_w_cls():
                        print("Loading CLIP too...")
                        self.clip_model = self.load_and_freeze_clip(clip_version)
                        self.encode_text_clip = self.clip_encode_text
                else:
                    raise ValueError('We only support [CLIP, BERT] text encoders') 
                
                self.embed_text = nn.Linear(self.llm_dim, self.latent_dim)
                if self.text_encoder_type == 'bert' and self.trans_dec_w_cls():
                    self.embed_text_clip = nn.Linear(clip_dim, self.latent_dim)
                
        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        seq_len, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float().unsqueeze(0)
    
    def bert_encode_text(self, raw_text):
        # enc_text = self.bert_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.bert_model(raw_text)  # self.bert_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def forward(self, x, timesteps, y=None, features_mode=None, ref_features=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        bs, njoints, nfeats, nframes = x.shape
        if ref_features is None and features_mode == 'transfer' and (timesteps == self.diffusion_steps-1).all():
            # intialize the output data with the noise of the leader
            x[self.transfer_idx['out']] = x[self.transfer_idx['leader']].clone()  # 'clone' is requested by python

        emb_step = self.embed_timestep(timesteps)  # [1, bs, d]

        if y.get('uncond', False):
            features_mode = None
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text'])
            if self.trans_dec_w_cls():
                enc_text_clip = self.encode_text_clip(y['text'])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
            if self.emb_before_mask:
                emb_text = self.mask_cond(self.embed_text(enc_text), force_mask=force_mask)
                # emb = self.mask_cond(self.embed_text(enc_text), force_mask=force_mask) + emb_step
                assert not self.trans_dec_w_cls(self), 'not implemented yet for self.emb_trans_dec'
            else:  # default
                emb_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)) 
            
            if self.arch == 'trans_enc':
                emb = emb_text + emb_step
            elif 'trans_dec' in self.arch:
                # set the condition to be used during cross attention
                if 'cross_cond' in self.emb_trans_dec:
                    emb_cross = emb_text
                else:
                    assert 'cross_tcond' in self.emb_trans_dec
                    emb_cross = emb_text + emb_step
                # set the condition ("class") to be concatenated to data
                if self.trans_dec_w_cls():
                    emb_cond = self.embed_text_clip(self.mask_cond(enc_text_clip, force_mask=force_mask))
                    if 'cls_cond' in self.emb_trans_dec:
                        emb_cls = emb_cond
                    elif 'cls_t_' in self.emb_trans_dec:
                        emb_cls = emb_step
                    else:  
                        assert 'cls_tcond' in self.emb_trans_dec
                        emb_cls = emb_cond + emb_step

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb_step.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        # create a mask to be used by the transformer
        # todo: move to collate
        frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)
        if self.trans_dec_w_cls() or self.arch == 'trans_enc':
            # in case there is a concatenated frame for t, add a mask for it too
            step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
            frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif 'trans_dec' in self.arch:
            if self.trans_dec_w_cls():
                # concatenate the cls token to the input (cls is either t or t+cond)
                if self.text_encoder_type == 'clip':
                    xseq = torch.cat((emb, x), axis=0)
                else: # bert
                    if x.ndim == 4:
                        emb_cls = emb_cls.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                    xseq = torch.cat((emb_cls, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.text_encoder_type == 'clip':
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask)
            elif self.text_encoder_type == 'bert':
                if timesteps[0] not in self.get_feat_idx.get('step', []): 
                    features_mode = None
                step_i = f'step{timesteps[0]:03d}'
                if features_mode == 'transfer' and ref_features is not None:
                    assert step_i in ref_features
                output, feat_dict = self.seqTransDecoder(tgt=xseq, memory=emb_cross, memory_key_padding_mask=text_mask,
                                                         tgt_key_padding_mask=frames_mask, mode=features_mode)
                if features_mode == 'get':
                    if step_i in self.get_dict:
                        # this condition is for the case where num_repetitions > 1
                        self.get_dict[step_i] = recursive_op2(self.get_dict[step_i], feat_dict, lambda x,y: torch.cat([x, y], axis=1))  # concatenate along the batch dimension
                    else:
                        self.get_dict[step_i] = feat_dict
            else:
                raise ValueError
            if self.trans_dec_w_cls():
                output = output[1:]  # [seqlen, bs, d]
            
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def trans_dec_w_cls(self):
        return 'cls_none' not in self.emb_trans_dec


    def _apply(self, fn):
        super()._apply(fn)
        # self.rot2xyz.smpl_model._apply(fn)  # todo: uncomment after smpl is installed


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        # self.rot2xyz.smpl_model.train(*args, **kwargs)  # todo: uncomment after smpl is installed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# For 2D embedding for mat representation
class DualPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(DualPositionalEncoding, self).__init__(d_model//2, dropout, max_len)
        self.concat_pe = torch.tensor([])  # cache

    def forward(self, x):
        # not used in the final model
        if x.shape[0] != self.concat_pe.shape[0]:
            temporal_pe = self.pe[:x.shape[0], :].unsqueeze(1).tile([1,1,x.shape[2],1])
            joint_pe = self.pe[:x.shape[2], :].unsqueeze(1).permute(1,2,0,3).tile([x.shape[0],1,1,1])
            self.concat_pe = torch.cat([temporal_pe, joint_pe], dim=-1)
        x = x + self.concat_pe
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2))
        x = x.reshape(nframes, bs, njoints*nfeats)

        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs = output.shape[:2]
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output
