# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import math
from argparse import Namespace
from distutils.util import strtobool

import numpy
import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect, ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer

from einops import repeat

class E2E(torch.nn.Module):
    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, odim, args, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # Check the relative positional encoding type
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        if (
            self.rel_pos_type is None
            and args.transformer_encoder_attn_layer_type == "rel_mha"
        ):
            args.transformer_encoder_attn_layer_type = "legacy_rel_mha"
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )

        idim = 80

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
        )
        
        
        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio

        self.proj_decoder = None
        if args.adim != args.ddim:
            self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

        if args.mtlalpha < 1:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )
        else:
            self.decoder = None
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")

        for name, param in self.decoder.named_parameters():
            param.requires_grad = False

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )

        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        
        ## compact audio memory ##
        self.embedding = torch.nn.Embedding(num_embeddings=202, embedding_dim=768)
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.layer_norm1 = torch.nn.LayerNorm(768)
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.layer_norm2 = torch.nn.LayerNorm(768)
        
        pre_state_dict = torch.load('/mnt/ssd3/jh/Exp/tmm/av_hubert/exp/qasr/base/checkpoints/checkpoint_best.pt')['model'] # v1
        #pre_state_dict = torch.load('/mnt/ssd2/mnt/ssd/jh/av_hubert/avhubert/exp/cam/cam_200_layer4/checkpoints/checkpoint_best.pt')['model'] # v2
        embedding_state_dict = {}
        for key in pre_state_dict.keys():
           if "embedding" in key and 'tacotron' not in key:
               embedding_state_dict[key[10:]] = pre_state_dict[key]
        self.embedding.load_state_dict(embedding_state_dict)
        
        
    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths, label):

        with torch.no_grad() :
            if self.transformer_input_layer == "conv1d":
                lengths = torch.div(lengths, 640, rounding_mode="trunc")
            padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
                
            x, _ = self.encoder(x, padding_mask)
        # x : B T D
        #import pdb;pdb.set_trace()

        B, T, D = x.size()
        s_units = torch.tensor(list(range(200))) + 1
        s_units = repeat(s_units, 't -> t b', b=B)
        speech_units = self.embedding(s_units.cuda()).detach() # 200 512 B
        #print(speech_units)
        
        x = x.transpose(0,1)
        
        x1, _ = self.mha0(query=x, key=speech_units, value=speech_units) # T B D
        x = x + x1
        x = self.layer_norm1(x)
        x2, _ = self.mha1(query=x, key=speech_units, value=speech_units) # T B D
        x2 = x + x2
        x2 = self.layer_norm2(x2)    
        # ctc loss
        
        x2 = x2.transpose(0,1)
        
        loss_ctc, ys_hat = self.ctc(x2, lengths, label)

        if self.proj_decoder:
            x = self.proj_decoder(x)

        # decoder loss

        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x2, padding_mask)
        #import pdb;pdb.set_trace()
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )


        return loss, loss_ctc, loss_att, acc
