# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)


@register_model("laser_lstm")
class LSTMModel(BaseFairseqModel):
    def __init__(self, base_model, model=None):
        super().__init__()

        self.base_model = base_model
        self.model = model

        self.update_num = 0

    def forward(self, source_lang_input, target_lang_input=None):
        source_out = self.base_model(**source_lang_input)

        target_out = None
        if self.model is not None and target_lang_input is not None:
            target_out = self.model(**target_lang_input)

        return {
            "source_out": source_out,
            "target_out": target_out,
        }

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            metavar="D",
            help="dropout probability",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-embed-path",
            default=None,
            type=str,
            metavar="STR",
            help="path to pre-trained encoder embedding",
        )
        parser.add_argument(
            "--encoder-hidden-size", type=int, metavar="N", help="encoder hidden size"
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="number of encoder layers"
        )
        parser.add_argument(
            "--encoder-bidirectional",
            action="store_true",
            help="make all layers of encoder bidirectional",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-embed-path",
            default=None,
            type=str,
            metavar="STR",
            help="path to pre-trained decoder embedding",
        )
        parser.add_argument(
            "--decoder-hidden-size", type=int, metavar="N", help="decoder hidden size"
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="number of decoder layers"
        )
        parser.add_argument(
            "--decoder-out-embed-dim",
            type=int,
            metavar="N",
            help="decoder output embedding dimension",
        )
        parser.add_argument(
            "--decoder-zero-init",
            type=str,
            metavar="BOOL",
            help="initialize the decoder hidden/cell state to zero",
        )
        parser.add_argument(
            "--decoder-lang-embed-dim",
            type=int,
            metavar="N",
            help="decoder language embedding dimension",
        )
        parser.add_argument(
            "--fixed-embeddings",
            action="store_true",
            help="keep embeddings fixed (ENCODER ONLY)",
        )  # TODO Also apply to decoder embeddings?

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument(
            "--encoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for encoder input embedding",
        )
        parser.add_argument(
            "--encoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for encoder output",
        )
        parser.add_argument(
            "--decoder-dropout-in",
            type=float,
            metavar="D",
            help="dropout probability for decoder input embedding",
        )
        parser.add_argument(
            "--decoder-dropout-out",
            type=float,
            metavar="D",
            help="dropout probability for decoder output",
        )

        parser.add_argument(
            "--controller-hidden-dim", type=int, metavar="N", help="controller hidden dim"
        )
        parser.add_argument(
            "--controller-latent-dim", type=int, metavar="N", help="controller latent dim"
        )

        parser.add_argument(
            "--base-model", type=str, default=None, help="pretrained language model"
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        pretrained_encoder_embed = None
        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path, task.target_dictionary, args.decoder_embed_dim
            )

        def build_encoder_decoder_model():
            encoder = LSTMEncoder(
                dictionary=task.source_dictionary,
                embed_dim=args.encoder_embed_dim,
                hidden_size=args.encoder_hidden_size,
                num_layers=args.encoder_layers,
                dropout_in=args.encoder_dropout_in,
                dropout_out=args.encoder_dropout_out,
                bidirectional=args.encoder_bidirectional,
                pretrained_embed=pretrained_encoder_embed,
                fixed_embeddings=args.fixed_embeddings,
                controller_hidden_dim=args.controller_hidden_dim,
                controller_latent_dim=args.controller_latent_dim,
                controller_output_dim=args.decoder_layers * args.decoder_hidden_size * 2,
            )

            assert task.source_dictionary == task.target_dictionary
            decoder = LSTMDecoder(
                dictionary=task.source_dictionary,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_size,
                out_embed_dim=args.decoder_out_embed_dim,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                zero_init=options.eval_bool(args.decoder_zero_init),
                encoder_embed_dim=args.encoder_embed_dim,
                encoder_output_units=encoder.output_units,
                pretrained_embed=pretrained_decoder_embed,
            )
            return TMPModel(encoder, decoder)

        base_model = build_encoder_decoder_model()

        model = None
        if args.base_model is not None:
            model = build_encoder_decoder_model()

            # TODO: load checkpoint
            # Fix base model
            for p in base_model.parameters():
                p.requires_grad = False

        return cls(base_model, model)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates


class TMPModel(FairseqEncoderDecoderModel):
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return {
            "encoder_out": encoder_out,
            "decoder_out": decoder_out,
        }


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        bidirectional=False,
        left_pad=True,
        pretrained_embed=None,
        padding_value=0.0,
        fixed_embeddings=False,
        controller_hidden_dim=512,
        controller_latent_dim=512,
        controller_output_dim=1024,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed
        if fixed_embeddings:
            self.embed_tokens.weight.requires_grad = False

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

        self.controller = Controller(
            input_dim=num_layers * self.output_units,
            hidden_dim=controller_hidden_dim,
            latent_dim=controller_latent_dim,
            output_dim=controller_output_dim,
        )

    def forward(self, src_tokens, src_lengths, dataset_name):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        try:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())
        except BaseException:
            raise Exception(f"Packing failed in dataset {dataset_name}")

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                return torch.cat(
                    [
                        torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(
                            1, bsz, self.output_units
                        )
                        for i in range(self.num_layers)
                    ],
                    dim=0,
                )

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float("-inf")).type_as(x)

        controller_out = self.controller(final_hiddens, final_cells)

        return {
            "encoder_out": (x, final_hiddens, final_cells),
            "controller_out": controller_out,
            "encoder_padding_mask": encoder_padding_mask if encoder_padding_mask.any() else None,
        }

    def reorder_encoder_out(self, encoder_out_dict, new_order):
        encoder_out_dict["encoder_out"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out_dict["encoder_out"]
        )
        encoder_out_dict["controller_out"] = {
            k: v.index_select(0, new_order) for k, v in encoder_out_dict["controller_out"].items()
        }
        if encoder_out_dict["encoder_padding_mask"] is not None:
            encoder_out_dict["encoder_padding_mask"] = encoder_out_dict[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out_dict

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        zero_init=False,
        encoder_embed_dim=512,
        encoder_output_units=512,
        pretrained_embed=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.layers = nn.ModuleList(
            [
                LSTMCell(
                    input_size=embed_dim if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

        self.zero_init = zero_init

    def forward(
        self,
        prev_output_tokens,
        encoder_out_dict,
        incremental_state=None,
    ):
        encoder_out = encoder_out_dict["encoder_out"]

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, "cached_state")
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            if self.zero_init:
                prev_hiddens = [
                    x.data.new(bsz, self.hidden_size).zero_() for i in range(num_layers)
                ]
                prev_cells = [x.data.new(bsz, self.hidden_size).zero_() for i in range(num_layers)]
            else:
                init = encoder_out_dict["controller_out"]["recons"]
                prev_hiddens = [
                    init[:, (2 * i) * self.hidden_size : (2 * i + 1) * self.hidden_size]
                    for i in range(num_layers)
                ]
                prev_cells = [
                    init[
                        :,
                        (2 * i + 1) * self.hidden_size : (2 * i + 2) * self.hidden_size,
                    ]
                    for i in range(num_layers)
                ]
            input_feed = x.data.new(bsz, self.hidden_size).zero_()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            input = x[j, :, :]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self,
            incremental_state,
            "cached_state",
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        # project back to size of vocabulary
        if hasattr(self, "additional_fc"):
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        x = self.fc_out(x)

        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, "cached_state")
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, "cached_state", new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number


class Controller(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(Controller, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.fc_hidden = nn.Linear(input_dim, hidden_dim)
        self.fc_cell = nn.Linear(input_dim, hidden_dim)

        self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_var = nn.Linear(2 * hidden_dim, latent_dim)

        # TODO: nn.Linear or Linear
        self.fc_out = nn.Linear(latent_dim, output_dim)

    def forward(self, final_hiddens: torch.Tensor, final_cells: torch.Tensor):
        """

        `B` is batch size.
        `E` is encoder layers.
        `D` is decoder layers.

        Args:
            final_hiddens: tensor of size ``E x B x H_in``
            final_cells: tensor of size ``E x B x H_in``
        """
        # E x B x H_in -> B x E x H_in -> B x (E * H_in)
        final_hiddens = final_hiddens.transpose(0, 1).flatten(start_dim=1)
        final_cells = final_cells.transpose(0, 1).flatten(start_dim=1)

        final_hiddens = F.leaky_relu(final_hiddens)
        final_cells = F.leaky_relu(final_cells)

        # B x (E * H_in) -> B x hidden_dim
        final_hiddens = self.fc_hidden(final_hiddens)
        final_cells = self.fc_cell(final_cells)

        # B x hidden_dim -> B x (2 * hidden_dim)
        h_c = torch.cat([final_hiddens, final_cells], dim=1)

        # B x (2 * hidden_dim) -> B x latent_dim
        mu = self.fc_mu(h_c)
        log_var = self.fc_var(h_c)
        z = self.reparameterize(mu, log_var)

        # B x latent_dim -> B x output_dim
        recons = self.fc_out(z)

        return {"mu": mu, "log_var": log_var, "z": z, "recons": recons}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture("laser_lstm", "laser_lstm")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_hidden_size = getattr(args, "encoder_hidden_size", args.encoder_embed_dim)
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", False)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", args.decoder_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 512)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.decoder_zero_init = getattr(args, "decoder_zero_init", "0")
    args.fixed_embeddings = getattr(args, "fixed_embeddings", False)

    args.controller_latent_dim = getattr(args, "controller_latent_dim", 1024)
    args.controller_hidden_dim = getattr(args, "controller_hidden_dim", 1024)
    args.base_model = getattr(args, "base_model", None)
