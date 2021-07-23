import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("vae_gan", dataclass=CrossEntropyCriterionConfig)
class VawGanCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.training = True
        self.num_iter = 0

        self.alpha = 1.0
        self.beta = 6.0
        self.gamma = 1.0
        self.anneal_steps = 10000

    def forward(self, model, sample, reduce=True):
        """
        Args:
            model:
            sample: see BilanguagePairDataset, MultitaskDatasetWrapper
            reduce:
        """
        net_out = model(
            sample["source_lang_batch"]["net_input"], sample["target_lang_batch"]["net_input"]
        )

        source_lang_batch = sample["source_lang_batch"]
        sample_size = (
            source_lang_batch["net_input"]["src_tokens"].size(0)
            if self.sentence_avg
            else source_lang_batch["ntokens"]
        )

        source_losses = self.compute_single_lang_losses(
            sample["source_lang_batch"],
            net_out["source_encoder_out"],
            net_out["source_decoder_out"],
            reduce=reduce,
        )
        target_losses = self.compute_single_lang_losses(
            sample["target_lang_batch"],
            net_out["target_encoder_out"],
            net_out["target_decoder_out"],
            reduce=reduce,
        )
        logging_output = {
            "source_losses": source_losses,
            "target_losses": target_losses,
        }
        loss = source_losses["loss"] + target_losses["loss"]

        return loss, sample_size, logging_output

    def compute_single_lang_losses(self, sample, encoder_out, decoder_out, reduce=True):
        """
        Args:
            model:
            sample:
            encoder_out: see LSTMEncoder
            decoder_out: see LSTMDecoder
            reduce:
        """
        dataset_name = sample["net_input"]["dataset_name"]
        dataset_len = len(self.task.datasets["train"][dataset_name])

        z = encoder_out["controller_out"]["z"]
        mu = encoder_out["controller_out"]["mu"]
        log_var = encoder_out["controller_out"]["log_var"]
        input = sample["net_input"]["src_tokens"]

        weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset

        # decoder_out[0]: B x T x C
        log_probs = F.log_softmax(decoder_out[0], dim=-1, dtype=torch.float32)
        log_probs = log_probs.view(-1, log_probs.size(-1))
        # B x T -> B * T,
        input = input.view(-1)
        recons_loss = F.nll_loss(
            log_probs,
            input,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        M_N = batch_size / dataset_len
        mat_log_q_z = self.log_density_gaussian(
            z.view(batch_size, 1, latent_dim),
            mu.view(1, batch_size, latent_dim),
            log_var.view(1, batch_size, latent_dim),
        )

        # Reference
        # https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54 # noqa
        dataset_size = (1 / M_N) * batch_size  # dataset size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = (
            torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1)).to(input.device)
        )
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.0

        loss = (
            recons_loss / batch_size
            + self.alpha * mi_loss
            + weight * (self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)
        )

        return {
            "loss": loss,
            "reconstruction": recons_loss,
            "KLD": kld_loss,
            "TC_loss": tc_loss,
            "MI_loss": mi_loss,
        }

    def log_density_gaussian(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
