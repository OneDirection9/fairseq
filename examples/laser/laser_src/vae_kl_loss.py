import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class VaeKLCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    alpha: float = field(
        default=1.0,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )
    anneal_steps: int = field(
        default=1000,
        metadata={"help": "hyper-parameters for Beta-tcvae"},
    )


@register_criterion("vae_kl", dataclass=VaeKLCriterionConfig)
class VaeKLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, alpha, beta, gamma, anneal_steps):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.training = True
        self.num_iter = 0

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.anneal_steps = anneal_steps

    def forward(self, model, sample, reduce=True):
        """
        Args:
            model:
            sample: see BilanguagePairDataset, MultitaskDatasetWrapper
            reduce:
        """
        bsz = sample["source_lang_batch"]["nsentences"]
        ntokens = sample["source_lang_batch"]["ntokens"] + sample["target_lang_batch"]["ntokens"]
        sample_size = bsz + bsz if self.sentence_avg else ntokens

        net_out = model(
            sample["source_lang_batch"]["net_input"], sample["target_lang_batch"]["net_input"]
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
        vae_loss = (source_losses["loss"] + target_losses["loss"]) / 2

        kl_loss = self.compute_kl_loss(
            net_out["source_encoder_out"]["controller_out"]["mu"],
            net_out["source_encoder_out"]["controller_out"]["log_var"],
            net_out["target_encoder_out"]["controller_out"]["mu"],
            net_out["target_encoder_out"]["controller_out"]["log_var"],
        )
        kl_loss = kl_loss.sum() / bsz

        loss = vae_loss + kl_loss
        logging_output = {
            "loss": loss.data,
            "vae_loss": vae_loss.data,
            "kl_loss": kl_loss.data,
            "sample_size": sample_size,
            "ntokens": ntokens,
            "nsentences": bsz * 2,
        }

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
        dataset_len = sum(len(ds) for ds in self.task.datasets["train"].values())

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

    def compute_kl_loss(self, source_mu, source_log_var, target_mu, target_log_var):
        # \log \frac{\sigma_{2}}{\sigma_{1}} +
        # \frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}} -
        # \frac{1}{2}

        # \log \frac{\sigma_{2}}{\sigma_{1}}
        kl_1 = target_log_var - source_log_var
        # \frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}
        kl_2 = (torch.exp(source_log_var) ** 2 + (source_mu - target_mu) ** 2) / (
            2 * torch.exp(target_log_var) ** 2
        )

        return kl_1 + kl_2 - 0.5

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        vae_loss_sum = sum(log.get("vae_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum, round=3)
        metrics.log_scalar("vae_loss", vae_loss_sum, round=3)
        metrics.log_scalar("kl_loss", kl_loss_sum, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
