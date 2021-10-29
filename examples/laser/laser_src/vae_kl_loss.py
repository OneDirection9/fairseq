import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from omegaconf import II
from torch.distributions import Normal, kl_divergence

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

eps = 1e-6


@dataclass
class VaeKLCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    beta: float = field(
        default=1.0,
        metadata={"help": "hyper-parameters for Beta-vae"},
    )
    gamma: float = field(
        default=10.0,
        metadata={"help": "hyper-parameters for Beta-vae"},
    )
    loss_type: str = field(
        default="B",
        metadata={"help": "loss type"},
    )
    c_max: int = field(
        default=25,
        metadata={"help": "max capacity"},
    )
    c_stop_iter: float = field(
        default=1e5,
        metadata={"help": "capacity max iter"},
    )


@register_criterion("vae_kl", dataclass=VaeKLCriterionConfig)
class VaeKLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, beta, gamma, loss_type, c_max, c_stop_iter):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.training = True

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.c_max = torch.Tensor([c_max])
        self.c_stop_iter = c_stop_iter

    def forward(self, model, sample, reduce=True):
        """
        Args:
            model:
            sample: see BilanguagePairDataset, MultitaskDatasetWrapper
            reduce:
        """
        bsz = sample["source_lang_batch"]["nsentences"]
        ntokens = sample["source_lang_batch"]["ntokens"] + sample["target_lang_batch"]["ntokens"]
        sample_size = 1

        net_out = model(
            sample["source_lang_batch"]["net_input"], sample["target_lang_batch"]["net_input"]
        )
        has_target = net_out["target_out"] is not None

        source_losses = self.compute_single_lang_losses(
            sample["source_lang_batch"],
            net_out["source_out"]["encoder_out"],
            net_out["source_out"]["decoder_out"],
            model.update_num,
            reduce=reduce,
        )

        if has_target:
            target_losses = self.compute_single_lang_losses(
                sample["target_lang_batch"],
                net_out["target_out"]["encoder_out"],
                net_out["target_out"]["decoder_out"],
                model.update_num,
                reduce=reduce,
            )
            vae_loss = source_losses["loss"] + target_losses["loss"]

            kl_loss = self.compute_kl_loss_v2(
                net_out["source_out"]["encoder_out"]["controller_out"]["mu"],
                net_out["source_out"]["encoder_out"]["controller_out"]["log_var"],
                net_out["target_out"]["encoder_out"]["controller_out"]["mu"],
                net_out["target_out"]["encoder_out"]["controller_out"]["log_var"],
            )
            loss = vae_loss + self.lam * kl_loss
        else:
            vae_loss = source_losses["loss"]
            loss = vae_loss

        logging_output = {
            "loss": loss.data,
            "vae_loss": vae_loss.data,
            "sample_size": sample_size,
            "ntokens": ntokens,
            "nsentences": bsz * 2,
            "source_recons": source_losses["reconstruction"].data,
            "source_KLD": source_losses["KLD"].data,
        }

        if has_target:
            logging_output.update(
                {
                    "kl_loss": kl_loss.data,
                    "target_recons": target_losses["reconstruction"].data,
                    "target_KLD": target_losses["KLD"].data,
                }
            )

        return loss, sample_size, logging_output

    def compute_single_lang_losses(self, sample, encoder_out, decoder_out, update_num, reduce=True):
        """
        Args:
            model:
            sample:
            encoder_out: see LSTMEncoder
            decoder_out: see LSTMDecoder
            reduce:
        """
        dataset_len = self.task.dataset_size

        z = encoder_out["controller_out"]["z"]
        mu = encoder_out["controller_out"]["mu"]
        log_var = encoder_out["controller_out"]["log_var"]
        tgt_tokens = sample["target"]

        batch_size, latent_dim = z.shape
        kld_weight = batch_size / dataset_len

        # decoder_out[0]: B x T x C
        log_probs = F.log_softmax(decoder_out[0], dim=-1, dtype=torch.float32)
        log_probs = log_probs.view(-1, log_probs.size(-1))
        # B x T -> B * T,
        tgt_tokens = tgt_tokens.view(-1)
        recons_loss = F.nll_loss(
            log_probs,
            tgt_tokens,
            ignore_index=self.padding_idx,
            reduction="sum",
        )
        recons_loss = recons_loss / sample["ntokens"]

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.c_max = self.c_max.to(tgt_tokens.device)
            C = torch.clamp(self.c_max / self.c_stop_iter * update_num, 0, self.c_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {
            "loss": loss,
            "reconstruction": recons_loss,
            "KLD": kld_loss,
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
        kl_1 = 0.5 * (target_log_var - source_log_var)
        # \frac{\sigma_{1}^{2}+\left(\mu_{1}-\mu_{2}\right)^{2}}{2 \sigma_{2}^{2}}
        kl_2 = (torch.exp(source_log_var) + (source_mu - target_mu) ** 2) / (
            2 * torch.exp(target_log_var) + eps
        )

        return kl_1 + kl_2 - 0.5

    def compute_kl_loss_v2(self, source_mu, source_log_var, target_mu, target_log_var):
        source_normal = Normal(source_mu, torch.exp(0.5 * source_log_var))
        target_normal = Normal(target_mu, torch.exp(0.5 * target_log_var))

        p_q = kl_divergence(source_normal, target_normal)
        q_p = kl_divergence(target_normal, source_normal)
        bsz, _ = p_q.size()

        return (p_q.sum() + q_p.sum()) / bsz

    def compute_js_loss(self, source_mu, source_log_var, target_mu, target_log_var):
        def compute(source_dist: Normal, target_dist: Normal):
            sample = source_dist.sample()

            source_prob = source_dist.log_prob(sample).exp()
            target_prob = target_dist.log_prob(sample).exp()

            mean_prob = 0.5 * (source_prob + target_prob)
            loss = torch.log(source_prob / mean_prob)
            bsz, _ = loss.size()
            return loss.sum() / bsz

        source_normal = Normal(source_mu, torch.exp(0.5 * source_log_var))
        target_normal = Normal(target_mu, torch.exp(0.5 * target_log_var))

        js_loss = 0.5 * (
            compute(source_normal, target_normal) + compute(target_normal, source_normal)
        )

        return js_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        vae_loss_sum = sum(log.get("vae_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)

        source_recons_sum = sum(log.get("source_recons", 0) for log in logging_outputs)
        source_KLD_sum = sum(log.get("source_KLD", 0) for log in logging_outputs)

        target_recons_sum = sum(log.get("target_recons", 0) for log in logging_outputs)
        target_KLD_sum = sum(log.get("target_KLD", 0) for log in logging_outputs)

        # ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, round=3)
        metrics.log_scalar("vae_loss", vae_loss_sum / sample_size, round=3)
        metrics.log_scalar("kl_loss", kl_loss_sum / sample_size, round=5)

        metrics.log_scalar("source_recons", source_recons_sum / sample_size, round=3)
        metrics.log_scalar("source_KLD", source_KLD_sum / sample_size, round=3)

        metrics.log_scalar("target_recons", target_recons_sum / sample_size, round=3)
        metrics.log_scalar("target_KLD", target_KLD_sum / sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
