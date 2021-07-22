import math
from dataclasses import dataclass

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

    def forward(self, model, sample, reduce=True):
        net_out = model(
            sample["source_lang_pair"]["net_input"], sample["target_lang_pair"]["net_input"]
        )

        print(net_out.keys())

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
