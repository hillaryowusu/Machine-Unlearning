import warnings
import torch

# pylint: disable=import-error
from taker.model_repos import test_model_repos
from taker import Model
from taker.data_classes import PruningConfig
from taker.prune import prune_and_evaluate

class TestPruneAndEvaluate:
    pruning_config = PruningConfig("nickypro/tinyllama-15m",
        attn_mode="pre-out", do_attn_mean_offset=False, use_accelerator=False,
        ff_frac=0.1, ff_eps=0.1, attn_frac=0.001, attn_eps=1e-4,
        token_limit=1000, focus="pile", cripple="code", ff_scoring="abs", attn_scoring="abs")

    def __run_testing(self, _pruning_config: PruningConfig):
        c = _pruning_config
        opt = Model(c.model_repo, limit=c.token_limit, dtype="fp32",
                    use_accelerator=c.use_accelerator)
        data = prune_and_evaluate(opt, c)

        pile_loss = data.loss_data['pile']['loss']
        code_loss = data.loss_data['code']['loss']
        assert pile_loss > 1
        assert code_loss > 1

    def test_prune_and_evaluate(self, model_repo):
        c = self.pruning_config
        c.model_repo          = model_repo
        c.attn_mode           = "pre-out"
        c.use_accelerator     = False
        c.do_attn_mean_offset = False
        c.ff_scoring = "abs"
        c.attn_scoring="abs"

        self.__
