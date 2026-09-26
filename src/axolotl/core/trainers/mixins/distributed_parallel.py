"""
Mixin for correctly saving fsdp
"""

from accelerate import PartialState
from transformers import Trainer


class DistributedParallelMixin(Trainer):
    """
    Mixin for correctly saving fsdp
    """

    def _wrap_model(self, model, *args, **kwargs):
        cfg = getattr(self, "axolotl_cfg", None)
        parallel = self.accelerator.parallelism_config
        distributed_type = self.accelerator.distributed_type
        pure_ddp = (
            getattr(distributed_type, "name", distributed_type) == "MULTI_GPU"
            and not any(
                (getattr(cfg, key, 1) or 1) > 1
                for key in (
                    "tensor_parallel_size",
                    "context_parallel_size",
                    "expert_parallel_size",
                )
            )
            and not any(
                getattr(parallel, key, False)
                for key in ("tp_enabled", "cp_enabled", "dp_shard_enabled")
            )
        )
        if pure_ddp and not getattr(model, "_axolotl_native_nvfp4_ddp_prepared", False):
            from torch.nn.parallel import DistributedDataParallel

            if not isinstance(model, DistributedDataParallel):
                from axolotl.monkeypatch.torchao_ddp import prepare_native_nvfp4_ddp

                if prepare_native_nvfp4_ddp(model, self.accelerator.device):
                    model._axolotl_native_nvfp4_ddp_prepared = True
        return super()._wrap_model(model, *args, **kwargs)

    def _ep_full_param_experts(self) -> bool:
        cfg = getattr(self, "axolotl_cfg", None)
        if not cfg or (getattr(cfg, "expert_parallel_size", 1) or 1) <= 1:
            return False
        if getattr(cfg, "adapter", None) or not self.is_fsdp_enabled:
            return False
        from axolotl.integrations.expert_parallel.shard import _detect_experts_modules

        return any(
            getattr(m, "num_experts_global", m.num_experts) > m.num_experts
            for _n, m in _detect_experts_modules(self.model)
        )

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if not self._ep_full_param_experts():
            return super().save_model(output_dir, _internal_call)

        # The FSDP full state dict holds only this rank's ep-slice of the experts; gather them
        # across the EP axis on every rank before rank 0 writes.
        from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
        from axolotl.integrations.expert_parallel.shard import (
            gather_ep_experts_into_state_dict,
        )

        ep_group = ExpertParallelPlugin._resolve_ep_group(self.axolotl_cfg)
        accelerator = self.accelerator
        orig_get_state_dict = accelerator.get_state_dict

        def _get_state_dict(model, unwrap=True):
            state_dict = orig_get_state_dict(model, unwrap=unwrap)
            gather_ep_experts_into_state_dict(state_dict, model, ep_group)
            return state_dict

        accelerator.get_state_dict = _get_state_dict
        try:
            return super().save_model(output_dir, _internal_call)
        finally:
            accelerator.__dict__.pop("get_state_dict", None)

    def _save(self, output_dir: str | None = None, state_dict=None):
        if (
            state_dict is None
            and self.accelerator.parallelism_config
            and self.accelerator.parallelism_config.dp_shard_enabled
        ):
            state_dict = self.accelerator.get_state_dict(self.model)
        super()._save(output_dir, state_dict=state_dict)

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        if (
            self.accelerator.distributed_type == "FSDP"
            and self.accelerator.state.fsdp_plugin is None
        ):
            # handle Context Parallelism without FSDP
            self.accelerator.state.distributed_type = "MULTI_GPU"
            self.accelerator.state._shared_state["distributed_type"] = "MULTI_GPU"
            PartialState().distributed_type = "MULTI_GPU"
