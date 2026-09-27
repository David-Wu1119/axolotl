"""Validation of tensor_parallel_size combinations."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestTensorParallelValidation:
    """TP composes with FSDP/CP but not with adapters or expert parallelism."""

    def test_rejects_adapter(self, min_base_cfg):
        cfg = (
            DictDefault(tensor_parallel_size=2, adapter="lora", lora_r=8) | min_base_cfg
        )
        with pytest.raises(ValueError, match="not supported with adapters"):
            validate_config(cfg)

    def test_rejects_expert_parallel(self, min_base_cfg):
        from axolotl.integrations.base import PluginManager

        plugin = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
        manager = PluginManager.get_instance()
        manager.register(plugin)
        try:
            cfg = (
                DictDefault(
                    tensor_parallel_size=2, expert_parallel_size=2, plugins=[plugin]
                )
                | min_base_cfg
            )
            with pytest.raises(ValueError, match="expert_parallel_size"):
                validate_config(cfg)
        finally:
            manager.plugins.pop(plugin, None)

    def test_allows_full_parameter(self, min_base_cfg):
        cfg = DictDefault(tensor_parallel_size=2) | min_base_cfg
        validate_config(cfg)

    def test_disables_cpu_ram_efficient_loading(self, min_base_cfg):
        cfg = (
            DictDefault(
                tensor_parallel_size=2,
                fsdp_version=2,
                fsdp_config={"cpu_ram_efficient_loading": True},
            )
            | min_base_cfg
        )
        out = validate_config(cfg)
        assert out.fsdp_config.cpu_ram_efficient_loading is False
