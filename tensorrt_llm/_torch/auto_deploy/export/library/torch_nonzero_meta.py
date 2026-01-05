"""Patch to enable torch.nonzero() meta implementation during export."""

import torch.fx.experimental._config as fx_config

from ..interface import BaseExportPatch, ExportPatchRegistry


@ExportPatchRegistry.register("torch_nonzero_meta")
class TorchNonzeroMetaPatch(BaseExportPatch):
    """Enable torch.nonzero() meta implementation during export.

    This patch sets torch.fx.experimental._config.meta_nonzero_assume_all_nonzero
    to True, which allows torch.export to handle torch.nonzero() calls on meta
    tensors by assuming all elements are non-zero. This is necessary for exporting
    models that use nonzero or torch.where(condition) operations.
    """

    def _apply_patch(self):
        """Apply the torch.nonzero meta patch."""
        # Store original config value
        self.original_values["meta_nonzero"] = fx_config.meta_nonzero_assume_all_nonzero
        # Enable meta nonzero during export
        fx_config.meta_nonzero_assume_all_nonzero = True

    def _revert_patch(self):
        """Revert the torch.nonzero meta patch."""
        # Restore original config value
        fx_config.meta_nonzero_assume_all_nonzero = self.original_values["meta_nonzero"]

