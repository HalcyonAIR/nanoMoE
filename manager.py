"""
MOEManager: Global singleton for MoE loss aggregation and telemetry.

Phase 1 additions:
- Routing event collection and flushing
- Lens interface (identity by default)
- Step/mode tracking for telemetry
"""

from typing import List, Optional, Set, Dict, TYPE_CHECKING
import torch

# Type hints for chrono types (avoid circular imports at runtime)
if TYPE_CHECKING:
    from chrono.events import RoutingEvent
    from chrono.io import TelemetryWriter
    from chrono.lens import ChronoLens, LensState


class MOEManager:
    """
    Wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model.

    Phase 1 telemetry additions:
    - Routing events collected per (step, layer)
    - Lens interface for future geometry warping
    - Step and mode tracking
    """

    def __init__(self):
        # Existing: auxiliary loss tracking
        self.aux_loss: List[torch.Tensor] = []
        self.router_z_loss: List[torch.Tensor] = []

        # NEW: Phase 1 telemetry (tightened type hints)
        self.routing_events: List['RoutingEvent'] = []
        self.telemetry_writer: Optional['TelemetryWriter'] = None
        self.current_step: int = 0
        self.run_id: str = ""
        self.mode: str = "TRAIN"  # "TRAIN" or "INFER"

        # NEW: Lens interface (Phase 1: identity)
        self._lens: Optional['ChronoLens'] = None
        self._lens_state: Optional['LensState'] = None

        # NEW: Alert history for persistent alert tracking
        self.alert_history: Dict[int, Dict[str, int]] = {}

        # Track number of MoE layers for validation
        self._expected_moe_layers: Optional[int] = None
        self._seen_layer_ids: Set[int] = set()

    # -------------------------------------------------------------------------
    # Existing methods: auxiliary loss tracking
    # -------------------------------------------------------------------------

    def reset_aux_loss(self) -> None:
        self.aux_loss = []

    def reset_router_z_loss(self) -> None:
        self.router_z_loss = []

    def add_aux_loss(self, loss: torch.Tensor) -> None:
        self.aux_loss.append(loss)

    def add_router_z_loss(self, loss: torch.Tensor) -> None:
        self.router_z_loss.append(loss)

    def aggregate_aux_loss(self) -> torch.Tensor:
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self) -> torch.Tensor:
        return sum(self.router_z_loss)

    # -------------------------------------------------------------------------
    # NEW: Phase 1 telemetry methods
    # -------------------------------------------------------------------------

    def initialize_telemetry(
        self,
        run_id: str,
        output_dir: str = "outputs",
        n_moe_layers: Optional[int] = None,
    ) -> None:
        """
        Initialize telemetry writer for this run.

        Args:
            run_id: Unique identifier for this run
            output_dir: Base directory for outputs
            n_moe_layers: Expected number of MoE layers (for validation)
        """
        from chrono.io import TelemetryWriter

        self.run_id = run_id
        self.telemetry_writer = TelemetryWriter(run_id, output_dir)
        self.routing_events = []
        self.alert_history = {}
        self._expected_moe_layers = n_moe_layers
        self._seen_layer_ids = set()

    def is_telemetry_enabled(self) -> bool:
        """Check if telemetry is active."""
        return self.telemetry_writer is not None

    def add_routing_event(
        self,
        layer_id: int,
        used_capacity: torch.Tensor,
        n_experts: int,
        top_k: int,
    ) -> None:
        """
        Called from Router.forward() to log dispatch.

        This is the golden hook point - captures actual dispatch after
        capacity constraints are applied.

        Args:
            layer_id: MoE layer index
            used_capacity: Tensor of shape [n_experts] with token counts
            n_experts: Number of experts in this layer
            top_k: Number of experts selected per token
        """
        if not self.is_telemetry_enabled():
            return  # Telemetry not enabled, skip silently

        from chrono.events import RoutingEvent

        # Track seen layer_ids for validation (reset happens in set_step)
        self._seen_layer_ids.add(layer_id)

        event = RoutingEvent.from_router_output(
            run_id=self.run_id,
            step=self.current_step,
            mode=self.mode,
            layer_id=layer_id,
            used_capacity=used_capacity,
            n_experts=n_experts,
            top_k=top_k,
        )
        self.routing_events.append(event)

    def flush_routing_events(self) -> int:
        """
        Write accumulated events to disk and clear buffer.

        Returns:
            Number of events flushed
        """
        if not self.is_telemetry_enabled():
            return 0

        count = len(self.routing_events)
        if count > 0:
            self.telemetry_writer.flush_events(self.routing_events)
            self.routing_events = []
        return count

    def set_step(self, step: int) -> None:
        """
        Update current step for event tracking.

        Also validates and resets layer_id tracking from the previous step.
        This ensures layer_id validation happens at step boundaries.
        """
        # Validate layer_ids from previous step before resetting
        if self._seen_layer_ids and self._expected_moe_layers is not None:
            expected = set(range(self._expected_moe_layers))
            if self._seen_layer_ids != expected:
                missing = expected - self._seen_layer_ids
                extra = self._seen_layer_ids - expected
                if missing:
                    print(f"WARNING: Step {self.current_step} missing layer_ids: {missing}")
                if extra:
                    print(f"WARNING: Step {self.current_step} unexpected layer_ids: {extra}")

        # Reset for new step
        self._seen_layer_ids = set()
        self.current_step = step

    def set_mode(self, mode: str) -> None:
        """Set TRAIN or INFER mode."""
        assert mode in ("TRAIN", "INFER"), f"Invalid mode: {mode}"
        self.mode = mode

    def get_routing_events(self) -> List['RoutingEvent']:
        """Get current routing events (for snapshot creation)."""
        return self.routing_events

    # -------------------------------------------------------------------------
    # NEW: Lens interface (Phase 1: identity by default)
    # -------------------------------------------------------------------------

    def attach_lens(self, lens: Optional['ChronoLens']) -> None:
        """
        Attach a lens module.

        Args:
            lens: ChronoLens instance or None (reverts to IdentityLens)
        """
        from chrono.lens import IdentityLens

        if lens is None:
            self._lens = IdentityLens()
        else:
            self._lens = lens

    def apply_lens(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        """
        Apply the lens to router input x.

        Phase 1 default is identity, so this call is safe to insert
        without changing behavior.

        Args:
            x: Router input tensor (e.g., [B, T, D])
            layer_id: MoE layer index

        Returns:
            Transformed tensor with same shape as x
        """
        # Lazy initialization of lens
        if self._lens is None:
            from chrono.lens import IdentityLens
            self._lens = IdentityLens()

        # Build/update lens state
        from chrono.lens import LensState

        self._lens_state = LensState(
            step=self.current_step,
            mode=self.mode,
            pressure=0.0,  # Phase 1: placeholder
            heat=0.0,      # Phase 1: placeholder
            forgetting=0.0,  # Phase 1: placeholder
            layer_metrics=None,  # Phase 1: placeholder
            meta=None,
        )

        return self._lens(x, self._lens_state, layer_id)


# Global singleton instance
MANAGER = MOEManager()
