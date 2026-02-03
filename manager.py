"""
MOEManager: Global singleton for MoE loss aggregation, telemetry, and control.

Phase 1: Routing event collection, snapshots, alerts
Phase 2: Pressure controller, lens warping, decision logging
"""

from typing import List, Optional, Set, Dict, TYPE_CHECKING
import torch

# Type hints for chronomoe types (avoid circular imports at runtime)
if TYPE_CHECKING:
    from chronomoe.events import RoutingEvent
    from chronomoe.io import TelemetryWriter
    from chronomoe.lens import ChronoLens
    from chronomoe.controller import Controller, ControlConfig


class MOEManager:
    """
    Wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model.

    Phase 1: Telemetry - routing events, snapshots, alerts
    Phase 2: Governance - controller, lenses, decisions
    """

    def __init__(self):
        # Existing: auxiliary loss tracking
        self.aux_loss: List[torch.Tensor] = []
        self.router_z_loss: List[torch.Tensor] = []
        self.lens_aux_loss: List[torch.Tensor] = []  # Phase 2: lens training signal

        # Phase 1: Telemetry
        self.routing_events: List['RoutingEvent'] = []
        self.telemetry_writer: Optional['TelemetryWriter'] = None
        self.current_step: int = 0
        self.run_id: str = ""
        self.mode: str = "TRAIN"  # "TRAIN" or "INFER"

        # Phase 1: Alert history for persistent alert tracking
        self.alert_history: Dict[int, Dict[str, int]] = {}

        # Track number of MoE layers for validation
        self._expected_moe_layers: Optional[int] = None
        self._seen_layer_ids: Set[int] = set()

        # Phase 2: Controller and lenses
        self.controller: Optional['Controller'] = None
        self.lenses: Dict[int, 'ChronoLens'] = {}
        self._controller_enabled: bool = False

    # -------------------------------------------------------------------------
    # Existing methods: auxiliary loss tracking
    # -------------------------------------------------------------------------

    def reset_aux_loss(self) -> None:
        self.aux_loss = []

    def reset_router_z_loss(self) -> None:
        self.router_z_loss = []

    def reset_lens_aux_loss(self) -> None:
        self.lens_aux_loss = []

    def add_aux_loss(self, loss: torch.Tensor) -> None:
        self.aux_loss.append(loss)

    def add_router_z_loss(self, loss: torch.Tensor) -> None:
        self.router_z_loss.append(loss)

    def add_lens_aux_loss(self, loss: torch.Tensor) -> None:
        self.lens_aux_loss.append(loss)

    def aggregate_aux_loss(self) -> torch.Tensor:
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self) -> torch.Tensor:
        return sum(self.router_z_loss)

    def aggregate_lens_aux_loss(self) -> torch.Tensor:
        if not self.lens_aux_loss:
            return torch.tensor(0.0)
        return sum(self.lens_aux_loss)

    # -------------------------------------------------------------------------
    # Phase 1: Telemetry methods
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
        from chronomoe.io import TelemetryWriter

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

        from chronomoe.events import RoutingEvent

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
    # Phase 2: Controller and lens methods
    # -------------------------------------------------------------------------

    def initialize_controller(
        self,
        n_layers: int,
        n_experts_per_layer: List[int],
        config: Optional['ControlConfig'] = None,
        output_dir: str = "outputs",
    ) -> None:
        """
        Initialize Phase 2 controller.

        Args:
            n_layers: Number of MoE layers
            n_experts_per_layer: List of expert counts per layer
            config: Controller hyperparameters (uses defaults if None)
            output_dir: Base directory for outputs
        """
        from chronomoe.controller import Controller, ControlConfig

        self.controller = Controller(
            n_layers=n_layers,
            n_experts_per_layer=n_experts_per_layer,
            config=config or ControlConfig(),
            output_dir=output_dir,
        )
        self.controller.initialize(self.run_id)
        self._controller_enabled = True
        print(f"ChronoMoE Phase 2 controller initialized")

    def is_controller_enabled(self) -> bool:
        """Check if Phase 2 controller is active."""
        return self._controller_enabled and self.controller is not None

    def register_lens(self, layer_id: int, lens: 'ChronoLens') -> None:
        """
        Register a lens for a layer.

        Args:
            layer_id: MoE layer index
            lens: ChronoLens instance
        """
        self.lenses[layer_id] = lens

    def apply_lens(self, x: torch.Tensor, layer_id: int) -> torch.Tensor:
        """
        Apply the lens to router input x.

        Phase 1: Identity (no warp)
        Phase 2: Low-rank warp gated by controller pressure

        Args:
            x: Router input tensor (e.g., [B, T, D])
            layer_id: MoE layer index

        Returns:
            Transformed tensor with same shape as x
        """
        if layer_id in self.lenses:
            return self.lenses[layer_id](x)
        return x  # No lens registered, return unchanged

    def update_controller(self, snapshot: 'SystemSnapshot') -> Optional[List]:
        """
        Update controller from snapshot and set lens scales.

        Call this at each eval checkpoint after creating the snapshot.

        Args:
            snapshot: SystemSnapshot with layer metrics

        Returns:
            List of ControlDecision logs, or None if controller not enabled
        """
        if not self.is_controller_enabled():
            return None

        decisions = self.controller.update(snapshot, self.lenses)

        # Log pressure/scale for each layer
        for decision in decisions:
            pressure = decision.computed['pressure']
            scale = decision.actuator['lens_scale']
            if pressure > 0.01:  # Only log if there's meaningful pressure
                print(f"  Layer {decision.layer_id}: pressure={pressure:.3f}, lens_scale={scale:.4f}")

        return decisions

    def get_lens_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get all lens parameters for optimizer.

        Returns:
            List of lens parameters
        """
        params = []
        for lens in self.lenses.values():
            params.extend(lens.parameters())
        return params

    def get_controller_state(self, layer_id: int):
        """Get current control state for a layer."""
        if self.controller:
            return self.controller.get_state(layer_id)
        return None

    def load_prior_state(self, prior_path: str) -> bool:
        """
        Load persisted control state as prior for Clock 3 persistence.

        This injects learned state (harm_backoff, mode_scores, pressure)
        into the controller, allowing it to start with "memory" of
        what worked in a previous run.

        Args:
            prior_path: Path to pickled prior state file

        Returns:
            True if prior was loaded successfully
        """
        import pickle
        from pathlib import Path

        if not self.is_controller_enabled():
            print("WARNING: Cannot load prior state - controller not enabled")
            return False

        path = Path(prior_path)
        if not path.exists():
            print(f"WARNING: Prior state file not found: {prior_path}")
            return False

        try:
            with open(path, 'rb') as f:
                layer_states = pickle.load(f)

            # Inject into controller states
            for layer_id, prior in layer_states.items():
                if layer_id in self.controller.states:
                    state = self.controller.states[layer_id]
                    # Load persisted values
                    if 'harm_backoff' in prior:
                        state.harm_backoff = prior['harm_backoff']
                    if 'mode_scores' in prior:
                        state.mode_scores = prior['mode_scores']
                    if 'pressure' in prior:
                        state.pressure = prior['pressure']
                    if 'active_mode' in prior:
                        state.active_mode = prior['active_mode']
                    # Critical for harm detection continuity
                    if 'prev_top2' in prior:
                        state.prev_top2 = prior['prev_top2']
                    if 'prev_scale' in prior:
                        state.prev_scale = prior['prev_scale']

            print(f"Clock 3: Loaded prior state for {len(layer_states)} layers from {prior_path}")
            return True

        except Exception as e:
            print(f"WARNING: Failed to load prior state: {e}")
            return False


# Global singleton instance
MANAGER = MOEManager()
