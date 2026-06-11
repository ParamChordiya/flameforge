"""A panel showing live training metrics (loss, LR, grad norm, epoch, best)."""

from __future__ import annotations

from textual.widgets import Static

from flameforge.training.base import TrainingMetrics


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class MetricsPanel(Static):
    """Displays the latest scalar training metrics in an aligned block."""

    def __init__(self) -> None:
        super().__init__("Metrics will appear once training starts.", id="metrics-panel")

    def update_metrics(self, m: TrainingMetrics) -> None:
        """Re-render the panel from a metrics snapshot.

        Args:
            m: The latest :class:`TrainingMetrics`.
        """
        best = f"{m.best_loss:.4f}" if m.best_loss is not None else "—"
        evals = f"{m.eval_loss:.4f}" if m.eval_loss is not None else "—"
        ckpt = f"step {m.last_checkpoint_step}" if m.last_checkpoint_step else "—"
        lines = [
            f"Loss:        {m.loss:.4f}",
            f"Eval loss:   {evals}",
            f"Best loss:   {best}",
            f"LR:          {m.learning_rate:.2e}",
            f"Grad norm:   {m.grad_norm:.3f}",
            f"Epoch:       {m.epoch:.2f}",
            f"Tokens/s:    {m.tokens_per_sec:,.0f}",
            f"Elapsed:     {_fmt_duration(m.elapsed_sec)}",
            f"ETA:         {_fmt_duration(m.eta_sec)}",
            f"Checkpoint:  {ckpt}",
        ]
        self.update("\n".join(lines))
