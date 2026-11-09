"""Resolved-config composition only; not connected to any pipeline entrypoint."""
from youtube_pipeline.configuration.models import RunConfig
from youtube_pipeline.polarization_signals import (
    PolarizationTemporalSignal, create_polarization_temporal_signal,
)


def create_configured_polarization_signal(config: RunConfig) -> PolarizationTemporalSignal:
    if not isinstance(config, RunConfig):
        raise TypeError("config must be RunConfig.")
    if config.signals is None or config.signals.polarization is None or config.polarization is None:
        raise ValueError("A polarization temporal definition and measure parameters are required.")
    temporal = config.signals.polarization
    return create_polarization_temporal_signal(temporal, {
        name: config.polarization.config_for(name) for name in temporal.enabled_measures
    })
