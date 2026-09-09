"""Swarm-Tech drone stack: hexagonal cooperative lifting + RT-DETR aerial vision."""

__version__ = "0.1.0"

from .config import (SwarmConfig, DroneSpec, FormationConfig, LiftConfig,
                     CommsConfig, VisionConfig)
from .formation import HexFormation, assign_slots
from .load_lift import LiftPlanner, LiftPhase
from .agent import DroneAgent
from .swarm import SwarmCoordinator

__all__ = [
    "SwarmConfig", "DroneSpec", "FormationConfig", "LiftConfig",
    "CommsConfig", "VisionConfig",
    "HexFormation", "assign_slots",
    "LiftPlanner", "LiftPhase",
    "DroneAgent", "SwarmCoordinator",
]
