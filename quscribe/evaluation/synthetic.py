"""
Synthetic multi-style dataset: random standard-gate circuits drawn in
several styles.

Every image comes with exact ground truth, which makes the dataset suitable for
development, for evaluation, and for the LoRA fine-tuning stretch goal.
"""

import logging
import random
from dataclasses import dataclass
from math import pi
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import (
    CCXGate,
    CPhaseGate,
    CRZGate,
    CXGate,
    CZGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SwapGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)

from quscribe.core.qasm import dump_qasm
from quscribe.core.rendering import DiagramStyle, render_circuit
from quscribe.errors import ConfigurationError
from quscribe.evaluation.dataset import (
    MANIFEST_FILENAME,
    BenchmarkItem,
    write_manifest,
)

logger = logging.getLogger(__name__)

IMAGES_DIRECTORY_NAME = "images"
MAX_SUPPORTED_QUBITS = 10  # Project scope: circuits of up to about 10 qubits.
SYNTHETIC_SOURCE = "synthetic"

# Angles that commonly appear in papers, so rendered labels
# look realistic (e.g. "π/4").
COMMON_ANGLES = (pi / 8, pi / 4, pi / 2, 3 * pi / 4, pi, -pi / 4, -pi / 2)


@dataclass(frozen=True)
class GateTemplate:
    """A gate type the generator can place.

    Attributes:
        gate_class: The Qiskit gate class.
        qubit_count: Number of qubits the gate acts on.
        parameter_count: Number of angle parameters.
    """

    gate_class: type[Gate]
    qubit_count: int
    parameter_count: int = 0


STANDARD_GATE_TEMPLATES = (
    GateTemplate(HGate, 1),
    GateTemplate(XGate, 1),
    GateTemplate(YGate, 1),
    GateTemplate(ZGate, 1),
    GateTemplate(SGate, 1),
    GateTemplate(SdgGate, 1),
    GateTemplate(TGate, 1),
    GateTemplate(TdgGate, 1),
    GateTemplate(RXGate, 1, 1),
    GateTemplate(RYGate, 1, 1),
    GateTemplate(RZGate, 1, 1),
    GateTemplate(CXGate, 2),
    GateTemplate(CZGate, 2),
    GateTemplate(SwapGate, 2),
    GateTemplate(CPhaseGate, 2, 1),
    GateTemplate(CRZGate, 2, 1),
    GateTemplate(CCXGate, 3),
)


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Parameters of a synthetic dataset.

    Attributes:
        circuit_count: Number of circuits to generate.
        min_qubits: Smallest circuit width.
        max_qubits: Largest circuit width (at most 10).
        min_gates: Fewest gates per circuit.
        max_gates: Most gates per circuit.
        measurement_probability: Chance that a circuit ends with measurements.
        styles: Drawing styles, assigned round-robin for balanced coverage.
        seed: Random seed, for reproducibility.
    """

    circuit_count: int = 100
    min_qubits: int = 2
    max_qubits: int = 6
    min_gates: int = 3
    max_gates: int = 20
    measurement_probability: float = 0.5
    styles: tuple[DiagramStyle, ...] = tuple(DiagramStyle)
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Args:
            None.

        Returns:
            None.

        Raises:
            ConfigurationError: If any value is out of range.
        """
        is_valid = (
            self.circuit_count >= 1
            and 1 <= self.min_qubits <= self.max_qubits <= MAX_SUPPORTED_QUBITS
            and 1 <= self.min_gates <= self.max_gates
            and 0.0 <= self.measurement_probability <= 1.0
            and bool(self.styles)
        )
        if not is_valid:
            raise ConfigurationError(
                f"Invalid synthetic dataset configuration: {self}. Qubits "
                f"must be within 1..{MAX_SUPPORTED_QUBITS}, minimums must not "
                f"exceed maximums, and at least one style is needed."
            )


class RandomCircuitGenerator:
    """
    Builds random circuits from a gate catalogue."""

    def __init__(
        self,
        random_source: random.Random,
        gate_templates: tuple[GateTemplate, ...] = STANDARD_GATE_TEMPLATES,
    ) -> None:
        """Create a generator.

        Args:
            random_source: Seeded random number generator.
            gate_templates: Gate types to choose from.
        """
        self._random_source = random_source
        self._gate_templates = gate_templates

    def generate(
        self,
        qubit_count: int,
        gate_count: int,
        add_measurements: bool
    ) -> QuantumCircuit:
        """
        Build one random circuit.

        Args:
            qubit_count: Circuit width.
            gate_count: Number of gates to place.
            add_measurements: Whether to measure every qubit at the end.

        Returns:
            QuantumCircuit: The random circuit.
        """
        circuit = QuantumCircuit(qubit_count)
        eligible_templates = [
            template for template in self._gate_templates
            if template.qubit_count <= qubit_count
        ]

        for gate_position in range(gate_count):
            template = self._random_source.choice(eligible_templates)
            target_qubits = self._random_source.sample(
                range(qubit_count),
                template.qubit_count
            )
            angles = [
                self._random_source.choice(COMMON_ANGLES)
                for parameter_index in range(template.parameter_count)
            ]
            circuit.append(template.gate_class(*angles), target_qubits)

        if add_measurements:
            circuit.measure_all()

        return circuit


class SyntheticDatasetGenerator:
    """
    Generates rendered circuit images with exact reference circuits.
    """

    def __init__(self, config: SyntheticDatasetConfig) -> None:
        """
        Create a dataset generator.

        Args:
            config: The dataset parameters.

        Returns:
            None.
        """
        self._config = config

    def generate(self, output_directory: Path) -> list[BenchmarkItem]:
        """
        Generate the dataset: images, reference circuits and a manifest.

        Args:
            output_directory: Where to write ``images/`` and ``manifest.jsonl``

        Returns:
            list[BenchmarkItem]: The generated items, also written to
                                    the manifest.

        Raises:
            RenderingError: If a circuit cannot be drawn.
        """
        config = self._config
        images_directory = output_directory / IMAGES_DIRECTORY_NAME
        images_directory.mkdir(parents=True, exist_ok=True)
        random_source = random.Random(config.seed)
        circuit_generator = RandomCircuitGenerator(random_source)
        items: list[BenchmarkItem] = []

        for circuit_index in range(config.circuit_count):
            circuit = circuit_generator.generate(
                qubit_count=random_source.randint(
                    config.min_qubits,
                    config.max_qubits
                ),
                gate_count=random_source.randint(
                    config.min_gates,
                    config.max_gates
                ),
                add_measurements=(
                    random_source.random() < config.measurement_probability
                ),
            )
            style = config.styles[circuit_index % len(config.styles)]
            item_id = f"synthetic-{circuit_index:05d}"
            image_path = images_directory / f"{item_id}.png"
            image_path.write_bytes(render_circuit(circuit, style).png_bytes)
            items.append(BenchmarkItem(
                item_id=item_id,
                image_path=image_path,
                reference_qasm=dump_qasm(circuit),
                style=style.value,
                source=SYNTHETIC_SOURCE,
            ))
            logger.info("Generated %s (%s style)", item_id, style.value)

        write_manifest(items, output_directory / MANIFEST_FILENAME)

        return items
