from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TimeSeriesMeta:
    """Metadata for a synthetic time-series dataset.

    Attributes:
        name: Human-readable dataset name.
        domain: Domain where the signal naturally appears.
        description: Short semantic description of the signal.
        pattern_type: Primary pattern label for the series.
        length: Number of points in the generated series.
        sampling_rate: Sampling cadence label (for example, "hourly").
        sampling_info: Additional sampling context.
        known_anomaly_indices: Indices with intentional anomalies or regime changes.
    """

    name: str
    domain: str
    description: str
    pattern_type: str
    length: int
    sampling_rate: str
    sampling_info: str
    known_anomaly_indices: list[int]


@dataclass(slots=True)
class TimeSeriesDataset:
    """Container for a synthetic time-series dataset and its metadata."""

    data_id: str
    values: list[float]
    metadata: TimeSeriesMeta


@dataclass(slots=True)
class _SeriesBlueprint:
    name: str
    domain: str
    description: str
    sampling_rate: str
    pattern_type: str


@dataclass(slots=True)
class TimeSeriesDataFactory:
    """Factory for deterministic synthetic time-series benchmark datasets.

    The factory yields diverse datasets across trend, seasonality, anomaly, drift,
    and regime-shift behaviors. All generation is deterministic under a fixed seed.
    """

    seed: int = 42
    min_length: int = 50
    max_length: int = 200

    def generate_datasets(self, num_datasets: int = 20) -> list[TimeSeriesDataset]:
        """Generate deterministic synthetic time-series datasets.

        Args:
            num_datasets: Number of datasets to produce.

        Returns:
            List of synthetic datasets with values and metadata.
        """

        if num_datasets <= 0:
            return []

        rng = np.random.default_rng(self.seed)
        blueprints = self._build_blueprints()
        datasets: list[TimeSeriesDataset] = []

        for index in range(num_datasets):
            blueprint = blueprints[index % len(blueprints)]
            length = int(rng.integers(self.min_length, self.max_length + 1))
            series_seed = self.seed + (index + 1) * 9_973
            values, anomaly_indices = self._generate_series(
                pattern_type=blueprint.pattern_type,
                length=length,
                seed=series_seed,
            )
            data_id = f"timeseries-{index:03d}"
            metadata = TimeSeriesMeta(
                name=f"{blueprint.name}-{index:02d}",
                domain=blueprint.domain,
                description=blueprint.description,
                pattern_type=blueprint.pattern_type,
                length=length,
                sampling_rate=blueprint.sampling_rate,
                sampling_info=f"uniform {blueprint.sampling_rate} intervals",
                known_anomaly_indices=anomaly_indices,
            )
            datasets.append(
                TimeSeriesDataset(
                    data_id=data_id,
                    values=[float(round(value, 4)) for value in values],
                    metadata=metadata,
                )
            )

        return datasets

    def _build_blueprints(self) -> list[_SeriesBlueprint]:
        return [
            _SeriesBlueprint(
                name="RetailWeeklyRevenue",
                domain="retail",
                description="Weekly store revenue with sustained growth.",
                sampling_rate="weekly",
                pattern_type="linear_trend",
            ),
            _SeriesBlueprint(
                name="CityPowerDemand",
                domain="energy",
                description="Electric demand with repeating daily cycles.",
                sampling_rate="hourly",
                pattern_type="seasonal",
            ),
            _SeriesBlueprint(
                name="MobileTrafficLoad",
                domain="telecom",
                description="Traffic load with trend and periodic weekly effects.",
                sampling_rate="hourly",
                pattern_type="trend_seasonal",
            ),
            _SeriesBlueprint(
                name="EquityMidcapPrice",
                domain="finance",
                description="Asset prices evolving as a noisy random walk.",
                sampling_rate="daily",
                pattern_type="random_walk",
            ),
            _SeriesBlueprint(
                name="FactoryOutputLineA",
                domain="manufacturing",
                description="Production level with abrupt regime shifts.",
                sampling_rate="daily",
                pattern_type="step_function",
            ),
            _SeriesBlueprint(
                name="ServerLatencyP95",
                domain="infrastructure",
                description="Mostly stable latency with sparse sharp spikes.",
                sampling_rate="minute",
                pattern_type="spike_anomaly",
            ),
            _SeriesBlueprint(
                name="SensorCalibrationBias",
                domain="iot",
                description="Sensor baseline that drifts upward over time.",
                sampling_rate="minute",
                pattern_type="gradual_drift",
            ),
            _SeriesBlueprint(
                name="UrbanMobilityFlow",
                domain="transport",
                description="Traffic flow from multiple overlapping rhythms.",
                sampling_rate="hourly",
                pattern_type="multimodal",
            ),
            _SeriesBlueprint(
                name="HospitalAdmissions",
                domain="healthcare",
                description="Admission counts with long-term trend.",
                sampling_rate="daily",
                pattern_type="linear_trend",
            ),
            _SeriesBlueprint(
                name="EcommerceSessionRate",
                domain="web",
                description="User sessions showing clear periodic behavior.",
                sampling_rate="hourly",
                pattern_type="seasonal",
            ),
            _SeriesBlueprint(
                name="CloudCpuUtilization",
                domain="cloud",
                description="CPU utilization with trend and recurring cycle.",
                sampling_rate="minute",
                pattern_type="trend_seasonal",
            ),
            _SeriesBlueprint(
                name="CommoditySpotPrice",
                domain="economics",
                description="Commodity prices with random-walk dynamics.",
                sampling_rate="daily",
                pattern_type="random_walk",
            ),
            _SeriesBlueprint(
                name="FulfillmentThroughput",
                domain="logistics",
                description="Throughput with step-like operational changes.",
                sampling_rate="hourly",
                pattern_type="step_function",
            ),
            _SeriesBlueprint(
                name="FraudSignalScore",
                domain="security",
                description="Mostly normal stream with anomaly spikes.",
                sampling_rate="minute",
                pattern_type="spike_anomaly",
            ),
            _SeriesBlueprint(
                name="ModelDriftMetric",
                domain="mlops",
                description="Model drift score with slow baseline shift.",
                sampling_rate="daily",
                pattern_type="gradual_drift",
            ),
            _SeriesBlueprint(
                name="WindFarmOutput",
                domain="renewables",
                description="Power output combining multiple frequencies.",
                sampling_rate="hourly",
                pattern_type="multimodal",
            ),
            _SeriesBlueprint(
                name="WarehouseEnergyUse",
                domain="operations",
                description="Energy use with directional trend.",
                sampling_rate="daily",
                pattern_type="linear_trend",
            ),
            _SeriesBlueprint(
                name="AirportPassengerFlow",
                domain="aviation",
                description="Passenger flow with cyclical periodicity.",
                sampling_rate="hourly",
                pattern_type="seasonal",
            ),
            _SeriesBlueprint(
                name="EdgeRequestVolume",
                domain="networking",
                description="Request volume with trend and seasonal variation.",
                sampling_rate="minute",
                pattern_type="trend_seasonal",
            ),
            _SeriesBlueprint(
                name="InsuranceClaimIndex",
                domain="insurance",
                description="Claim index exhibiting regime changes.",
                sampling_rate="weekly",
                pattern_type="step_function",
            ),
        ]

    def _generate_series(
        self,
        pattern_type: str,
        length: int,
        seed: int,
    ) -> tuple[np.ndarray, list[int]]:
        rng = np.random.default_rng(seed)
        time = np.arange(length, dtype=float)

        if pattern_type == "linear_trend":
            slope = rng.uniform(0.08, 0.35) * (1 if rng.random() > 0.35 else -1)
            baseline = rng.uniform(20.0, 80.0)
            noise = rng.normal(0.0, 1.5, length)
            values = baseline + slope * time + noise
            return values, []

        if pattern_type == "seasonal":
            period = int(rng.integers(8, 20))
            amplitude = rng.uniform(6.0, 15.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            baseline = rng.uniform(40.0, 90.0)
            noise = rng.normal(0.0, 1.2, length)
            values = (
                baseline
                + amplitude * np.sin((2.0 * np.pi * time / period) + phase)
                + noise
            )
            return values, []

        if pattern_type == "trend_seasonal":
            period = int(rng.integers(10, 24))
            amplitude = rng.uniform(4.0, 10.0)
            trend = rng.uniform(0.03, 0.22)
            baseline = rng.uniform(30.0, 70.0)
            noise = rng.normal(0.0, 1.3, length)
            seasonal_component = amplitude * np.sin(2.0 * np.pi * time / period)
            values = baseline + trend * time + seasonal_component + noise
            return values, []

        if pattern_type == "random_walk":
            drift = rng.uniform(-0.08, 0.18)
            innovations = rng.normal(drift, 1.8, length)
            start = rng.uniform(70.0, 130.0)
            values = start + np.cumsum(innovations)
            return values, []

        if pattern_type == "step_function":
            first_cp = int(length * 0.33)
            second_cp = int(length * 0.66)
            base = np.full(length, rng.uniform(25.0, 55.0))
            step_1 = rng.uniform(8.0, 20.0) * (1 if rng.random() > 0.4 else -1)
            step_2 = rng.uniform(8.0, 20.0) * (1 if rng.random() > 0.4 else -1)
            base[first_cp:] += step_1
            base[second_cp:] += step_2
            noise = rng.normal(0.0, 1.1, length)
            values = base + noise
            return values, [first_cp, second_cp]

        if pattern_type == "spike_anomaly":
            baseline = rng.uniform(45.0, 75.0)
            trend = rng.uniform(-0.05, 0.06)
            values = baseline + trend * time + rng.normal(0.0, 1.0, length)
            candidate_indices = np.arange(5, length - 5)
            spikes = rng.choice(candidate_indices, size=3, replace=False)
            magnitudes = rng.uniform(12.0, 25.0, size=3)
            signs = np.where(rng.random(3) > 0.2, 1.0, -1.0)
            values[spikes] += magnitudes * signs
            return values, sorted(int(index) for index in spikes)

        if pattern_type == "gradual_drift":
            drift_start = int(length * 0.4)
            drift = np.maximum(0.0, time - drift_start) * rng.uniform(0.08, 0.22)
            baseline = rng.uniform(25.0, 60.0)
            noise = rng.normal(0.0, 1.0, length)
            values = baseline + drift + noise
            return values, [drift_start]

        if pattern_type == "multimodal":
            period_1 = int(rng.integers(7, 14))
            period_2 = int(rng.integers(20, 40))
            amp_1 = rng.uniform(4.0, 9.0)
            amp_2 = rng.uniform(5.0, 12.0)
            baseline = rng.uniform(35.0, 80.0)
            component_1 = amp_1 * np.sin(2.0 * np.pi * time / period_1)
            component_2 = amp_2 * np.sin(
                2.0 * np.pi * time / period_2 + rng.uniform(0.0, np.pi)
            )
            values = baseline + component_1 + component_2 + rng.normal(0.0, 1.2, length)
            return values, []

        raise ValueError(f"Unsupported pattern type: {pattern_type}")
