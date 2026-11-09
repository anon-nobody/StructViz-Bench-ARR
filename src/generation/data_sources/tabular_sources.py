from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DatasetMeta:
    """Metadata that documents a synthetic tabular dataset."""

    name: str
    domain: str
    description: str
    column_descriptions: dict[str, str]


@dataclass(slots=True)
class TabularDataset:
    """Container for a generated dataframe and its metadata."""

    meta: DatasetMeta
    dataframe: pd.DataFrame


@dataclass(slots=True)
class TabularDataFactory:
    """Build deterministic synthetic tabular datasets for StructViz-Bench."""

    seed: int = 42

    def generate_all(self) -> list[TabularDataset]:
        """Generate all deterministic synthetic tabular datasets.

        Returns:
            List of 20 datasets spanning multiple real-world domains.
        """
        builders = [
            self._finance_stock_prices,
            self._finance_revenue_mix,
            self._healthcare_patient_vitals,
            self._healthcare_lab_results,
            self._education_student_scores,
            self._education_attendance,
            self._retail_daily_sales,
            self._retail_inventory,
            self._environment_weather,
            self._environment_pollution,
            self._sports_player_stats,
            self._sports_team_results,
            self._demographics_census,
            self._demographics_income,
            self._transportation_traffic,
            self._energy_grid_load,
            self._agriculture_crop_yield,
            self._manufacturing_quality,
            self._tourism_hotel_occupancy,
            self._ecommerce_campaign_performance,
        ]
        return [builder() for builder in builders]

    def _rng_for(self, dataset_name: str) -> np.random.Generator:
        seed_offset = sum(ord(char) for char in dataset_name)
        return np.random.default_rng(self.seed + seed_offset)

    def _month_dates(self, rows: int, start: str) -> pd.Series:
        return pd.Series(
            pd.date_range(start=start, periods=rows, freq="MS"), name="period"
        )

    def _week_dates(self, rows: int, start: str) -> pd.Series:
        return pd.Series(
            pd.date_range(start=start, periods=rows, freq="W"), name="week"
        )

    def _make_dataset(
        self,
        *,
        name: str,
        domain: str,
        description: str,
        dataframe: pd.DataFrame,
        column_descriptions: dict[str, str],
    ) -> TabularDataset:
        meta = DatasetMeta(
            name=name,
            domain=domain,
            description=description,
            column_descriptions=column_descriptions,
        )
        return TabularDataset(meta=meta, dataframe=dataframe)

    def _finance_stock_prices(self) -> TabularDataset:
        rows = 24
        rng = self._rng_for("finance_stock_prices")
        base = np.linspace(95, 138, rows) + rng.normal(0, 3.2, rows)
        data = pd.DataFrame(
            {
                "ticker": np.array(["ALP", "BET", "CRX", "DLT"])[
                    rng.integers(0, 4, rows)
                ],
                "period": self._month_dates(rows, "2023-01-01"),
                "close_price": np.round(base, 2),
                "trade_volume_k": rng.integers(800, 5200, rows),
                "daily_return_pct": np.round(rng.normal(0.4, 2.1, rows), 2),
                "market_segment": np.array(["growth", "value", "income"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="finance_stock_prices",
            domain="finance",
            description="Monthly stock activity for a set of listed companies.",
            dataframe=data,
            column_descriptions={
                "ticker": "Stock ticker symbol.",
                "period": "Month represented by the row.",
                "close_price": "End-of-month close price in USD.",
                "trade_volume_k": "Monthly traded volume in thousands of shares.",
                "daily_return_pct": "Average daily return percentage in that month.",
                "market_segment": "Portfolio segment assignment.",
            },
        )

    def _finance_revenue_mix(self) -> TabularDataset:
        rows = 20
        rng = self._rng_for("finance_revenue_mix")
        data = pd.DataFrame(
            {
                "business_unit": np.array(["cloud", "devices", "ads", "services"])[
                    rng.integers(0, 4, rows)
                ],
                "quarter_start": pd.date_range("2021-01-01", periods=rows, freq="QS"),
                "revenue_musd": np.round(rng.normal(320, 70, rows).clip(min=120), 2),
                "operating_margin_pct": np.round(
                    rng.normal(18, 6, rows).clip(min=2), 2
                ),
                "opex_musd": np.round(rng.normal(110, 22, rows).clip(min=60), 2),
                "region": np.array(["NA", "EU", "APAC", "LATAM"])[
                    rng.integers(0, 4, rows)
                ],
            }
        )
        return self._make_dataset(
            name="finance_revenue_mix",
            domain="finance",
            description="Quarterly revenue and margin observations by business unit.",
            dataframe=data,
            column_descriptions={
                "business_unit": "Revenue-generating business unit.",
                "quarter_start": "Quarter start date.",
                "revenue_musd": "Revenue in million USD.",
                "operating_margin_pct": "Operating margin percentage.",
                "opex_musd": "Operating expenses in million USD.",
                "region": "Primary region for the revenue mix.",
            },
        )

    def _healthcare_patient_vitals(self) -> TabularDataset:
        rows = 30
        rng = self._rng_for("healthcare_patient_vitals")
        data = pd.DataFrame(
            {
                "patient_group": np.array(["A", "B", "C", "D"])[
                    rng.integers(0, 4, rows)
                ],
                "week": self._week_dates(rows, "2024-01-07"),
                "heart_rate_bpm": np.round(rng.normal(76, 8, rows), 1),
                "systolic_bp": np.round(rng.normal(122, 12, rows), 1),
                "glucose_mg_dl": np.round(rng.normal(104, 18, rows), 1),
                "risk_band": np.array(["low", "medium", "high"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="healthcare_patient_vitals",
            domain="healthcare",
            description="Weekly aggregate patient vital signs from outpatient monitoring.",
            dataframe=data,
            column_descriptions={
                "patient_group": "An anonymized patient cohort.",
                "week": "Observation week.",
                "heart_rate_bpm": "Average heart rate in beats per minute.",
                "systolic_bp": "Average systolic blood pressure.",
                "glucose_mg_dl": "Average blood glucose level.",
                "risk_band": "Clinical risk grouping.",
            },
        )

    def _healthcare_lab_results(self) -> TabularDataset:
        rows = 18
        rng = self._rng_for("healthcare_lab_results")
        data = pd.DataFrame(
            {
                "clinic": np.array(["north", "south", "east", "west"])[
                    rng.integers(0, 4, rows)
                ],
                "test_date": pd.date_range("2024-02-01", periods=rows, freq="D"),
                "hemoglobin_g_dl": np.round(rng.normal(13.7, 1.4, rows), 2),
                "ldl_mg_dl": np.round(rng.normal(112, 28, rows).clip(min=55), 2),
                "creatinine_mg_dl": np.round(
                    rng.normal(1.0, 0.22, rows).clip(min=0.5), 2
                ),
                "lab_flag": np.array(["normal", "borderline", "critical"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="healthcare_lab_results",
            domain="healthcare",
            description="Routine laboratory panel results across clinics.",
            dataframe=data,
            column_descriptions={
                "clinic": "Clinic where the panel was processed.",
                "test_date": "Date of test panel.",
                "hemoglobin_g_dl": "Hemoglobin concentration.",
                "ldl_mg_dl": "Low-density lipoprotein value.",
                "creatinine_mg_dl": "Creatinine measurement.",
                "lab_flag": "Clinical interpretation of panel status.",
            },
        )

    def _education_student_scores(self) -> TabularDataset:
        rows = 26
        rng = self._rng_for("education_student_scores")
        data = pd.DataFrame(
            {
                "classroom": np.array(["9A", "9B", "10A", "10B"])[
                    rng.integers(0, 4, rows)
                ],
                "exam_date": pd.date_range("2024-03-15", periods=rows, freq="7D"),
                "math_score": np.round(
                    rng.normal(76, 11, rows).clip(min=35, max=100), 1
                ),
                "science_score": np.round(
                    rng.normal(74, 12, rows).clip(min=30, max=100), 1
                ),
                "reading_score": np.round(
                    rng.normal(79, 9, rows).clip(min=40, max=100), 1
                ),
                "support_program": np.array(["none", "tutoring", "lab"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="education_student_scores",
            domain="education",
            description="Assessment outcomes across school classrooms.",
            dataframe=data,
            column_descriptions={
                "classroom": "Student classroom section.",
                "exam_date": "Date of classroom assessment.",
                "math_score": "Average math score.",
                "science_score": "Average science score.",
                "reading_score": "Average reading score.",
                "support_program": "Instructional support track.",
            },
        )

    def _education_attendance(self) -> TabularDataset:
        rows = 22
        rng = self._rng_for("education_attendance")
        data = pd.DataFrame(
            {
                "school": np.array(["north_high", "central_high", "west_high"])[
                    rng.integers(0, 3, rows)
                ],
                "month": self._month_dates(rows, "2023-09-01"),
                "attendance_pct": np.round(
                    rng.normal(92, 4.8, rows).clip(min=72, max=100), 2
                ),
                "late_arrivals": rng.integers(8, 78, rows),
                "disciplinary_events": rng.integers(0, 16, rows),
                "term": np.array(["fall", "winter", "spring"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="education_attendance",
            domain="education",
            description="Monthly attendance and behavior indicators by school.",
            dataframe=data,
            column_descriptions={
                "school": "School identifier.",
                "month": "Month of attendance reporting.",
                "attendance_pct": "Attendance percentage.",
                "late_arrivals": "Number of late arrivals.",
                "disciplinary_events": "Count of recorded behavior incidents.",
                "term": "Academic term.",
            },
        )

    def _retail_daily_sales(self) -> TabularDataset:
        rows = 35
        rng = self._rng_for("retail_daily_sales")
        data = pd.DataFrame(
            {
                "store": np.array(["store_01", "store_02", "store_03", "store_04"])[
                    rng.integers(0, 4, rows)
                ],
                "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
                "transactions": rng.integers(90, 560, rows),
                "gross_sales_usd": np.round(
                    rng.normal(11800, 2200, rows).clip(min=5000), 2
                ),
                "discount_usd": np.round(rng.normal(740, 180, rows).clip(min=120), 2),
                "channel": np.array(["in_store", "online", "pickup"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="retail_daily_sales",
            domain="retail",
            description="Daily retail sales activity across channels.",
            dataframe=data,
            column_descriptions={
                "store": "Store location code.",
                "date": "Calendar date of sales record.",
                "transactions": "Number of completed transactions.",
                "gross_sales_usd": "Gross sales value in USD.",
                "discount_usd": "Discount value granted in USD.",
                "channel": "Primary shopping channel.",
            },
        )

    def _retail_inventory(self) -> TabularDataset:
        rows = 19
        rng = self._rng_for("retail_inventory")
        data = pd.DataFrame(
            {
                "category": np.array(["grocery", "apparel", "electronics", "home"])[
                    rng.integers(0, 4, rows)
                ],
                "snapshot_week": self._week_dates(rows, "2024-01-07"),
                "units_in_stock": rng.integers(120, 2200, rows),
                "units_sold": rng.integers(40, 980, rows),
                "backorder_units": rng.integers(0, 140, rows),
                "supplier_tier": np.array(["gold", "silver", "bronze"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="retail_inventory",
            domain="retail",
            description="Inventory and fulfillment snapshots for retail categories.",
            dataframe=data,
            column_descriptions={
                "category": "Product category.",
                "snapshot_week": "Week of inventory snapshot.",
                "units_in_stock": "Units physically available.",
                "units_sold": "Units sold during the snapshot period.",
                "backorder_units": "Units pending supplier fulfillment.",
                "supplier_tier": "Supplier relationship tier.",
            },
        )

    def _environment_weather(self) -> TabularDataset:
        rows = 28
        rng = self._rng_for("environment_weather")
        data = pd.DataFrame(
            {
                "city": np.array(["lakeside", "ridgeview", "seaport"])[
                    rng.integers(0, 3, rows)
                ],
                "date": pd.date_range("2024-04-01", periods=rows, freq="D"),
                "temperature_c": np.round(rng.normal(21, 6, rows), 1),
                "rainfall_mm": np.round(rng.gamma(2.2, 2.4, rows), 1),
                "humidity_pct": np.round(
                    rng.normal(61, 11, rows).clip(min=25, max=96), 1
                ),
                "weather_type": np.array(["sunny", "cloudy", "rainy", "stormy"])[
                    rng.integers(0, 4, rows)
                ],
            }
        )
        return self._make_dataset(
            name="environment_weather",
            domain="environment",
            description="Daily weather observations across cities.",
            dataframe=data,
            column_descriptions={
                "city": "City name.",
                "date": "Date of weather measurement.",
                "temperature_c": "Average daily temperature in Celsius.",
                "rainfall_mm": "Daily rainfall in millimeters.",
                "humidity_pct": "Average humidity percentage.",
                "weather_type": "Dominant weather category.",
            },
        )

    def _environment_pollution(self) -> TabularDataset:
        rows = 24
        rng = self._rng_for("environment_pollution")
        data = pd.DataFrame(
            {
                "station": np.array(
                    ["station_n", "station_s", "station_e", "station_w"]
                )[rng.integers(0, 4, rows)],
                "month": self._month_dates(rows, "2023-01-01"),
                "pm25": np.round(rng.normal(34, 12, rows).clip(min=8), 2),
                "no2": np.round(rng.normal(41, 9, rows).clip(min=12), 2),
                "ozone": np.round(rng.normal(28, 8, rows).clip(min=6), 2),
                "air_quality_band": np.array(["good", "moderate", "unhealthy"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="environment_pollution",
            domain="environment",
            description="Monthly air-quality pollutant readings by monitoring station.",
            dataframe=data,
            column_descriptions={
                "station": "Monitoring station identifier.",
                "month": "Reporting month.",
                "pm25": "Fine particulate concentration.",
                "no2": "Nitrogen dioxide concentration.",
                "ozone": "Ground-level ozone concentration.",
                "air_quality_band": "Categorized air quality status.",
            },
        )

    def _sports_player_stats(self) -> TabularDataset:
        rows = 25
        rng = self._rng_for("sports_player_stats")
        data = pd.DataFrame(
            {
                "player_role": np.array(["guard", "forward", "center"])[
                    rng.integers(0, 3, rows)
                ],
                "game_date": pd.date_range("2024-01-03", periods=rows, freq="3D"),
                "points": rng.integers(6, 42, rows),
                "assists": rng.integers(1, 14, rows),
                "rebounds": rng.integers(2, 18, rows),
                "home_away": np.array(["home", "away"])[rng.integers(0, 2, rows)],
            }
        )
        return self._make_dataset(
            name="sports_player_stats",
            domain="sports",
            description="Player-level game performance records.",
            dataframe=data,
            column_descriptions={
                "player_role": "Primary player role.",
                "game_date": "Date of game.",
                "points": "Points scored in game.",
                "assists": "Assists recorded in game.",
                "rebounds": "Rebounds recorded in game.",
                "home_away": "Whether game was home or away.",
            },
        )

    def _sports_team_results(self) -> TabularDataset:
        rows = 18
        rng = self._rng_for("sports_team_results")
        data = pd.DataFrame(
            {
                "team": np.array(["hawks", "lynx", "orcas", "rangers"])[
                    rng.integers(0, 4, rows)
                ],
                "match_week": self._week_dates(rows, "2024-02-04"),
                "goals_for": rng.integers(0, 6, rows),
                "goals_against": rng.integers(0, 5, rows),
                "possession_pct": np.round(
                    rng.normal(52, 7, rows).clip(min=35, max=70), 1
                ),
                "result": np.array(["win", "draw", "loss"])[rng.integers(0, 3, rows)],
            }
        )
        return self._make_dataset(
            name="sports_team_results",
            domain="sports",
            description="Weekly match outcomes and team performance stats.",
            dataframe=data,
            column_descriptions={
                "team": "Team name.",
                "match_week": "Week of match.",
                "goals_for": "Goals scored by team.",
                "goals_against": "Goals conceded by team.",
                "possession_pct": "Average possession percentage.",
                "result": "Match outcome category.",
            },
        )

    def _demographics_census(self) -> TabularDataset:
        rows = 21
        rng = self._rng_for("demographics_census")
        data = pd.DataFrame(
            {
                "region": np.array(["north", "south", "east", "west", "central"])[
                    rng.integers(0, 5, rows)
                ],
                "census_year": pd.date_range("2010-01-01", periods=rows, freq="YS"),
                "population_k": rng.integers(120, 3500, rows),
                "median_age": np.round(rng.normal(37, 5, rows).clip(min=24, max=56), 1),
                "urbanization_pct": np.round(
                    rng.normal(64, 11, rows).clip(min=30, max=92), 1
                ),
                "growth_tier": np.array(["declining", "stable", "growing"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="demographics_census",
            domain="demographics",
            description="Census-style demographic snapshots by region and year.",
            dataframe=data,
            column_descriptions={
                "region": "Geographic region.",
                "census_year": "Reference census year.",
                "population_k": "Population in thousands.",
                "median_age": "Median age of residents.",
                "urbanization_pct": "Population share in urban areas.",
                "growth_tier": "Population growth classification.",
            },
        )

    def _demographics_income(self) -> TabularDataset:
        rows = 20
        rng = self._rng_for("demographics_income")
        data = pd.DataFrame(
            {
                "district": np.array(["D1", "D2", "D3", "D4", "D5"])[
                    rng.integers(0, 5, rows)
                ],
                "survey_date": pd.date_range("2024-01-01", periods=rows, freq="2W"),
                "median_income_usd": np.round(
                    rng.normal(58200, 9200, rows).clip(min=32000), 2
                ),
                "unemployment_pct": np.round(
                    rng.normal(5.6, 1.8, rows).clip(min=1.8), 2
                ),
                "household_size": np.round(
                    rng.normal(2.8, 0.6, rows).clip(min=1.4, max=5.2), 2
                ),
                "housing_type": np.array(["owned", "rented", "mixed"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="demographics_income",
            domain="demographics",
            description="Socioeconomic district indicators from household surveys.",
            dataframe=data,
            column_descriptions={
                "district": "Surveyed district code.",
                "survey_date": "Date of survey period.",
                "median_income_usd": "Median household income in USD.",
                "unemployment_pct": "District unemployment percentage.",
                "household_size": "Average household size.",
                "housing_type": "Dominant housing tenure type.",
            },
        )

    def _transportation_traffic(self) -> TabularDataset:
        rows = 32
        rng = self._rng_for("transportation_traffic")
        data = pd.DataFrame(
            {
                "corridor": np.array(["A12", "B07", "C18", "D03"])[
                    rng.integers(0, 4, rows)
                ],
                "day": pd.date_range("2024-05-01", periods=rows, freq="D"),
                "vehicle_count": rng.integers(1800, 9900, rows),
                "avg_speed_kmh": np.round(
                    rng.normal(58, 9, rows).clip(min=28, max=82), 2
                ),
                "incident_count": rng.integers(0, 9, rows),
                "traffic_state": np.array(["free_flow", "moderate", "congested"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="transportation_traffic",
            domain="transportation",
            description="Daily roadway traffic and incident indicators.",
            dataframe=data,
            column_descriptions={
                "corridor": "Road corridor identifier.",
                "day": "Observation day.",
                "vehicle_count": "Total observed vehicles.",
                "avg_speed_kmh": "Average speed in km/h.",
                "incident_count": "Traffic incidents observed.",
                "traffic_state": "Categorized traffic condition.",
            },
        )

    def _energy_grid_load(self) -> TabularDataset:
        rows = 24
        rng = self._rng_for("energy_grid_load")
        data = pd.DataFrame(
            {
                "zone": np.array(["north_grid", "east_grid", "south_grid"])[
                    rng.integers(0, 3, rows)
                ],
                "hour_start": pd.date_range("2024-06-01", periods=rows, freq="h"),
                "load_mw": np.round(rng.normal(840, 150, rows).clip(min=420), 2),
                "renewable_share_pct": np.round(
                    rng.normal(38, 10, rows).clip(min=8, max=75), 2
                ),
                "spot_price_usd": np.round(rng.normal(72, 15, rows).clip(min=28), 2),
                "demand_band": np.array(["off_peak", "mid_peak", "peak"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="energy_grid_load",
            domain="energy",
            description="Hourly electricity load and market indicators by grid zone.",
            dataframe=data,
            column_descriptions={
                "zone": "Electric grid zone.",
                "hour_start": "Hour block start timestamp.",
                "load_mw": "Average load in megawatts.",
                "renewable_share_pct": "Share of renewable generation.",
                "spot_price_usd": "Spot market price in USD per MWh.",
                "demand_band": "Demand period classification.",
            },
        )

    def _agriculture_crop_yield(self) -> TabularDataset:
        rows = 17
        rng = self._rng_for("agriculture_crop_yield")
        data = pd.DataFrame(
            {
                "farm_zone": np.array(["zone_1", "zone_2", "zone_3", "zone_4"])[
                    rng.integers(0, 4, rows)
                ],
                "harvest_month": self._month_dates(rows, "2023-04-01"),
                "yield_ton": np.round(rng.normal(92, 24, rows).clip(min=30), 2),
                "rainfall_mm": np.round(rng.normal(78, 21, rows).clip(min=25), 2),
                "fertilizer_kg": np.round(rng.normal(180, 40, rows).clip(min=90), 2),
                "crop_type": np.array(["wheat", "corn", "soy", "rice"])[
                    rng.integers(0, 4, rows)
                ],
            }
        )
        return self._make_dataset(
            name="agriculture_crop_yield",
            domain="agriculture",
            description="Monthly crop yield and input factors by farm zone.",
            dataframe=data,
            column_descriptions={
                "farm_zone": "Agricultural zone identifier.",
                "harvest_month": "Month of harvest record.",
                "yield_ton": "Crop yield in tons.",
                "rainfall_mm": "Rainfall in mm over period.",
                "fertilizer_kg": "Fertilizer applied in kilograms.",
                "crop_type": "Dominant crop type.",
            },
        )

    def _manufacturing_quality(self) -> TabularDataset:
        rows = 27
        rng = self._rng_for("manufacturing_quality")
        data = pd.DataFrame(
            {
                "line": np.array(["line_a", "line_b", "line_c"])[
                    rng.integers(0, 3, rows)
                ],
                "inspection_date": pd.date_range("2024-01-01", periods=rows, freq="2D"),
                "units_produced": rng.integers(450, 2100, rows),
                "defect_rate_pct": np.round(
                    rng.normal(2.8, 0.9, rows).clip(min=0.8, max=6.0), 2
                ),
                "downtime_hours": np.round(rng.normal(3.5, 1.4, rows).clip(min=0.2), 2),
                "shift": np.array(["day", "swing", "night"])[rng.integers(0, 3, rows)],
            }
        )
        return self._make_dataset(
            name="manufacturing_quality",
            domain="manufacturing",
            description="Quality-control and production metrics by manufacturing line.",
            dataframe=data,
            column_descriptions={
                "line": "Production line identifier.",
                "inspection_date": "Date of quality inspection.",
                "units_produced": "Units produced during interval.",
                "defect_rate_pct": "Defective units percentage.",
                "downtime_hours": "Equipment downtime hours.",
                "shift": "Operational shift label.",
            },
        )

    def _tourism_hotel_occupancy(self) -> TabularDataset:
        rows = 16
        rng = self._rng_for("tourism_hotel_occupancy")
        data = pd.DataFrame(
            {
                "hotel_class": np.array(["economy", "midscale", "luxury"])[
                    rng.integers(0, 3, rows)
                ],
                "week_start": self._week_dates(rows, "2024-01-07"),
                "occupancy_pct": np.round(
                    rng.normal(71, 12, rows).clip(min=32, max=97), 2
                ),
                "average_daily_rate_usd": np.round(
                    rng.normal(148, 32, rows).clip(min=72), 2
                ),
                "booking_lead_days": rng.integers(2, 42, rows),
                "segment": np.array(["business", "leisure", "group"])[
                    rng.integers(0, 3, rows)
                ],
            }
        )
        return self._make_dataset(
            name="tourism_hotel_occupancy",
            domain="tourism",
            description="Weekly hotel occupancy and booking behavior indicators.",
            dataframe=data,
            column_descriptions={
                "hotel_class": "Hotel market segment.",
                "week_start": "Start date of reporting week.",
                "occupancy_pct": "Occupied room percentage.",
                "average_daily_rate_usd": "Average daily room rate in USD.",
                "booking_lead_days": "Average booking lead time in days.",
                "segment": "Guest segment.",
            },
        )

    def _ecommerce_campaign_performance(self) -> TabularDataset:
        rows = 23
        rng = self._rng_for("ecommerce_campaign_performance")
        data = pd.DataFrame(
            {
                "campaign": np.array(
                    ["spring_launch", "retargeting", "new_users", "vip_loyalty"]
                )[rng.integers(0, 4, rows)],
                "day": pd.date_range("2024-03-01", periods=rows, freq="D"),
                "impressions_k": rng.integers(40, 800, rows),
                "click_through_rate_pct": np.round(
                    rng.normal(3.6, 0.8, rows).clip(min=1.2), 2
                ),
                "conversion_rate_pct": np.round(
                    rng.normal(2.1, 0.55, rows).clip(min=0.7), 2
                ),
                "revenue_usd": np.round(rng.normal(9200, 2500, rows).clip(min=2800), 2),
                "channel": np.array(["search", "social", "email", "affiliate"])[
                    rng.integers(0, 4, rows)
                ],
            }
        )
        return self._make_dataset(
            name="ecommerce_campaign_performance",
            domain="retail",
            description="Daily digital campaign performance and monetization metrics.",
            dataframe=data,
            column_descriptions={
                "campaign": "Campaign identifier.",
                "day": "Reporting day.",
                "impressions_k": "Ad impressions in thousands.",
                "click_through_rate_pct": "CTR percentage.",
                "conversion_rate_pct": "Conversion percentage.",
                "revenue_usd": "Attributed revenue in USD.",
                "channel": "Acquisition channel.",
            },
        )
