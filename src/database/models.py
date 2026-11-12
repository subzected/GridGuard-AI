from datetime import datetime, timezone
from sqlalchemy import String, Float, DateTime, Integer, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class WeatherRecord(Base):
    __tablename__ = "weather_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    location_id: Mapped[str] = mapped_column(String(80), index=True)
    location_name: Mapped[str] = mapped_column(String(120))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    relative_humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    rain_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_gusts_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_direction_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_hpa: Mapped[float | None] = mapped_column(Float, nullable=True)
    precipitation_probability_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    thunderstorm_probability_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    weather_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="unknown")
    temporal_resolution: Mapped[str] = mapped_column(String(30), default="hourly")

Index(
    "ix_weather_location_valid",
    WeatherRecord.location_id,
    WeatherRecord.valid_time
)

# Approximate city/area weather points for prototyping only; not feeder locations.
MONITORING_LOCATIONS = [
    {"location_id":"kolkata","name":"Kolkata","latitude":22.5726,"longitude":88.3639},
    {"location_id":"howrah","name":"Howrah","latitude":22.5958,"longitude":88.2636},
    {"location_id":"siliguri","name":"Siliguri","latitude":26.7271,"longitude":88.3953},
    {"location_id":"darjeeling","name":"Darjeeling","latitude":27.0410,"longitude":88.2663},
    {"location_id":"asansol","name":"Asansol","latitude":23.6889,"longitude":86.9661},
    {"location_id":"durgapur","name":"Durgapur","latitude":23.5204,"longitude":87.3119},
    {"location_id":"bardhaman","name":"Bardhaman","latitude":23.2324,"longitude":87.8615},
    {"location_id":"kharagpur","name":"Kharagpur","latitude":22.3460,"longitude":87.2320},
    {"location_id":"haldia","name":"Haldia","latitude":22.0667,"longitude":88.0698},
    {"location_id":"berhampore","name":"Berhampore","latitude":24.0988,"longitude":88.2679},
    {"location_id":"malda","name":"Malda","latitude":25.0108,"longitude":88.1411},
    {"location_id":"krishnanagar","name":"Krishnanagar","latitude":23.4058,"longitude":88.4907},
    {"location_id":"cooch_behar","name":"Cooch Behar","latitude":26.3242,"longitude":89.4510},
    {"location_id":"bankura","name":"Bankura","latitude":23.2324,"longitude":87.0784},
    {"location_id":"purulia","name":"Purulia","latitude":23.3322,"longitude":86.3652},
]
