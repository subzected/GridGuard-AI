import os
import logging
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from src.ingestion.open_meteo import collect_all_locations

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)


def run_weather_collection():

    logger.info("Starting scheduled weather collection")

    try:
        collect_all_locations()

        logger.info("Weather collection job completed")

    except Exception:
        logger.exception("Weather collection job failed")


def main():

    minutes = int(
        os.getenv("POLL_INTERVAL_MINUTES", "30")
    )

    if minutes < 1:
        raise ValueError(
            "POLL_INTERVAL_MINUTES must be at least 1"
        )

    scheduler = BlockingScheduler(
        timezone="Asia/Kolkata"
    )

    scheduler.add_job(
        run_weather_collection,
        trigger="interval",
        minutes=minutes,
        id="open_meteo_collection",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300
    )

    logger.info("Performing initial weather collection")

    # Initial collection
    run_weather_collection()

    logger.info(
        "Weather scheduler started. Interval: %s minutes",
        minutes
    )

    try:
        scheduler.start()

    except (KeyboardInterrupt, SystemExit):
        logger.info("Weather scheduler stopped")

    finally:
        if scheduler.running:
            scheduler.shutdown(wait=True)


if __name__ == "__main__":
    main()




# import logging
# import os
# from dotenv import load_dotenv
# from apscheduler.schedulers.blocking import BlockingScheduler
# from src.ingestion.open_meteo import collect_all_locations

# load_dotenv()
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# def main():
#     minutes = int(os.getenv("POLL_INTERVAL_MINUTES", "30"))
#     scheduler = BlockingScheduler(timezone="UTC")
#     scheduler.add_job(collect_all_locations, "interval", minutes=minutes,
#                       id="open_meteo_collection", max_instances=1,
#                       coalesce=True, misfire_grace_time=300)
#     collect_all_locations()  # initial collection
#     logging.info("Weather collection scheduled every %s minutes", minutes)
#     scheduler.start()

# if __name__ == "__main__":
#     main()
