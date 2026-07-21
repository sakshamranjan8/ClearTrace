from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIRECTORY = PROJECT_ROOT / "inference"

LOG_DIRECTORY = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIRECTORY / "hourly_update.log"


PIPELINE_STEPS = [
    (
        "Update OpenAQ observation cache",
        "update_observation_cache.py",
    ),
    (
        "Prepare station pollutant history",
        "prepare_pollutant_history.py",
    ),
    (
        "Update station weather cache",
        "update_weather_cache.py",
    ),
    (
        "Build live feature base",
        "build_live_features.py",
    ),
    (
        "Add AQI and AQI lag features",
        "add_aqi_features.py",
    ),
    (
        "Add temporal and spatial features",
        "add_temporal_spatial_features.py",
    ),
    (
        "Build latest model input",
        "build_model_input.py",
    ),
    (
        "Predict AQI for all stations",
        "predict_station_aqi.py",
    ),
]


def write_log(message):
    """
    Print a message and append it to the persistent log file.
    """

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    formatted_message = (
        f"[{timestamp}] {message}"
    )

    print(
        formatted_message,
        flush=True,
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            formatted_message + "\n"
        )


def run_pipeline_step(
    step_number,
    step_name,
    script_name,
):
    script_path = (
        INFERENCE_DIRECTORY / script_name
    )

    if not script_path.exists():
        raise FileNotFoundError(
            f"Pipeline script does not exist: "
            f"{script_path}"
        )

    command = [
        sys.executable,
        "-u",
        str(script_path),
    ]

    write_log("")
    write_log(
        f"STEP {step_number}/{len(PIPELINE_STEPS)} "
        f"STARTED: {step_name}"
    )

    write_log(
        "Command: "
        + " ".join(command)
    )

    step_start = time.perf_counter()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(
            f"Could not capture output for: "
            f"{step_name}"
        )

    for output_line in process.stdout:
        cleaned_line = output_line.rstrip()

        if cleaned_line:
            write_log(
                f"[{script_name}] {cleaned_line}"
            )

    return_code = process.wait()

    duration_seconds = (
        time.perf_counter() - step_start
    )

    if return_code != 0:
        raise RuntimeError(
            f"STEP {step_number} FAILED\n"
            f"Step name: {step_name}\n"
            f"Script: {script_path}\n"
            f"Exit code: {return_code}\n"
            f"Duration: {duration_seconds:.1f} seconds\n"
            f"Full output is available in: {LOG_FILE}"
        )

    write_log(
        f"STEP {step_number} COMPLETED: "
        f"{step_name} "
        f"({duration_seconds:.1f} seconds)"
    )


def run_hourly_update():
    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline_start = time.perf_counter()
    completed_steps = []

    write_log("=" * 70)
    write_log("CLEARTRACE HOURLY UPDATE STARTED")
    write_log(
        f"Python interpreter: {sys.executable}"
    )
    write_log(
        f"Project root: {PROJECT_ROOT}"
    )

    try:
        for step_number, (
            step_name,
            script_name,
        ) in enumerate(
            PIPELINE_STEPS,
            start=1,
        ):
            run_pipeline_step(
                step_number=step_number,
                step_name=step_name,
                script_name=script_name,
            )

            completed_steps.append(step_name)

    except Exception as error:
        total_duration = (
            time.perf_counter()
            - pipeline_start
        )

        write_log("")
        write_log("CLEARTRACE HOURLY UPDATE FAILED")
        write_log(
            f"Completed steps: "
            f"{len(completed_steps)}/"
            f"{len(PIPELINE_STEPS)}"
        )

        if completed_steps:
            write_log(
                "Successfully completed: "
                + ", ".join(completed_steps)
            )

        write_log(
            f"Error type: "
            f"{type(error).__name__}"
        )
        write_log(
            f"Error details: {error}"
        )
        write_log(
            f"Failed after "
            f"{total_duration:.1f} seconds"
        )
        write_log(
            f"Log file: {LOG_FILE}"
        )
        write_log("=" * 70)

        raise

    total_duration = (
        time.perf_counter()
        - pipeline_start
    )

    write_log("")
    write_log(
        "CLEARTRACE HOURLY UPDATE COMPLETED "
        "SUCCESSFULLY"
    )
    write_log(
        f"Completed steps: "
        f"{len(completed_steps)}/"
        f"{len(PIPELINE_STEPS)}"
    )
    write_log(
        f"Total duration: "
        f"{total_duration:.1f} seconds"
    )
    write_log(
        f"Station forecasts are ready."
    )
    write_log("=" * 70)


if __name__ == "__main__":
    try:
        run_hourly_update()

    except Exception:
        # A non-zero exit code lets n8n recognize
        # that the workflow failed.
        sys.exit(1)