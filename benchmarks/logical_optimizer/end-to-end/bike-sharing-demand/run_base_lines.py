import time
import subprocess
import sys
import os


def run_pipeline(pipeline_name, input_filename):
    """Run a pipeline script and measure its execution time."""
    print(f"\n{'=' * 60}")
    print(f"Running {pipeline_name}...")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Run the pipeline script
        # Prepare environment with selected input filename
        env = dict(os.environ)
        env["BIKE_INPUT_FILE"] = input_filename

        result = subprocess.run(
            [sys.executable, pipeline_name],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )

        elapsed_time = time.time() - start_time

        # Print the output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        print(f"\n✓ {pipeline_name} completed in {elapsed_time:.2f} seconds")

        return elapsed_time, True

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {pipeline_name} failed after {elapsed_time:.2f} seconds")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return elapsed_time, False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {pipeline_name} error after {elapsed_time:.2f} seconds: {e}")
        return elapsed_time, False


def main():
    """Run all pipelines sequentially and report timing results."""
    # Allow selecting input file variant via CLI arg; fallback to 'train.csv'
    input_filename = sys.argv[1] if len(sys.argv) > 1 else "train_augmented_3x.csv"
    pipelines = [f"pipeline{i}.py" for i in range(5)]

    results = {}
    total_start = time.time()

    print("Starting pipeline execution...")
    print(f"Total pipelines to run: {len(pipelines)}")

    for pipeline in pipelines:
        elapsed, success = run_pipeline(pipeline, input_filename)
        results[pipeline] = {
            'time': elapsed,
            'success': success
        }

    total_time = time.time() - total_start

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")

    for pipeline, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{pipeline:20s} - {result['time']:8.2f}s - {status}")

    print(f"{'-' * 60}")
    print(f"{'Total time:':20s}   {total_time:8.2f}s")
    print(f"{'=' * 60}")

    # Count successes
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nCompleted: {successful}/{len(pipelines)} pipelines successful")


if __name__ == "__main__":
    main()