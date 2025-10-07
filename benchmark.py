# benchmark.py
import subprocess
import time
from pathlib import Path
import argparse

def benchmark(video_file: Path):
    """Runs the summarizer and profiles its time and memory usage."""
    if not video_file.exists():
        print(f"Error: Video file '{video_file}' not found.")
        return

    output_file = video_file.with_name(f"{video_file.stem}_summary.mp4")
    command = [
        "mprof", "run", "python", "summarize.py", str(video_file),
        "--output", str(output_file),
        "--ratio", "0.1",
        "--scale", "360",
        "--fps", "2"
    ]

    print(f"--- Starting Benchmark for {video_file.name} ---")
    start_time = time.time()
    
    # Run the process
    result = subprocess.run(command, capture_output=True, text=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n--- Benchmark Results ---")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if result.returncode == 0:
        print("Summarization completed successfully.")
        # mprof will save a .dat file. We can print the last line for peak memory.
        try:
            with open(list(Path(".").glob("mprofile_*.dat"))[-1]) as f:
                lines = f.readlines()
                peak_mem_line = [line for line in lines if line.startswith("MEM")]
                if peak_mem_line:
                    peak_mem = max(float(parts[1]) for parts in (line.split() for line in peak_mem_line))
                    print(f"Peak memory usage: {peak_mem:.2f} MiB")
        except (IOError, IndexError, ValueError):
            print("Could not parse memory profile data.")
    else:
        print("Summarization failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
    # Clean up profiling data
    for f in Path(".").glob("mprofile_*.dat"):
        f.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the video summarizer.")
    parser.add_argument("video", type=Path, help="Path to the sample video file.")
    args = parser.parse_args()
    benchmark(args.video)