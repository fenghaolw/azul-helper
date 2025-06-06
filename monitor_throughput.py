import argparse
import os
import re
from datetime import datetime
from glob import glob


def parse_actor_logs(log_directory):
    """
    Parses all 'log-actor-*.txt' files in a directory to find game completion times.

    Args:
        log_directory (str): The path to the directory containing the log files.

    Returns:
        list: A sorted list of datetime objects, one for each completed game.
    """
    # Find all actor log files in the specified directory
    log_pattern = os.path.join(log_directory, "log-actor-*.txt")

    log_files = glob(log_pattern)

    if not log_files:
        print(f"Error: No actor logs found in directory '{log_directory}'")
        return []

    print(f"Found {len(log_files)} actor log file(s) to analyze...")

    game_completion_times = []

    # Regex to find lines indicating a completed game and capture the timestamp.
    # Format: [YYYY-MM-DD HH:MM:SS.mmm] Game X: Returns: ...
    game_line_regex = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] Game \d+: Returns:"
    )

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                for line in f:
                    match = game_line_regex.search(line)
                    if match:
                        timestamp_str = match.group(1)
                        # Convert timestamp string to a datetime object
                        dt_object = datetime.strptime(
                            timestamp_str, "%Y-%m-%d %H:%M:%S.%f"
                        )
                        game_completion_times.append(dt_object)
        except Exception as e:
            print(f"Warning: Could not read or parse file {log_file}. Error: {e}")

    # Sort the timestamps chronologically
    game_completion_times.sort()

    return game_completion_times


def main():
    """Main function to parse arguments and calculate throughput."""
    parser = argparse.ArgumentParser(
        description="Monitor AlphaZero training throughput by parsing actor logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_dir",
        help="The directory containing the log files (e.g., 'models/libtorch_alphazero_azul/').",
    )
    args = parser.parse_args()

    # Convert the input path to an absolute path to avoid ambiguity
    log_directory_path = os.path.abspath(args.log_dir)

    print(f"Attempting to find logs in absolute path: {log_directory_path}")

    # Check if the directory actually exists
    if not os.path.isdir(log_directory_path):
        print(f"\nError: The specified directory does not exist or is not a directory.")
        print(f"Please check the path: {log_directory_path}")
        return

    completion_times = parse_actor_logs(log_directory_path)

    total_games = len(completion_times)

    if total_games < 2:
        print("\nNot enough data to calculate throughput.")
        print(f"Total games completed so far: {total_games}")
        return

    # Calculate duration from the first completed game to the last
    start_time = completion_times[0]
    end_time = completion_times[-1]
    duration = end_time - start_time

    total_seconds = duration.total_seconds()

    # Avoid division by zero if all games finished in the same second
    if total_seconds == 0:
        print(
            "\nAll recorded games finished in the same second. Cannot calculate rate."
        )
        return

    total_hours = total_seconds / 3600.0
    games_per_hour = total_games / total_hours

    print("\n--- Training Throughput Report ---")
    print(f"  Total Games Completed: {total_games}")
    print(f"  Time of First Game Finish: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Time of Last Game Finish:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"  Total Duration Analyzed: {str(duration).split('.')[0]}"
    )  # Show readable duration
    print("------------------------------------")
    print(f"  Current Throughput: {games_per_hour:.2f} games per hour")
    print("------------------------------------")


if __name__ == "__main__":
    main()
