#!/usr/bin/env python3
"""
Utility script to find the Azul AI API server port.
This helps clients automatically discover which port the server is running on.
"""

import json
import sys
import urllib.error
import urllib.request
from typing import Optional


def find_api_server_port(
    start_port: int = 5000, max_attempts: int = 10
) -> Optional[dict]:
    """
    Find the Azul AI API server by checking ports starting from start_port.

    Returns:
        Dict with server info if found, None otherwise
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Azul-API-Finder/1.0")

            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    if data.get("status") == "healthy":
                        return {
                            "port": port,
                            "url": f"http://localhost:{port}",
                            "agent_type": data.get("active_agent_type", "unknown"),
                            "server_info": data.get("server", {}),
                            "health_data": data,
                        }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
        ):
            # Port not responding or server not running
            continue
        except json.JSONDecodeError:
            # Invalid JSON response
            continue

    return None


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Find Azul AI API Server")
    parser.add_argument(
        "--start-port",
        type=int,
        default=5000,
        help="Starting port to check (default: 5000)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Maximum ports to check (default: 10)",
    )
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    print(
        f"🔍 Searching for Azul AI API server starting from port {args.start_port}..."
    )

    server_info = find_api_server_port(args.start_port, args.max_attempts)

    if server_info:
        if args.json:
            print(json.dumps(server_info, indent=2))
        else:
            print(f"✅ Found Azul AI API server!")
            print(f"   🌐 URL: {server_info['url']}")
            print(f"   🤖 Agent: {server_info['agent_type']}")
            print(f"   📡 Port: {server_info['port']}")

            # Additional server details
            health = server_info["health_data"]
            if health.get("agent_initialized"):
                print(
                    f"   ✅ Agent initialized: {health.get('current_agent_type', 'unknown')}"
                )
            if health.get("neural_network_available"):
                print(f"   🧠 Neural network available")
            if health.get("mcts_available"):
                print(f"   🎯 MCTS available")
            if health.get("heuristic_available"):
                print(f"   🔧 Heuristic available")

        sys.exit(0)
    else:
        if args.json:
            print('{"error": "Server not found"}')
        else:
            print(
                f"❌ No Azul AI API server found on ports {args.start_port}-{args.start_port + args.max_attempts - 1}"
            )
            print("💡 Try starting the server with: python3 api_server.py")

        sys.exit(1)


if __name__ == "__main__":
    main()
