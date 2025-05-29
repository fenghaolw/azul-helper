#!/usr/bin/env python3
"""
Unified Azul AI Startup Script

This script provides a complete startup solution for the Azul AI system,
combining smart server management with UI launching capabilities.
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from find_api_port import find_api_server_port


def check_python_requirements():
    """Check if required Python dependencies are installed."""
    try:
        import flask
        import flask_cors

        print("✅ Python dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing Python dependencies: {e}")
        print("💡 Install with: pip install flask flask-cors")
        return False


def check_node_dependencies():
    """Check if Node.js dependencies are installed."""
    webapp_dir = Path("webapp")
    node_modules = webapp_dir / "node_modules"

    if not webapp_dir.exists():
        print("❌ webapp directory not found")
        return False

    if not node_modules.exists():
        print("❌ Node.js dependencies not found")
        print("💡 Install with: cd webapp && npm install")
        return False

    print("✅ Node.js dependencies found")
    return True


def is_server_running():
    """Check if a server is already running."""
    return find_api_server_port()


def start_api_server(
    port=5000, kill_existing=False, agent_type="auto", background=False
):
    """Start the Python API server with smart port management."""
    print("🐍 Starting Python API server...")

    # Check if server is already running
    existing_server = is_server_running()
    if existing_server and not kill_existing:
        print(f"✅ API server already running on {existing_server['url']}")
        return existing_server, None

    # Prepare server command
    cmd = [
        sys.executable,
        "api_server.py",
        "--port",
        str(port),
        "--agent-type",
        agent_type,
    ]

    if kill_existing:
        cmd.append("--kill-existing")

    try:
        if background:
            # Start in background
            api_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            # Wait for server to start
            time.sleep(3)

            # Check if server is now running
            new_server = is_server_running()
            if new_server:
                print(f"✅ API server started on {new_server['url']}")
                print(f"   Agent: {new_server['agent_type']}")
                return new_server, api_process
            else:
                print("❌ API server failed to start")
                return None, None
        else:
            # Start in foreground (for non-UI mode)
            api_process = subprocess.Popen(cmd)
            time.sleep(2)
            new_server = is_server_running()
            if new_server:
                print(f"✅ API server started on {new_server['url']}")
                return new_server, api_process
            else:
                print("❌ API server failed to start")
                return None, None

    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None, None


def start_webapp():
    """Start the webapp development server."""
    print("🌐 Starting webapp development server...")
    try:
        webapp_dir = Path("webapp")
        webapp_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=webapp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give it time to start
        time.sleep(5)

        # Check if it's still running
        if webapp_process.poll() is None:
            print("✅ Webapp started on http://localhost:3000")
            return webapp_process
        else:
            stdout, stderr = webapp_process.communicate()
            print(f"❌ Webapp failed to start:")
            if stderr:
                print(f"Error: {stderr.decode()}")
            return None

    except Exception as e:
        print(f"❌ Failed to start webapp: {e}")
        return None


def open_browser(url="http://localhost:3000"):
    """Open the webapp in the default browser."""
    print(f"🌍 Opening {url} in browser...")
    try:
        webbrowser.open(url)
        print("✅ Browser opened successfully")
    except Exception as e:
        print(f"❌ Failed to open browser: {e}")
        print(f"💡 Manually navigate to: {url}")


def main():
    """Main function for the unified startup script."""
    parser = argparse.ArgumentParser(description="Azul AI Unified Startup Script")

    # Mode selection
    parser.add_argument(
        "--server-only", "-s", action="store_true", help="Start only the API server"
    )
    parser.add_argument(
        "--ui-only",
        "-u",
        action="store_true",
        help="Start only the webapp (assumes server is running)",
    )
    parser.add_argument(
        "--check-only",
        "-c",
        action="store_true",
        help="Only check if services are running",
    )

    # Server options
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Preferred API server port (default: 5000)",
    )
    parser.add_argument(
        "--kill-existing",
        "-k",
        action="store_true",
        help="Kill existing server process",
    )
    parser.add_argument(
        "--agent-type",
        "-a",
        choices=["auto", "mcts", "heuristic", "improved_heuristic"],
        default="auto",
        help="Agent type (default: auto)",
    )

    # UI options
    parser.add_argument(
        "--no-browser",
        "-n",
        action="store_true",
        help="Don't open browser automatically",
    )
    parser.add_argument(
        "--background",
        "-b",
        action="store_true",
        help="Run API server in background (UI mode only)",
    )

    args = parser.parse_args()

    print("🎮 Azul AI Unified Startup Script")
    print("=" * 50)

    # Check-only mode
    if args.check_only:
        print("\n🔍 Checking services...")

        # Check API server
        server_info = is_server_running()
        if server_info:
            print(f"✅ API Server: {server_info['url']} ({server_info['agent_type']})")
        else:
            print("❌ API Server: Not running")

        # Check webapp (simple check for port 3000)
        try:
            import urllib.request

            with urllib.request.urlopen("http://localhost:3000", timeout=2) as response:
                if response.status == 200:
                    print("✅ Webapp: http://localhost:3000")
                else:
                    print("❌ Webapp: Not responding")
        except Exception:
            print("❌ Webapp: Not running")

        sys.exit(0)

    # Server-only mode
    if args.server_only:
        print("\n🐍 Server-only mode")
        if not check_python_requirements():
            sys.exit(1)

        server_info, api_process = start_api_server(
            args.port, args.kill_existing, args.agent_type, background=False
        )

        if not server_info:
            sys.exit(1)

        print(f"\n🎯 API Server running on {server_info['url']}")
        print("⏹️  Press Ctrl+C to stop")

        try:
            if api_process:
                api_process.wait()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

        sys.exit(0)

    # UI-only mode
    if args.ui_only:
        print("\n🌐 UI-only mode")
        if not check_node_dependencies():
            sys.exit(1)

        # Check if API server is already running
        server_info = is_server_running()
        if server_info:
            print(f"✅ Using existing API server on {server_info['url']}")
        else:
            print("⚠️  No API server found - webapp will use TypeScript AI only")

        webapp_process = start_webapp()
        if not webapp_process:
            sys.exit(1)

        if not args.no_browser:
            time.sleep(2)
            open_browser()

        print(f"\n🎯 Webapp running on http://localhost:3000")
        print("⏹️  Press Ctrl+C to stop")

        try:
            webapp_process.wait()
        except KeyboardInterrupt:
            print("\n👋 Webapp stopped")
            webapp_process.terminate()

        sys.exit(0)

    # Full UI mode (default)
    print("\n🚀 Full UI mode (API server + webapp)")

    # Check dependencies
    print("\n📋 Checking dependencies...")
    if not check_python_requirements():
        sys.exit(1)

    if not check_node_dependencies():
        sys.exit(1)

    print("\n🐍 Starting API server...")
    server_info, api_process = start_api_server(
        args.port, args.kill_existing, args.agent_type, background=True
    )

    if not server_info:
        print("⚠️  API server failed - continuing with TypeScript AI only")
        api_process = None

    print("\n🌐 Starting webapp...")
    webapp_process = start_webapp()
    if not webapp_process:
        print("❌ Webapp failed to start")
        if api_process:
            api_process.terminate()
        sys.exit(1)

    # Open browser
    if not args.no_browser:
        time.sleep(2)
        open_browser()

    # Status summary
    print("\n" + "=" * 50)
    print("🎯 Services are running!")
    if server_info:
        print(f"🐍 Python API: {server_info['url']} ({server_info['agent_type']})")
    else:
        print("🐍 Python API: Not available (using TypeScript AI)")
    print("🌐 Webapp: http://localhost:3000")

    print("\n💡 Tips:")
    print("• The webapp will auto-discover the API server port")
    print("• Use the 🐍 Python AI button to switch between AI types")
    print("• Check AI Statistics for connection status")
    print("• Use F12 for browser dev tools")

    print("\n⏹️  Press Ctrl+C to stop all services")
    print("=" * 50)

    try:
        # Monitor processes
        while True:
            time.sleep(1)

            # Check webapp
            if webapp_process.poll() is not None:
                print("\n❌ Webapp terminated unexpectedly")
                break

            # Check API server
            if api_process and api_process.poll() is not None:
                print("\n⚠️  API server terminated")
                api_process = None

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")

        if webapp_process:
            webapp_process.terminate()
            print("✅ Webapp stopped")

        if api_process:
            api_process.terminate()
            print("✅ API server stopped")

        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
