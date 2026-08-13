import subprocess
import sys
import time
import socket
import http.client


def test_no_subcommand_starts_server(tmp_path):
    """`python main.py` with no subcommand should start the API server."""
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--storage-dir", str(tmp_path / "sessions")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Wait for server to be ready (up to 10s)
        for _ in range(100):
            try:
                conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=0.5)
                conn.request("GET", "/api/health")
                r = conn.getresponse()
                if r.status == 200:
                    break
            except (ConnectionRefusedError, socket.timeout):
                pass
            time.sleep(0.1)
        else:
            stderr = proc.stderr.read().decode()
            assert False, f"server did not start within 10s. stderr: {stderr}"

        # Verify health endpoint responds
        conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=2)
        conn.request("GET", "/api/health")
        r = conn.getresponse()
        assert r.status == 200
        body = r.read().decode()
        assert "ok" in body
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_server_subcommand_starts_server(tmp_path):
    """`python main.py server` should also start the API server."""
    proc = subprocess.Popen(
        [sys.executable, "main.py", "server",
         "--storage-dir", str(tmp_path / "sessions"),
         "--port", "8765"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        for _ in range(100):
            try:
                conn = http.client.HTTPConnection("127.0.0.1", 8765, timeout=0.5)
                conn.request("GET", "/api/health")
                r = conn.getresponse()
                if r.status == 200:
                    break
            except (ConnectionRefusedError, socket.timeout):
                pass
            time.sleep(0.1)
        else:
            stderr = proc.stderr.read().decode()
            assert False, f"server did not start within 10s. stderr: {stderr}"

        conn = http.client.HTTPConnection("127.0.0.1", 8765, timeout=2)
        conn.request("GET", "/api/health")
        r = conn.getresponse()
        assert r.status == 200
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_server_help_in_main_help():
    """`python main.py --help` should list server as a subcommand."""
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert "server" in result.stdout
