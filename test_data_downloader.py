#!/usr/bin/env python3
"""
test_data_downloader.py
=======================
Test script to verify data_downloader.py functionality.

Usage:
    python test_data_downloader.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and check if it succeeds."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✓ {description} - PASSED")
            return True
        else:
            print(f"✗ {description} - FAILED (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False


def check_directories():
    """Check if data directories are created."""
    print(f"\n{'='*80}")
    print("TEST: Check Data Directories")
    print(f"{'='*80}")

    required_dirs = [
        "data/bhavcopy",
        "data/vix",
        "data/fii_dii",
        "data/extended",
        "logs",
        "models",
        "output",
    ]

    all_exist = True
    for d in required_dirs:
        path = Path(d)
        if path.exists():
            print(f"✓ {d} exists")
        else:
            print(f"✗ {d} does not exist")
            all_exist = False

    if all_exist:
        print("✓ All directories exist - PASSED")
    else:
        print("✗ Some directories missing - FAILED")

    return all_exist


def main():
    """Run all tests."""
    print("="*80)
    print("DATA DOWNLOADER TEST SUITE")
    print("="*80)

    tests = []

    # Test 1: Help command
    tests.append(run_command(
        ["python", "data_downloader.py", "--help"],
        "Display help"
    ))

    # Test 2: Global info
    tests.append(run_command(
        ["python", "data_downloader.py", "--global-info"],
        "Display global signals info"
    ))

    # Test 3: Check directories are created
    tests.append(check_directories())

    # Test 4: Test with no arguments (should show help)
    tests.append(run_command(
        ["python", "data_downloader.py"],
        "No arguments (should show help)"
    ))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(tests)
    total = len(tests)

    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
