"""run_all_tests.py — run every test file of SFF v3 in order and stop at the first failure.

    python run_all_tests.py
"""
import subprocess
import sys

TESTS = ["test_config.py", "test_raw_data_validation.py", "test_phase1.py", "test_phase2_3.py",
         "test_phase4_5.py", "test_pipeline.py", "test_engineering.py", "test_statistics.py", "test_ideas.py", "test_nucleo.py", "test_topologia.py"]

if __name__ == "__main__":
    for test_file in TESTS:
        print(f"\n▶ {test_file}")
        completed = subprocess.run([sys.executable, test_file])
        if completed.returncode != 0:
            print(f"\nSTOPPED at {test_file}")
            sys.exit(1)
    print("\nevery test file passed")
