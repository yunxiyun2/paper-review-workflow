import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from paper_review_workflow.cli import main
from paper_review_workflow.engine import ReviewEngine
from paper_review_workflow.storage.json_file import JsonFileStorage
from paper_review_workflow.core.models import WorkflowRun, WorkflowStatus


def test_list_runs_empty(capsys, tmp_path):
    """list-runs with no runs should print header only"""
    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "list-runs",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "RUN_ID" in out


def test_show_run_nonexistent(capsys, tmp_path):
    """show-run with nonexistent ID should exit 1 and print to stderr"""
    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "show-run", "nonexistent",
    ])
    assert exit_code == 1


def test_show_run_existing(capsys, tmp_path):
    """show-run with existing run should print the run JSON"""
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    run = WorkflowRun(status=WorkflowStatus.SUCCESS)
    storage.save_run(run)
    storage.close()

    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "show-run", run.id,
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert run.id in out
    assert "success" in out


def test_list_runs_with_data(capsys, tmp_path):
    """list-runs with saved runs should list them"""
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    run1 = WorkflowRun(status=WorkflowStatus.SUCCESS)
    run2 = WorkflowRun(status=WorkflowStatus.FAILURE)
    storage.save_run(run1)
    storage.save_run(run2)
    storage.close()

    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "list-runs",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert run1.id in out
    assert run2.id in out


def test_list_runs_filter_by_status(capsys, tmp_path):
    """list-runs --status success should only show successful runs"""
    storage = JsonFileStorage(data_dir=str(tmp_path))
    storage.open()
    run1 = WorkflowRun(status=WorkflowStatus.SUCCESS)
    run2 = WorkflowRun(status=WorkflowStatus.FAILURE)
    storage.save_run(run1)
    storage.save_run(run2)
    storage.close()

    exit_code = main([
        "--storage", "json", "--storage-dir", str(tmp_path),
        "list-runs", "--status", "success",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert run1.id in out  # success run shown
    assert run2.id not in out  # failure run filtered out
