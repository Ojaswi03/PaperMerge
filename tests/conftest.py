"""Tests never create real cloud experiment runs from machine-local settings."""
import pytest


@pytest.fixture(autouse=True)
def disable_cloud_tracking(monkeypatch):
    monkeypatch.setenv('PAPERMERGE_LIVE_TRACKING','disabled')
