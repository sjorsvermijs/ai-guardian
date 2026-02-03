"""Test configuration and fixtures"""

import pytest


@pytest.fixture
def sample_audio_path():
    """Path to sample audio file"""
    return "data/sample_audio.wav"


@pytest.fixture
def sample_video_path():
    """Path to sample video file"""
    return "data/sample_video.mp4"


@pytest.fixture
def sample_image_path():
    """Path to sample image file"""
    return "data/sample_image.jpg"
