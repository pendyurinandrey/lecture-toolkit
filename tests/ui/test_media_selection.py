"""Правила выбора файлов в интерфейсе (без Qt)."""

import pytest

from common.media_info import MediaInfo
from ui.media_selection import (
    audio_output_extension,
    audio_output_filter,
    audio_output_label,
    check_new_file,
    is_audio,
    open_file_filter,
    open_file_title,
    source_caption,
    video_output_visible,
)

M4A = MediaInfo(kind="audio", audio_codec="aac", sample_rate=48000, channels=1)
MP3 = MediaInfo(kind="audio", audio_codec="mp3", sample_rate=44100, channels=2)
VIDEO = MediaInfo(kind="video", audio_codec="aac", sample_rate=48000, channels=2,
                  video_codec="h264", width=1920, height=1080, pix_fmt="yuv420p", frame_rate="30/1")


class TestOpenFileFilter:
    def test_first_file_may_be_video_or_audio(self):
        f = open_file_filter(None)
        assert f.startswith("Видео и аудио (*.mp4 *.m4a *.mp3)")
        assert "Видео MP4 (*.mp4)" in f and "*.m4a *.mp3" in f and f.endswith("Все файлы (*)")

    def test_after_a_video_only_video(self):
        assert open_file_filter(VIDEO) == "Видео MP4 (*.mp4);;Все файлы (*)"

    def test_after_an_m4a_only_m4a(self):
        assert open_file_filter(M4A) == "Аудио M4A (*.m4a);;Все файлы (*)"

    def test_after_an_mp3_only_mp3(self):
        assert open_file_filter(MP3) == "Аудио MP3 (*.mp3);;Все файлы (*)"

    @pytest.mark.parametrize("reference, expected", [
        (None, "Выберите видео или аудио файл"), (VIDEO, "Выберите видеофайл"), (M4A, "Выберите аудиофайл")])
    def test_dialog_title(self, reference, expected):
        assert open_file_title(reference) == expected


class TestSourceCaption:
    def test_not_chosen_yet(self):
        assert "не выбран" in source_caption(None) and "первый добавленный файл" in source_caption(None)

    def test_video(self):
        assert source_caption(VIDEO).startswith("Источник: видео")

    def test_audio_shows_format_and_parameters(self):
        assert source_caption(M4A) == ("Источник: аудио (M4A/AAC, 48 кГц, моно). "
                                       "Добавлять можно только аудио в таком же формате")
        assert source_caption(MP3).startswith("Источник: аудио (MP3, 44.1 кГц, стерео).")

    def test_unsupported_audio_codec_does_not_crash(self):
        assert "OPUS" in source_caption(MediaInfo(kind="audio", audio_codec="opus", sample_rate=48000, channels=6))


class TestCheckNewFile:
    def test_first_file_of_any_supported_kind_is_accepted(self):
        for info in (VIDEO, M4A, MP3):
            assert check_new_file(None, info, "x") is None

    def test_first_file_with_unsupported_audio_format_is_refused(self):
        wav = MediaInfo(kind="audio", audio_codec="pcm_s16le", sample_rate=44100, channels=2)
        message = check_new_file(None, wav, "talk.wav")
        assert message.startswith("talk.wav:") and "PCM_S16LE" in message and "AAC (.m4a) или MP3 (.mp3)" in message

    def test_compatible_file_is_accepted(self):
        assert check_new_file(M4A, M4A, "part2.m4a") is None
        assert check_new_file(VIDEO, VIDEO, "b.mp4") is None

    def test_video_after_audio_and_audio_after_video_are_refused(self):
        assert "смешивать видео и аудио нельзя" in check_new_file(M4A, VIDEO, "v.mp4")
        assert "смешивать видео и аудио нельзя" in check_new_file(VIDEO, M4A, "a.m4a")

    def test_mp3_after_m4a_is_refused_with_the_difference(self):
        assert check_new_file(M4A, MP3, "b.mp3") == \
            "b.mp3: кодек звука MP3 вместо AAC; частота звука 44100 Гц вместо 48000 Гц; каналов звука: 2 вместо 1"

    def test_different_sample_rate_is_refused(self):
        other = MediaInfo(kind="audio", audio_codec="aac", sample_rate=44100, channels=1)
        assert check_new_file(M4A, other, "b.m4a") == "b.m4a: частота звука 44100 Гц вместо 48000 Гц"


class TestAudioOutput:
    def test_default_and_video_mode_extract_m4a(self):
        for reference in (None, VIDEO):
            assert audio_output_extension(reference) == ".m4a"
            assert audio_output_filter(reference) == "M4A аудио (*.m4a)"
            assert audio_output_label(reference) == "Аудио (M4A):"

    def test_audio_mode_follows_the_source_format(self):
        assert audio_output_extension(M4A) == ".m4a" and audio_output_label(M4A) == "Аудио (M4A):"
        assert audio_output_extension(MP3) == ".mp3"
        assert audio_output_filter(MP3) == "MP3 аудио (*.mp3)"
        assert audio_output_label(MP3) == "Аудио (MP3):"

    def test_unsupported_audio_falls_back_to_m4a(self):
        assert audio_output_extension(MediaInfo(kind="audio", audio_codec="opus")) == ".m4a"


class TestModeFlags:
    def test_is_audio(self):
        assert is_audio(M4A) and is_audio(MP3) and not is_audio(VIDEO) and not is_audio(None)

    def test_video_row_is_hidden_only_in_audio_mode(self):
        assert video_output_visible(None) and video_output_visible(VIDEO)
        assert not video_output_visible(M4A) and not video_output_visible(MP3)
