"""Тип файла по содержимому и совместимость для склейки без перекодирования."""

import pytest

from common.media_info import MediaInfo, audio_extension, incompatibilities, parse_probe


def stream(codec_type, codec_name, **fields):
    disposition = {"attached_pic": fields.pop("attached_pic", 0)}
    return {"codec_type": codec_type, "codec_name": codec_name, "disposition": disposition, **fields}


AAC_MONO = stream("audio", "aac", sample_rate="48000", channels=1)
H264 = stream("video", "h264", width=1920, height=1080, pix_fmt="yuv420p", r_frame_rate="30/1")
STEREO = stream("audio", "aac", sample_rate="48000", channels=2)


class TestParseProbe:
    def test_m4a_is_audio(self):
        info = parse_probe({"streams": [AAC_MONO]})
        assert info == MediaInfo(kind="audio", audio_codec="aac", sample_rate=48000, channels=1)

    def test_mp3_is_audio(self):
        info = parse_probe({"streams": [stream("audio", "mp3", sample_rate="44100", channels=2)]})
        assert (info.kind, info.audio_codec, info.sample_rate, info.channels) == ("audio", "mp3", 44100, 2)

    def test_cover_art_is_not_video(self):
        cover = stream("video", "mjpeg", width=600, height=600, r_frame_rate="0/0", attached_pic=1)
        assert parse_probe({"streams": [AAC_MONO, cover]}).kind == "audio"

    def test_mp4_with_video_and_sound_is_video(self):
        info = parse_probe({"streams": [H264, STEREO]})
        assert (info.kind, info.video_codec, info.width, info.height) == ("video", "h264", 1920, 1080)
        assert (info.pix_fmt, info.frame_rate, info.audio_codec, info.channels) == ("yuv420p", "30/1", "aac", 2)

    def test_mp4_without_a_video_stream_is_audio(self):
        assert parse_probe({"streams": [AAC_MONO]}).kind == "audio"

    def test_video_without_sound(self):
        info = parse_probe({"streams": [H264]})
        assert info.kind == "video" and info.audio_codec is None and info.sample_rate is None

    @pytest.mark.parametrize("data", [{}, {"streams": []}, {"streams": None},
                                      {"streams": [stream("subtitle", "mov_text")]}])
    def test_nothing_playable_is_an_error(self, data):
        with pytest.raises(ValueError, match="нет ни видео-, ни аудиопотока"):
            parse_probe(data)

    def test_missing_or_broken_numbers_become_none(self):
        info = parse_probe({"streams": [stream("audio", "aac", sample_rate="N/A")]})
        assert info.sample_rate is None and info.channels is None

    def test_only_the_first_audio_stream_counts(self):
        second = stream("audio", "mp3", sample_rate="8000", channels=1)
        assert parse_probe({"streams": [AAC_MONO, second]}).audio_codec == "aac"


class TestAudioExtension:
    @pytest.mark.parametrize("codec, expected", [("aac", ".m4a"), ("mp3", ".mp3"), ("opus", None), ("pcm_s16le", None), (None, None)])
    def test_extension_follows_the_codec(self, codec, expected):
        assert audio_extension(MediaInfo(kind="audio", audio_codec=codec)) == expected


def audio(**fields):
    return MediaInfo(**{"kind": "audio", "audio_codec": "aac", "sample_rate": 48000, "channels": 1, **fields})


def video(**fields):
    return MediaInfo(**{"kind": "video", "audio_codec": "aac", "sample_rate": 48000, "channels": 2,
                        "video_codec": "h264", "width": 1920, "height": 1080, "pix_fmt": "yuv420p",
                        "frame_rate": "30/1", **fields})


class TestIncompatibilities:
    def test_identical_files_are_compatible(self):
        assert incompatibilities(audio(), audio()) == []
        assert incompatibilities(video(), video()) == []

    def test_mixing_video_and_audio_is_refused(self):
        assert incompatibilities(video(), audio()) == ["это аудио, а первый файл — видео; смешивать видео и аудио нельзя"]
        assert incompatibilities(audio(), video()) == ["это видео, а первый файл — аудио; смешивать видео и аудио нельзя"]

    def test_the_silent_corruption_case_is_caught(self):
        # 48 кГц стерео + 44.1 кГц моно ffmpeg склеивает молча и теряет секунды звука
        problems = incompatibilities(audio(channels=2), audio(sample_rate=44100, channels=1))
        assert problems == ["частота звука 44100 Гц вместо 48000 Гц", "каналов звука: 1 вместо 2"]

    def test_m4a_and_mp3_differ_by_codec(self):
        assert incompatibilities(audio(), audio(audio_codec="mp3")) == ["кодек звука MP3 вместо AAC"]

    def test_missing_sound_on_either_side(self):
        assert incompatibilities(audio(), audio(audio_codec=None, sample_rate=None, channels=None)) == \
            ["в файле нет звука, а в первом есть"]
        assert incompatibilities(audio(audio_codec=None, sample_rate=None, channels=None), audio()) == \
            ["в файле есть звук, а в первом нет"]

    def test_video_parameters(self):
        assert incompatibilities(video(), video(width=1280, height=720)) == ["разрешение 1280×720 вместо 1920×1080"]
        assert incompatibilities(video(), video(video_codec="hevc")) == ["кодек видео HEVC вместо H264"]
        assert incompatibilities(video(), video(pix_fmt="yuv444p")) == ["формат пикселей yuv444p вместо yuv420p"]
        assert incompatibilities(video(), video(frame_rate="60/1")) == ["частота кадров 60 вместо 30"]

    def test_fractional_frame_rate_is_readable(self):
        assert incompatibilities(video(frame_rate="30000/1001"), video(frame_rate="30/1")) == \
            ["частота кадров 30 вместо 29.97"]

    def test_several_problems_are_listed_together(self):
        problems = incompatibilities(video(), video(width=1280, height=720, sample_rate=44100))
        assert len(problems) == 2

    def test_audio_only_ignores_video_fields(self):
        assert incompatibilities(audio(), audio(width=1, height=1)) == []
