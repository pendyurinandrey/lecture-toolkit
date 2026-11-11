"""Сведения о медиафайле: тип (видео или аудио) и параметры потоков, от которых
зависит склейка без перекодирования.

Тип определяется по СОДЕРЖИМОМУ, а не по расширению: в .m4a/.mp3 бывает встроенная
обложка (её ffprobe показывает как видеопоток с признаком attached_pic — это не видео),
а .mp4 бывает без видеопотока.

Склеивать без перекодирования можно только файлы с одинаковыми параметрами потоков.
Если параметры разные, ffmpeg склеивает молча и портит результат (проверено: 48 кГц
стерео + 44.1 кГц моно дали итог на 15 секунд короче суммы, код возврата 0), поэтому
несовместимость нужно ловить заранее: см. incompatibilities().

Частоту кадров НЕ сравниваем: в MP4 у каждого кадра своя метка времени, и при склейке без
перекодирования она сохраняется (проверено на записях экрана с переменной частотой ~24.3 к/с,
у которых ffprobe оценивает r_frame_rate по-разному: 97/4 и 73/3, — склейка корректна).
Зато сравниваем параметры кодека (extradata: SPS/PPS у H.264/HEVC): в MP4 хранится один набор
на весь поток, и при разных наборах картинка после места склейки молча портится.

Модуль чистый — ничего не запускает; данные ffprobe приходят из ffmpeg.probe_media().
"""

from dataclasses import dataclass
from typing import Optional

VIDEO_EXTENSIONS = (".mp4",)
AUDIO_EXTENSIONS = (".m4a", ".mp3")
# Расширение выходного аудиофайла определяется кодеком: без перекодирования AAC кладётся
# в .m4a, MP3 — в .mp3.
AUDIO_CODEC_EXTENSIONS = {"aac": ".m4a", "mp3": ".mp3"}

_KIND_NAMES = {"video": "видео", "audio": "аудио"}


@dataclass(frozen=True)
class MediaInfo:
    kind: str                          # "video" | "audio"
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    video_codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    pix_fmt: Optional[str] = None
    video_extradata: Optional[str] = None  # хеш параметров кодека видео, как в ffprobe: "MD5:2c5b…"


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_probe(data: dict) -> MediaInfo:
    """MediaInfo из JSON-вывода `ffprobe -show_entries stream=... -of json`.
    ValueError, если в файле нет ни видео-, ни аудиопотока."""
    streams = data.get("streams") or []
    video = [s for s in streams
             if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not video and not audio:
        raise ValueError("в файле нет ни видео-, ни аудиопотока")

    fields = {"kind": "video" if video else "audio"}
    if audio:
        fields.update(audio_codec=audio[0].get("codec_name"),
                      sample_rate=_to_int(audio[0].get("sample_rate")),
                      channels=_to_int(audio[0].get("channels")))
    if video:
        fields.update(video_codec=video[0].get("codec_name"),
                      width=_to_int(video[0].get("width")), height=_to_int(video[0].get("height")),
                      pix_fmt=video[0].get("pix_fmt"), video_extradata=video[0].get("extradata_hash"))
    return MediaInfo(**fields)


def audio_extension(info: MediaInfo) -> Optional[str]:
    """Расширение выходного файла для склейки аудио без перекодирования (None — формат не поддержан)."""
    return AUDIO_CODEC_EXTENSIONS.get(info.audio_codec)


def _codec(name: Optional[str]) -> str:
    return name.upper() if name else "нет"


def incompatibilities(reference: MediaInfo, other: MediaInfo) -> list:
    """Чем other отличается от reference так, что склейка без перекодирования невозможна.
    Пустой список — файлы совместимы. Тексты — для показа пользователю."""
    if reference.kind != other.kind:
        return [f"это {_KIND_NAMES[other.kind]}, а первый файл — {_KIND_NAMES[reference.kind]}; "
                "смешивать видео и аудио нельзя"]

    problems = []
    if reference.audio_codec is None or other.audio_codec is None:
        if reference.audio_codec != other.audio_codec:
            problems.append("в файле нет звука, а в первом есть" if other.audio_codec is None
                            else "в файле есть звук, а в первом нет")
    else:
        if reference.audio_codec != other.audio_codec:
            problems.append(f"кодек звука {_codec(other.audio_codec)} вместо {_codec(reference.audio_codec)}")
        if reference.sample_rate != other.sample_rate:
            problems.append(f"частота звука {other.sample_rate} Гц вместо {reference.sample_rate} Гц")
        if reference.channels != other.channels:
            problems.append(f"каналов звука: {other.channels} вместо {reference.channels}")

    if reference.kind == "video":
        video_problems = len(problems)
        if reference.video_codec != other.video_codec:
            problems.append(f"кодек видео {_codec(other.video_codec)} вместо {_codec(reference.video_codec)}")
        if (reference.width, reference.height) != (other.width, other.height):
            problems.append(f"разрешение {other.width}×{other.height} вместо {reference.width}×{reference.height}")
        if reference.pix_fmt != other.pix_fmt:
            problems.append(f"формат пикселей {other.pix_fmt} вместо {reference.pix_fmt}")
        # Хеш сравниваем, только когда он известен у обоих: без него проверять нечего.
        # Если уже отличаются кодек, разрешение или формат пикселей, параметры кодека заведомо другие — не дублируем.
        if (reference.video_extradata and other.video_extradata and len(problems) == video_problems
                and reference.video_extradata != other.video_extradata):
            problems.append("параметры кодека видео отличаются (запись сделана с другими настройками)")
    return problems
