from enum import Enum
from pathlib import Path
from typing import Union, Tuple

from utils import apply_subprocess


class VideoPreset(Enum):
    ultrafast = 'ultrafast'
    superfast = 'superfast'
    veryfast = 'veryfast'
    faster = 'faster'
    fast = 'fast'
    medium = 'medium'
    slow = 'slow'
    slower = 'slower'
    veryslow = 'veryslow'


def compress_video(input_filename: Union[str, Path],
                   output_filename: Union[str, Path],
                   crf: int = 28,
                   preset: VideoPreset = VideoPreset.medium,
                   width: int = 224,
                   height: int = 224,
                   fps: int = 30,
                   sample_rate: int = 32000,
                   channels: int = 1) -> Tuple[bool, str]:
    command = ['ffmpeg',
               '-y',  # (optional) overwrite output file if it exists
               '-i', f'"{input_filename}"',
               '-filter:v',
               f'"scale=\'if(gt(a,1),trunc(oh*a/2)*2,{width})\':\'if(gt(a,1),{height},trunc(ow*a/2)*2)\'"',
               '-c:v', 'libx265',
               '-crf', f'{crf}',
               '-preset', preset.value,
               '-r', f'{fps}',
               '-ac', f'{channels}',
               '-ar', f'{sample_rate}',
               '-threads', '1',
               f'"{output_filename}"']
    _, log = apply_subprocess(command)
    return output_filename.exists(), log
