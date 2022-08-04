import argparse
import json
import shutil
import subprocess
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Tuple, NamedTuple, List, Union, Optional, Dict

import pandas as pd
from joblib import delayed
from tqdm import tqdm

from utils import ProgressParallel


class VideoClipInfo(NamedTuple):
    video_id: str
    start_time: int
    end_time: int
    split: Optional[str]
    label: Optional[str]


def download_video(video_id: str,
                   output_filename: Path,
                   start_time: int,
                   end_time: int,
                   num_retries: int = 5,
                   post_process_args: Optional[Dict] = None,
                   url_base: str = 'https://www.youtube.com/watch?v=',
                   ) -> Tuple[bool, str]:
    assert len(video_id) == 11, f'video_identifier must have length 11 but got {video_id}'
    status = False

    if post_process_args is None:
        tmp_filename = output_filename
    else:
        tmp_filename = (output_filename.parent.parent / 'tmp') / f'{uuid.uuid4()}.mp4'

    command = ['yt-dlp',
               '--quiet',
               '--no-warnings',
               '--retries', str(num_retries),
               '-f', 'mp4',
               '--download-sections', f'*{start_time}-{end_time}',
               '-o', f'{tmp_filename}',
               f'{url_base + video_id}']
    try:
        subprocess.check_output(command, shell=False, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        return status, str(error.output)

    if post_process_args is not None:
        default = dict(crf='18', preset='veryslow', width='224', height='224')
        default.update(post_process_args)

        command = ['ffmpeg',
                   '-i', f'"{tmp_filename}"',
                   '-vf', f"scale={default['width']}:{default['height']}",
                   '-c:v', 'libx264',
                   '-crf', str(default['crf']),
                   '-preset', str(default['preset']),
                   '-c:a', 'copy',
                   '-threads', '1',
                   '-loglevel', 'panic',
                   f'"{output_filename}"']
        command = ' '.join(command)

        try:
            subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            return status, str(error.output)

        tmp_filename.unlink()

    status = output_filename.exists()

    return status, 'Downloaded'


def download_wrapper(row: VideoClipInfo,
                     label_to_dir: defaultdict[Path],
                     by: Optional[str] = None,
                     time_format: str = '05d',
                     num_retries: int = 5,
                     post_process_args: Optional[Dict] = None) -> Tuple[str, bool, str]:
    filename = get_video_filename(row, label_to_dir, by, time_format)

    if filename.exists():
        return row.video_id, True, 'Exists'

    downloaded, log = download_video(row.video_id, filename, row.start_time, row.end_time, num_retries,
                                     post_process_args)

    return row.video_id, downloaded, log


def parse_input(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    columns = dict([
        ('youtube_id', 'video_id'),
        ('time_start', 'start_time'),
        ('time_end', 'end_time'),
    ])
    df.rename(columns=columns, inplace=True)
    df['start_time'] = df['start_time'].apply(int)
    df['end_time'] = df['end_time'].apply(int)
    return df


def create_dir(dataset: pd.DataFrame,
               output_dir: Union[str, Path],
               by: Optional[str] = None) -> defaultdict[Path]:
    output_dir = Path(output_dir)
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    (output_dir / 'tmp').mkdir(exist_ok=False)

    label_to_dir = defaultdict(lambda: data_dir)

    if by is None:
        warnings.warn('No splitting scheme found. All files will be stored in {}'.format(data_dir))
    elif by in dataset.columns:
        for cat in dataset[by].unique():
            cat_dir = output_dir / cat
            cat_dir.mkdir(exist_ok=True)
            label_to_dir[cat] = cat_dir
    else:
        raise ValueError(f'No column named {by}')
    return label_to_dir


def get_video_filename(row: VideoClipInfo,
                       label_to_dir: defaultdict[Path],
                       by: Optional[str] = None,
                       time_format: str = '06d') -> Path:
    filename = f"{row.video_id}_{row.start_time:{time_format}}_{row.end_time:{time_format}}.mp4"
    if by is None:
        base = label_to_dir.default_factory()
    else:
        base = label_to_dir[getattr(row, by)]
    return base / filename


def main(csv_path: str,
         output_dir: str,
         structure_by: Optional[str] = None,
         time_format: str = '06d',
         retries: int = 5,
         n_jobs: int = 32,
         post_process_args: Optional[Dict] = dict(crf='18', preset='veryslow', width='224', height='224'),
         verbose: bool = True) -> List:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    dataset = parse_input(csv_path)
    label_to_dir = create_dir(dataset, output_dir, by=structure_by)

    it = dataset.itertuples(name='VideoClipInfo')

    if n_jobs == 1:
        status_list = []
        if verbose:
            it = tqdm(it, desc='Crawling from YT', total=dataset.shape[0])
        for row in it:
            status_list.append(
                download_wrapper(row, label_to_dir, structure_by, time_format, retries, post_process_args))
    else:
        status_list = ProgressParallel(use_tqdm=verbose, total=dataset.shape[0], n_jobs=n_jobs, prefer='threads')(
            delayed(download_wrapper)(row, label_to_dir, structure_by, time_format, retries, post_process_args) for row
            in it
        )

    default_dir = label_to_dir.default_factory()
    if len(list(default_dir.iterdir())) == 0:
        default_dir.rmdir()

    shutil.rmtree((output_dir / 'tmp'))

    report_path = output_dir / (csv_path.stem + '_download_report.json')
    with open(report_path, 'w') as f:
        json.dump(status_list, f)

    print(f"Downloaded {len(list(filter(lambda x: x[1], status_list)))} / {len(status_list)} videos. See {report_path} "
          f"for download report.")
    return status_list


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('csv_path', type=str,
                        help=('CSV file containing the following columns: '
                              'youtube_id, time_start, time_end (optional: label, split)'))
    parser.add_argument('output_dir', type=str,
                        help='Output directory where videos will be saved.')
    parser.add_argument('-b', '--by', type=str, default=None, dest='structure_by',
                        help=('Column of the CSV to be used to create a folder structure of the dataset. A class label '
                              'or the dataset split would be useful here. Default: all videos are stored in one dir'))
    parser.add_argument('-f', '--time-format', type=str, default='06d',
                        help='Format of the timestamp in the filename of the videos')
    parser.add_argument('-n', '--n_jobs', type=int, default=32,
                        help='Number of jobs')
    parser.add_argument('-r', '--retries', type=int, default=10,
                        help='Number of retries for downloading one video')
    parser.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                        help='Prevent the display of a progress bar')


    main(**vars(parser.parse_args()))
