import argparse
import json
import shutil
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Tuple, NamedTuple, List, Union, Optional, Dict

import pandas as pd
from joblib import delayed
from tqdm import tqdm

from postprocess import compress_video, VideoPreset
from utils import ProgressParallel, apply_subprocess


class VideoClipInfo(NamedTuple):
    video_id: str
    start_time: Optional[int]
    end_time: Optional[int]
    split: Optional[str]


def download_video(video_id: str,
                   output_filename: Path,
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None,
                   num_retries: int = 5,
                   postprocess: bool = False,
                   post_process_args: Dict = {},
                   url_base: str = 'https://www.youtube.com/watch?v=',
                   ) -> Tuple[bool, str]:
    assert len(video_id) == 11, f'video_identifier must have length 11 but got {video_id}'

    if postprocess:
        tmp_filename = (output_filename.parent.parent / 'tmp') / f'{uuid.uuid4()}.mp4'
    else:
        tmp_filename = output_filename

    section = f'--download-sections *{start_time}-{end_time}' if start_time is not None and end_time is not None else ''

    command = ['yt-dlp',
               '--quiet',
               '--no-warnings',
               '--retries', f'{num_retries}',
               '-f', 'mp4',
               section,
               '-o', f'{tmp_filename}',
               f'{url_base + video_id}']
    downloaded, log = apply_subprocess(command)

    if postprocess and downloaded:
        post_success, post_log = compress_video(tmp_filename, output_filename, **post_process_args)

        # remove tmp file
        tmp_filename.unlink()
        return post_success, log + 'Postprocess: ' + post_log

    status = output_filename.exists()
    return status, log


def download_wrapper(row: VideoClipInfo,
                     label_to_dir: defaultdict[Path],
                     by: Optional[str] = None,
                     time_format: str = '05d',
                     num_retries: int = 5,
                     postprocess: bool = False,
                     post_process_args: Dict = {}) -> Tuple[str, bool, str]:
    filename = get_video_filename(row, label_to_dir, by, time_format)

    if filename.exists():
        return row.video_id, True, 'exists'

    downloaded, log = download_video(row.video_id, filename, row.start_time, row.end_time, num_retries,
                                     postprocess, post_process_args)

    return row.video_id, downloaded, log


def parse_input(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'youtube_id': 'video_id', 'time_start': 'start_time', 'time_end': 'end_time'})
    df = df.drop(columns=df.columns.difference(VideoClipInfo._fields))
    df['start_time'] = df['start_time'].apply(int) if 'start_time' in df.columns else None
    df['end_time'] = df['end_time'].apply(int) if 'end_time' in df.columns else None
    return df


def create_dir(dataset: pd.DataFrame,
               output_dir: Union[str, Path],
               by: Optional[str] = None) -> defaultdict[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    (output_dir / 'tmp').mkdir(exist_ok=True)

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
    if row.start_time is None and row.end_time is None:
        filename = f"{row.video_id}.mp4"
    else:
        filename = f"{row.video_id}_{row.start_time:{time_format}}_{row.end_time:{time_format}}.mp4"

    base = label_to_dir.default_factory() if by is None else label_to_dir[getattr(row, by)]

    return base / filename


def main(csv_path: str,
         output_dir: str,
         structure_by: Optional[str] = None,
         time_format: str = '06d',
         retries: int = 5,
         n_jobs: int = 32,
         postprocess: bool = False,
         post_process_args: Dict = {},
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
                download_wrapper(row, label_to_dir, structure_by, time_format, retries, postprocess,
                                 post_process_args))
    else:
        status_list = ProgressParallel(use_tqdm=verbose, total=dataset.shape[0], n_jobs=n_jobs, prefer='threads')(
            delayed(download_wrapper)(row, label_to_dir, structure_by, time_format, retries, postprocess,
                                      post_process_args) for row in it
        )

    default_dir = label_to_dir.default_factory()
    if len(list(default_dir.iterdir())) == 0:
        default_dir.rmdir()

    shutil.rmtree((output_dir / 'tmp'))

    report_path = output_dir / (csv_path.stem + '_download_report.json')
    report = dict(
        csv_path=str(csv_path),
        output_dir=str(output_dir),
        structure_by=structure_by,
        time_format=time_format,
        retries=retries,
        n_jobs=n_jobs,
        postprocess=postprocess,
        post_process_args=post_process_args,
        status_list=status_list
    )
    with open(report_path, 'w') as f:
        json.dump(report, f)

    print(f"Downloaded {len(list(filter(lambda x: x[1], status_list)))} / {len(status_list)} videos. See {report_path} "
          f"for download report.")
    return status_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helper script for downloading and trimming videos.')
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
    parser.add_argument('-r', '--retries', type=int, default=5,
                        help='Number of retries for downloading one video')
    parser.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                        help='Prevent the display of a progress bar')

    post_parsers = parser.add_subparsers(required=False, description='Postprocessing arguments',
                                         dest='postprocess')
    post_parser = post_parsers.add_parser('postprocess', help='Postprocess the downloaded videos')
    post_parser.add_argument('--width', help='Width of the output video', type=int, default=224)
    post_parser.add_argument('--height', help='Height of the output video', type=int, default=224)
    post_parser.add_argument('--codec', help='Codec', type=str, default='libx265')
    post_parser.add_argument('--crf', help='CRF', type=int, default=28)
    post_parser.add_argument('--preset', help='Preset', type=VideoPreset, default=VideoPreset.fast)
    post_parser.add_argument('--fps', help='Frames per second', type=int, default=30)
    post_parser.add_argument('--ar', help='Audio sample rate', type=int, default=32000)
    post_parser.add_argument('--ac', help='Number of audio channels', type=int, default=1)

    args = vars(parser.parse_args())
    if args.pop('postprocess') is not None:
        args['post_process_args'] = {}
        args['postprocess'] = True
        for k in ['width', 'height', 'codec', 'crf', 'preset', 'fps', 'ar', 'ac']:
            args['post_process_args'][k] = args.pop(k)

    print(args)
    main(**args)
