import gzip
import json
import logging
import os
import pickle
import random
import shutil
import time
import zipfile
from abc import ABC
from argparse import ArgumentParser, Namespace
from typing import Iterable, Any, Optional

import requests
from tqdm import tqdm

DOWNLOAD_DATA_URLS_MAPPER = {
    'convai2': {
        'url': 'http://parl.ai/downloads/convai2/convai2_fix_723.tgz',
        'file': 'convai2.tgz'
    }
}


class BaseCollector(ABC):

    def __init__(self, config: Namespace, logger_object: Optional[logging.Logger] = None):

        self.config = config
        self.logger = logger_object

        self.data_dir = os.path.join(os.getcwd(), self.config.data_dir)
        self.raw_data_dir = os.path.join(self.data_dir, 'raw_data')
        self.make_dir(path=self.raw_data_dir)

    @staticmethod
    def make_dir(path: str, override: bool = False):
        if override:
            shutil.rmtree(path, ignore_errors=True)
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

    @staticmethod
    def read_gz_file(path: str) -> Iterable[bytes]:
        file_object = gzip.open(path, mode='r')
        for sample in file_object:
            yield sample

    @staticmethod
    def pickling(python_object: Any, path: str):
        with open(path, mode='wb') as file_object:
            pickle.dump(python_object, file_object)

    @staticmethod
    def save_json_file(data: Any, path: str, file: str):

        outfile = os.path.join(path, file)

        with open(file=outfile, mode='w') as file_object:
            json.dump(data, file_object, ensure_ascii=False)

    def logging(self, message: str):
        if not self.config.no_verbose:
            if self.logger is not None:
                self.logger.info(msg=message)
            else:
                print(message)

    def download_file(self,
                      url: str,
                      path: str,
                      file: str,
                      re_download: bool = False):

        outfile = os.path.join(path, file)
        download_flag = not os.path.isfile(outfile) or re_download

        self.logging(f'Start downloading {url} to {outfile}')

        exp_back_off = [2 ** r for r in reversed(range(self.config.n_retry))]

        resume_file = outfile + '.part'
        progress_bar = tqdm(unit='B',
                            unit_scale=True,
                            desc='Downloading {}'.format(file),
                            disable=self.config.no_verbose)

        while download_flag and self.config.n_retry >= 0:

            resume = os.path.isfile(resume_file)
            if resume:
                resume_pos = os.path.getsize(resume_file)
                mode = 'ab'
            else:
                resume_pos = 0
                mode = 'wb'
            response = None

            with requests.Session() as session:
                try:
                    header = (
                        {'Range': 'bytes=%d-' % resume_pos, 'Accept-Encoding': 'identity'}
                        if resume
                        else {}
                    )
                    response = session.get(url, stream=True, timeout=5, headers=header)

                    # negative reply could be 'none' or just missing
                    if resume and response.headers.get('Accept-Ranges', 'none') == 'none':
                        resume_pos = 0
                        mode = 'wb'

                    total_size = int(response.headers.get('Content-Length', -1))
                    # server returns remaining size if resuming, so adjust total
                    total_size += resume_pos
                    progress_bar.total = total_size
                    done = resume_pos

                    with open(resume_file, mode) as f:
                        for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                            if total_size > 0:
                                done += len(chunk)
                                if total_size < done:
                                    # don't freak out if content-length was too small
                                    total_size = done
                                    progress_bar.total = total_size
                                progress_bar.update(len(chunk))
                        break
                except requests.exceptions.ConnectionError:
                    self.config.n_retry -= 1
                    progress_bar.clear()
                    if self.config.n_retry >= 0:
                        self.logging(f'Connection error, retrying. ({self.config.n_retry} retries left)')
                        time.sleep(exp_back_off[self.config.n_retry])
                    else:
                        self.logging('Retried too many times, stopped retrying')
                finally:
                    if response:
                        response.close()
        if self.config.n_retry < 0:
            raise RuntimeWarning('Connection broken too many times. Stopped retrying.')

        if download_flag and self.config.n_retry > 0:
            progress_bar.update(done - progress_bar.n)
            if done < total_size:
                raise RuntimeWarning(
                    'Received less data than specified in '
                    + 'Content-Length header for '
                    + url
                    + '.'
                    + ' There may be a download problem.'
                )
            shutil.move(resume_file, outfile)

        progress_bar.close()

    def unpacking(self,
                  path: str,
                  file: str,
                  data_type: Optional[str] = None,
                  delete_source: bool = True):

        self.logging(f'unpacking {file}')
        full_path = os.path.join(path, file)

        if data_type is None:
            if file.endswith('.tgz') or file.endswith('.tar.gz'):
                data_type = 'gz'
            elif file.endswith('.zip'):
                data_type = 'zip'
            else:
                raise ValueError('Specify data_type')

        if data_type in ['gz', 'tar']:
            shutil.unpack_archive(full_path, path)
        elif data_type == 'zip':
            with zipfile.ZipFile(full_path, '') as zip_ref:
                zip_ref.extractall(path)
        else:
            raise ValueError(f'Not available data_type. Current: {data_type}')

        if delete_source:
            os.remove(full_path)

    def download(self):
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError

    def run(self):
        self.download()
        self.collect()


class ConvAI2Collector(BaseCollector):

    RAW_TRAIN_FILE = 'train_none_original.txt'
    RAW_VALID_FILE = 'valid_none_original.txt'

    TRAIN_FILE = 'train.json'
    TRAIN_WITH_CANDIDATES_FILE = 'train_with_candidates.json'
    VALID_FILE = 'valid.json'
    VALID_WITH_CANDIDATES_FILE = 'valid_with_candidates.json'

    INDEX_TO_TEXT_FILE = 'index_to_text.json'

    def __init__(self, config: Namespace, logger_object: Optional[logging.Logger] = None):
        super().__init__(config=config, logger_object=logger_object)

        self.data_url = DOWNLOAD_DATA_URLS_MAPPER[self.config.data_source]['url']
        self.archive_file = DOWNLOAD_DATA_URLS_MAPPER[self.config.data_source]['file']
        self.archive_file_path = os.path.join(self.data_dir, self.archive_file)

        self.text_to_index = {
            'hello': 0
        }

    def parse_file(self, path: str, file: str):

        file = os.path.join(path, file)

        data = list()
        data_with_candidates = list()

        with open(file) as file_object:

            previous_n = 0

            context_indices = list()
            last_response_index = None

            while True:

                line = file_object.readline().strip()

                if not line:
                    break

                if 'your persona:' in line \
                        or "partner's persona:" in line:
                    continue

                space_index = line.find(' ')

                current_n = int(line[:space_index])
                sample = line[space_index + 1:].split('\t')

                input_phrase, response, _, candidates = sample
                candidates = candidates.split('|')

                if current_n < previous_n:
                    context_indices = list()
                    last_response_index = None

                if input_phrase == '__SILENCE__':
                    response_index = self.text_to_index.get(response,
                                                            len(self.text_to_index))
                    self.text_to_index[response] = response_index
                    last_response_index = response_index
                    context_indices.append(last_response_index)

                input_phrase_index = self.text_to_index.get(input_phrase,
                                                            len(self.text_to_index))
                self.text_to_index[input_phrase] = input_phrase_index

                response_index = self.text_to_index.get(response,
                                                        len(self.text_to_index))
                self.text_to_index[response] = response_index

                if last_response_index is not None:
                    data.append((
                        last_response_index,
                        input_phrase_index,
                        context_indices[:]
                    ))
                    context_indices.append(last_response_index)

                data.append((
                    input_phrase_index,
                    response_index,
                    context_indices[:] if context_indices else [0]
                ))

                context_indices.append(input_phrase_index)

                last_response_index = response_index

                for n in range(len(candidates)):
                    candidate_index = self.text_to_index.get(candidates[n], len(self.text_to_index))
                    self.text_to_index[candidates[n]] = candidate_index
                    candidates[n] = candidate_index

                data_with_candidates.append((
                    input_phrase_index,
                    response_index,
                    context_indices[:] if context_indices else [0],
                    candidates
                ))

                previous_n = current_n

        return data, data_with_candidates

    def download(self):
        if not os.path.exists(self.archive_file_path) or self.config.download:
            self.download_file(url=self.data_url, path=self.raw_data_dir, file=self.archive_file)
            self.unpacking(path=self.raw_data_dir, file=self.archive_file, delete_source=False)

    def collect(self):
        self.logging('Parsing train')
        train_data, train_data_with_candidates = self.parse_file(path=self.raw_data_dir,
                                                                 file=self.RAW_TRAIN_FILE)

        self.logging('Parsing valid')
        valid_data, valid_data_with_candidates = self.parse_file(path=self.raw_data_dir,
                                                                 file=self.RAW_VALID_FILE)

        index_to_text = {value: key for key, value in self.text_to_index.items()}

        if not self.config.no_shuffle:
            random.shuffle(train_data)
            random.shuffle(train_data_with_candidates)
            random.shuffle(valid_data)
            random.shuffle(valid_data_with_candidates)

        self.logging('Saving train')
        self.save_json_file(data=train_data,
                            path=self.data_dir,
                            file=self.TRAIN_FILE)
        self.save_json_file(data=train_data_with_candidates,
                            path=self.data_dir,
                            file=self.TRAIN_WITH_CANDIDATES_FILE)

        self.logging('Saving train')
        self.save_json_file(data=valid_data,
                            path=self.data_dir,
                            file=self.VALID_FILE)
        self.save_json_file(data=valid_data_with_candidates,
                            path=self.data_dir,
                            file=self.VALID_WITH_CANDIDATES_FILE)

        self.logging('Saving index_to_text')
        self.save_json_file(data=index_to_text,
                            path=self.data_dir,
                            file=self.INDEX_TO_TEXT_FILE)


if __name__ == '__main__':

    logger = logging.getLogger(__file__)

    parser = ArgumentParser()

    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--no_verbose', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--download', action='store_true')

    parser.add_argument('--n_retry', type=int, default=5)
    parser.add_argument('--chunk_size', type=int, default=32768)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.data_source == 'convai2':
        collector = ConvAI2Collector(config=args, logger_object=logger)
    else:
        raise ValueError('Not available data_source')

    collector.run()
