import hashlib
import imageio
import logging
import os
import re
import sys
import tempfile

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from typing import List


class ImageSaver:
    def _write_file(self, filename, image_data):
        imageio.imwrite(filename, image_data.astype('uint8'))


class LocalFilesystemSaver(ImageSaver):
    def __init__(self, directory: str):
        self.logger = logging.getLogger()
        self._path = directory

    def __make_output_filename(self, image):
        r = re.compile(r'\.fits', re.IGNORECASE)
        basename_in = os.path.basename(image['src_filename'])
        basename_out = r.sub('', basename_in)
        basename_out += '.{0}.png'.format(image['subimage_index'])
        absolute_out = os.path.join(self._path, basename_out)
        return absolute_out

    def __call__(self, image) -> str:
        output_filename = self.__make_output_filename(image)

        self.logger.info('Exporting tile {0} from {1} to {2}'.format(
            image['subimage_index'],
            image['src_filename'],
            output_filename,
        ))

        self._write_file(output_filename, image['image_data'])

        return output_filename


class GoogleDriveUploader(ImageSaver):
    def __init__(self, folder_id: str):
        self.logger = logging.getLogger()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.logger.info(f'Temporary directory: {self._tmpdir}')

        self.logger.info('Authenticating to Google Drive')
        self._gauth = GoogleAuth()

        if 'google.colab' in sys.modules:
            self.__colab_auth()
        else:
            self._gauth.LocalWebserverAuth()

        self._drive = GoogleDrive(self._gauth)
        self._file_creation_params = {
            'parents': [{
                'id': folder_id,
            }],
        }

    def __colab_auth(self):
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        auth.authenticate_user()
        self._gauth.credentials = GoogleCredentials.get_application_default()

    def __make_temp_filename(self, image) -> str:
        image_hash = hashlib.sha256(image['image_data']).hexdigest()
        filename = f'{image_hash}.png'
        return os.path.join(self._tmpdir.name, filename)

    def _upload(self, filename):
        metadata = {
            **self._file_creation_params,
            **{'title': os.path.basename(filename)},
        }
        remote_file = self._drive.CreateFile(metadata)
        remote_file.SetContentFile(filename)
        remote_file.Upload()

    def __call__(self, image) -> str:
        tempfile = self.__make_temp_filename(image)
        self._write_file(tempfile, image['image_data'])
        self._upload(tempfile)
        os.remove(tempfile)
