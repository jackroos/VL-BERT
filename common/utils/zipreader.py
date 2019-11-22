import zipfile
import os
import io
import time
from PIL import Image


class ZipReader(object):
    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path in zip_bank:
            return zip_bank[path]
        else:
            print("creating new zip_bank")
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
            return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_zip_at = path.index('.zip@')
        if pos_zip_at == len(path):
            print("character '@' is not found from the given path '%s'" % (path))
            assert 0
        pos_at = pos_zip_at + len('.zip@') - 1

        zip_path = path[0: pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
               len(os.path.splitext(file_foler_name)[-1]) == 0 and \
               file_foler_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path)+1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=['.*']):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and str.lower(os.path.splitext(file_foler_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path)+1:])

        return file_lists

    @staticmethod
    def imread(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        im = Image.open(io.BytesIO(data))
        return im

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data
