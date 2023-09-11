import os
import ujson

from decoder.decoder_utils import *  # noqa


def decodeSave(filename, verbose=True, overwrite=False):
    """Decode and save decoded data in json format.

    Parameters:
        filename (str): access path + filename of the file to decode.
        verbose (bool): if True, print status infos during decoding.
        overwrite (bool): if False, an already decoded file will not be decoded again.

    Returns:
        Boolean: True if decoding is successful, else False.
    """
    fileout = filename.replace(".txt", ".json")
    if not overwrite and os.path.isfile(fileout):
        print("File already decoded (set overwrite to true to decode)")
        return True

    try:
        jsonData = decode(filename, verbose)
        print("Write to {}...".format(fileout))
        with open(fileout, "w") as f:
            f.write(ujson.dumps(jsonData))
        return True
    except Exception:
        return False


def decodeSaveFolder(path, verbose=True, overwrite=False):
    """Decode and save all txt files in a folder.

    Parameters:
        path (str): access path of the folder.
        verbose (bool): if True, print status infos during decoding.
        overwrite (bool): if False, an already decoded file will not be decoded again.
    """
    file_names = [file for file in os.listdir(path) if (file[-4:] == ".txt")]
    for file in file_names:
        res = decodeSave(path + file, verbose, overwrite)
        if not res:
            print(f"Could not decode file {file} ...")
    return
