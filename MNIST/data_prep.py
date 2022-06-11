import shutil
import pathlib
import gzip

import idx2numpy

ARCHIVED_DATA_DIR = "archived_data"
DATA_DIR = "data"


def gunzip(file, work_dir):
    p_file = pathlib.Path(file)
    new_path = pathlib.Path(work_dir, p_file.stem)
    with gzip.open(file, 'rb') as g:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(new_path), 'wb') as f:
            shutil.copyfileobj(g, f)


if __name__ == "__main__":
    shutil.register_unpack_format("gz", [".gz"], gunzip)
    archive_formats = shutil.get_unpack_formats()
    for f in pathlib.Path(ARCHIVED_DATA_DIR).iterdir():
        try:
            print(f"Extracting {f}")
            shutil.unpack_archive(str(f), DATA_DIR)
        except shutil.ReadError:
            print(f"Couldn't extract {f}")
