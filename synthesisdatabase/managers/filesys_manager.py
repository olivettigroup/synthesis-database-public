from autologging import (logged, traced, TRACE)
from os.path import (join, relpath)
from datetime import (datetime)
from logging import (basicConfig)
from re import (search)
import os

@logged
class FilesysManager:
  def dir_setup(self, root_dir, files_dir, logs_dir):
    base_dir = root_dir

    files_dir = join(base_dir, files_dir)
    logs_dir = join(base_dir, logs_dir)
    misc_dir = join(files_dir, 'misc/')

    doi_pdf_map = join(misc_dir, 'doi_pdf_map.json')
    doi_fail_log = join(misc_dir, 'doi_fail_log.json')
    pdf_files_dir = join(files_dir, 'pdfs/')
    html_files_dir = join(files_dir, 'htmls/')

    for directory in [files_dir, pdf_files_dir, html_files_dir, logs_dir, misc_dir]:
      if not os.path.exists(directory):
        os.makedirs(directory)

    return (
      pdf_files_dir,
      html_files_dir,
      logs_dir,
      doi_pdf_map,
      doi_fail_log
      )

  def log_setup(self, logs_dir):
    basicConfig(level=TRACE, filename=datetime.now().strftime(logs_dir + '%Y_%m_%d_%Hh_%Mm_%Ss.log'),
    format="%(levelname)s:%(name)s:%(funcName)s:%(message)s")

    prov_file = open(datetime.now().strftime(logs_dir + '%Y_%m_%d_%Hh_%Mm_%Ss.meta'), 'wb')

    return prov_file
