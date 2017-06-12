import sys
sys.path.append("..")

from models import (Paper, Paragraph)
from time import sleep
from pymongo import MongoClient
from pymatgen.matproj.rest import (MPRester)
from articledownloader.articledownloader import (ArticleDownloader)
from os import (environ, path, remove, listdir, strerror)
from autologging import (logged, traced)
from json import (dumps, loads)
from bson.objectid import (ObjectId)
from re import (search, sub)
from functools import (wraps)
from bs4 import BeautifulSoup
import bibtexparser
import time

reload(sys)
sys.setdefaultencoding('utf8')


@logged
class DownloadManager:
  ad = ArticleDownloader(environ.get('ELS_API_KEY'), environ.get('CRF_API_KEY'))
  connection = MongoClient()
  dl_doi_pdf_map = {}
  doi_fails = []
  dl_dois = []
  rows_per_query = 0

  def __init__(self, db):
    self.db = db

  def set_dois_per_query(self, num_docs):
    self.rows_per_query = int(num_docs)


  def get_dois(self, queries, mode, wait_time=0):
    if mode == 'mp':
      self.__logger.info( 'Searching with MP queries' )
      mpr = MPRester(environ.get('MAPI_KEY'), endpoint="https://www.materialsproject.org/rest")
      mpids = []

      for query in queries:
        try:
          entries = mpr.get_entries(query)
          for entry in entries:
            mpids.extend(entry.data['task_ids'])
          for mpid in mpids:
            mpid = mpr.get_materials_id_from_task_id(mpid)['materials_id']
            bibtex = mpr.get_materials_id_references(mpid)
            parsed_bibtex = bibtexparser.loads(bibtex)
            for item in parsed_bibtex.entries:
              if 'doi' in item:
                if item['doi'] not in self.dl_dois:
                  self.dl_dois.append(item['doi'])
        except:
          self.__logger.warning( 'FAILURE: Failed to get DOIs from MP:' + str(query) )
    elif mode == 'cr':
      self.__logger.info( 'Searching with CR queries' )

      for query in queries:
        dl_dois = []
        try:
          dl_dois = self.ad.get_dois_from_search(query, rows=self.rows_per_query)
        except Exception, e:
          self.__logger.warning( 'FAILURE: Failed to get DOIs from CR: ' + str(query) )
          self.__logger.warning( 'EXCEPTION: ' + str(e) )

        sleep(wait_time)
        self.dl_dois.extend(dl_dois)
    elif mode == 'issn':
      self.__logger.info( 'Searching with ISSN queries' )

      for query in queries:
        dl_dois = []
        try:
          dl_dois = self.ad.get_dois_from_journal_issn(query, rows=self.rows_per_query, pub_after=1900)
        except Exception, e:
          self.__logger.warning( 'FAILURE: Failed to get DOIs from CR by ISSN: ' + str(query) )
          self.__logger.warning( 'EXCEPTION: ' + str(e) )

        sleep(wait_time)
        self.dl_dois.extend(dl_dois)

  def build_doi_pdf_map(self, doi_pdf_map):
    self.dl_dois = set(self.dl_dois)
    self.__logger.info( str(len(self.dl_dois)) + ' DOIs retrieved. Setting up PDF map...')

    for doi in self.dl_dois:
      safe_doi = str(doi).translate(None, '/.()')
      self.dl_doi_pdf_map[str(safe_doi) + '.pdf'] = doi

    #Load up old DOIs
    if path.isfile(doi_pdf_map):
      with open(doi_pdf_map, 'rb') as f:
        old_map = loads(f.read())
        for key in old_map:
          if not key in self.dl_doi_pdf_map:
            self.dl_doi_pdf_map[key] = old_map[key]

    with open(doi_pdf_map, 'wb') as f:
      f.write(dumps(self.dl_doi_pdf_map))

    self.__logger.info(dumps(self.dl_doi_pdf_map, sort_keys=True, indent=2))


  def get_articles(self, files_dir, doi_pdf_map, doi_fail_log, file_ext='.pdf'):
    total_dls, success_dls, fail_dls = 0, 0, 0

    self.__logger.info('Attempting to download...')
    for doi in self.dl_dois:
      safe_doi = str(doi).translate(None, '/.()')
      filename = path.join(files_dir, str(safe_doi) + file_ext)
      if not path.isfile(filename) and not path.isdir(filename + ".d"):
        try:
          download_success = self.get_article(doi, filename)
          if download_success:
            success_dls += 1
          else:
            self.__logger.warning('FAILURE: Unable to download file: ' + str(doi))
            remove(filename)
            self.doi_fails.append(doi)
            fail_dls += 1
        except:
          self.__logger.warning('FAILURE: Error while trying to download file: ' + str(doi))
          remove(filename)
          self.doi_fails.append(doi)
          fail_dls += 1

      total_dls += 1

    self.__logger.info(dumps(self.doi_fails, sort_keys=True, indent=2))

    with open(doi_fail_log, 'wb') as f:
      f.write(dumps(self.doi_fails))

    self.__logger.info( 'Total attempted downloads: ' + str(total_dls) )
    self.__logger.info( 'Total successful downloads: ' + str(success_dls) )
    self.__logger.info( 'Total failed downloads: ' + str(fail_dls) )

  def get_article(self, doi, filename, els_delay=1, wil_delay=15):
    writefile = open(filename, 'wb')
    download_success = False

    #Use DOI prefixes to optimize downloading attempts
    els_dois = ['10\.1016', '10\.1006']
    wil_dois = ['10\.1002', '10\.1111']
    spr_dois = ['10\.1007', '10\.1140', '10\.1891']
    rsc_dois = ['10\.1039']
    ecs_dois = ['10\.1149']
    nat_dois = ['10\.1038']

    #Use blacklist (unsubbed journals) to speed up downloading
    blacklist = [
      '10\.1002\/chin',
      '10\.1002\/ange',
      '10\.1002\/apj',
      '10\.1002\/elsc',
      '10\.1002\/ffj',
      '10\.1002\/cjoc',
    ]

    blacklist_match = any([search(d, doi) for d in blacklist])
    if blacklist_match: return False

    els_match = any([search(d, doi) for d in els_dois])
    wil_match = any([search(d, doi) for d in wil_dois])
    spr_match = any([search(d, doi) for d in spr_dois])
    rsc_match = any([search(d, doi) for d in rsc_dois])
    ecs_match = any([search(d, doi) for d in ecs_dois])
    nat_match = any([search(d, doi) for d in nat_dois])

    if wil_match:
      download_success = self.ad.get_html_from_doi(doi, writefile, 'wiley')
      sleep(wil_delay)
    if els_match:
      download_success = self.ad.get_html_from_doi(doi, writefile, 'elsevier')
      sleep(els_delay)
    elif rsc_match:
      download_success = self.ad.get_html_from_doi(doi, writefile, 'rsc')
    # elif ecs_match:
    #   download_success = self.ad.get_pdf_from_doi(doi, writefile, 'ecs')
    elif spr_match:
      download_success = self.ad.get_html_from_doi(doi, writefile, 'springer')
    elif nat_match:
      download_success = self.ad.get_html_from_doi(doi, writefile, 'nature')

    if writefile.tell() == 0:
      writefile.close()
      return False #Empty file reporting

    writefile.close()

    return download_success

  def is_title_relevant(self, title):
    title = title.lower()

    irrelevant_words = [
      "medical",
      "dna",
      "rna",
      "protein",
      "bacteria",
      "biomedicine",
      "bioassay",
      "cellular",
    ]

    for word in irrelevant_words:
      if word in title:
        return False

    return True

  def save_papers(self, pdf_files_dir, html_files_dir, doi_pdf_map, collection='papers', overwrite=False, file_locs=[], para_classifier=None):
    self.dl_doi_pdf_map = loads(open(doi_pdf_map, 'rb').read())

    for filename in file_locs:
      doi = ''
      is_html_file = bool(filename[-4:] == 'html')
      if is_html_file:
        if filename[:-5] + '.pdf' in self.dl_doi_pdf_map:
          doi = self.dl_doi_pdf_map[filename[:-5] + '.pdf']
      else: #PDF file
        if filename[:-2] in self.dl_doi_pdf_map:
          doi = self.dl_doi_pdf_map[filename[:-2]]

      if doi == '':
        if not is_html_file:
          doi = filename[:-6] #strip the file suffix and use safe_doi as doi
        else:
          doi = filename[:-5]
        self.__logger.info("INFO: Used backup DOI (not in map): " + str(doi))

      if self.connection[self.db][collection].find({'doi': doi}).count() == 1 and not overwrite:
        self.__logger.info("SKIPPED: Not overwriting and paper already in DB: " + str(doi))
        continue

      try:
        paper = open(path.join(pdf_files_dir, filename, 'docseg.json')).read()
        try:
          plaintext = loads(unicode(paper), strict=False)
        except:
          self.__logger.warning("FAILURE: Invalid JSON from watr-works: " + str(doi))
          pass
      except:
        self.__logger.warning("FAILURE: No docseg found from watr-works: " + str(doi))
        pass

      safe_doi = str(doi).translate(None, '/.()')

      title = None
      if title is None:
        title_match = self.connection[self.db].doi_title_abstract_map.find_one({'doi': doi}, {'title': True})
        if title_match is not None:
          title = title_match['title']

      if title is None: title = self.ad.get_title_from_doi(doi, 'crossref')
      if title is None: title = unicode('')

      if not self.is_title_relevant(title):
        self.__logger.info("WARNING: Irrelevant title detected; paper skipped: " + str(doi))
        continue

      try:
        abstract = unicode('')

        if abstract == '':
          abstract_match = self.connection[self.db].doi_title_abstract_map.find_one({'doi': doi},{'abstract':True})
          if abstract_match is not None:
            abstract = abstract_match['abstract']

        if abstract == '':
          #Use DOI prefixes to optimize downloading attempts
          els_dois = ['10\.1016', '10\.1006']

          if any([search(d, doi) for d in els_dois]):
            abstract = self.ad.get_abstract_from_doi(doi, 'elsevier')
          else:
            abstract = unicode('')

        if abstract is None: abstract = unicode('')

        new_paper = Paper()
        del new_paper['_id'] #prevents duplication; ID assigned on insertion
        new_paper['doi'] = doi
        new_paper['abstract'] = abstract
        new_paper['title'] = title
        if not is_html_file:
          new_paper['pdf_loc'] = unicode(path.join(pdf_files_dir, filename, safe_doi + '.pdf'))
        else:
          new_paper['pdf_loc'] = unicode(path.join(html_files_dir, filename, safe_doi + '.html'))
        new_paper['modified'] = int(time.time())
        new_paper['paragraphs'] = []

        #Compute paragraphs
        html_paragraphs_used = False
        recipe_found = False

        #Override to use HTML paragraphs when available
        if path.isfile(path.join(html_files_dir, safe_doi + '.html')):
          html_text = open(path.join(html_files_dir, safe_doi + '.html'), 'rb').read()
          soup = BeautifulSoup(html_text, 'html.parser')
          paragraphs = soup.find_all('p') + soup.find_all('div', {'class':'NLM_p'}) + soup.find_all('span')
          paragraphs = [p.getText() for p in paragraphs]
          paragraphs = [p.replace('\n','').replace('\t','') for p in paragraphs]
          paragraphs = [p for p in paragraphs if len(p) > 80]

          if len(paragraphs) > 20:
            for paragraph in paragraphs:
              new_paragraph = Paragraph()
              new_paragraph['_id'] = unicode(ObjectId())
              if para_classifier.predict_one(paragraph):
                new_paragraph['type'] = unicode('recipe')
                recipe_found = True
              new_paragraph['text'] = paragraph
              new_paper['paragraphs'].append(new_paragraph)

            html_paragraphs_used = True
            self.__logger.info("INFO: Used HTML paragraphs for paper: " + str(doi))

        if not html_paragraphs_used:
          para_label_ids = []
          for line in plaintext['labels']:
            if line[0] == 'ds:para-begin':
              para_label_ids.append(line[1][0])

          para_label_iter = iter(para_label_ids)
          try:
            next_label = next(para_label_iter)
          except:
            self.__logger.warning("WARNING: No paragraphs detected in file: " + str(doi))
            continue

          current_para = ''
          for line in plaintext['lines']:
            if line[2] == next_label:
              if current_para != '':
                new_paragraph = Paragraph()
                new_paragraph['_id'] = unicode(ObjectId())
                if para_classifier.predict_one(current_para):
                  new_paragraph['type'] = unicode('recipe')
                  recipe_found = True
                new_paragraph['text'] = current_para
                new_paper['paragraphs'].append(new_paragraph)

                current_para = ''

              try:
                next_label = next(para_label_iter)
              except:
                break

            for token in line[0]:
              if search('{.*?}', token) is not None: token = sub('[{}_^]', '', token)
              current_para += token + ' '

      except Exception, e:
        self.__logger.warning('FAILURE: Unable to save paper: ' + str(doi))
        self.__logger.warning('ERR_MSG: ' + str(e))
        continue

      if len(new_paper['paragraphs']) == 0:
        self.__logger.warning('WARNING: No paragraphs found; skipping paper: ' + str(doi))
        continue

      if not recipe_found:
        self.__logger.warning('WARNING: No recipe found; skipping paper: ' + str(doi))
        continue

      if self.connection[self.db][collection].find({'doi': doi}).count() == 0:
        self.connection[self.db][collection].insert_one(new_paper)
      elif self.connection[self.db][collection].find({'doi': doi}).count() == 1 and overwrite and not used_backup_doi:
        self.connection[self.db][collection].update_one({'doi': doi},  {'$set': new_paper})

      if self.connection[self.db].doi_title_abstract_map.find({'doi': doi}).count() == 0:
        self.connection[self.db].doi_title_abstract_map.insert_one({
          'doi': new_paper['doi'],
          'title': new_paper['title'],
          'abstract': new_paper['abstract']
        })
