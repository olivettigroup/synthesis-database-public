# The Synthesis Project - Database

The Synthesis Project aims to catalog and predict materials synthesis routes. This repository contains code for populating a materials synthesis database from a corpus of PDF/HTML journal articles. The database is exposed to the public via the Synthesis Project API (not contained in this repository).

This repository includes the following:

+ Higher-level scripts and wrappers for downloading a corpus of articles (with dependencies on `olivettigroup/article-downloader`)
+ Objects and scripts for extracting text and parsing / entity recognition / etc. for materials science text
