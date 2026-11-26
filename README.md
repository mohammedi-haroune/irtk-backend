# irtk-backend

Information Retrieval Tool-Kit, the Django REST API.

IRTK lets you upload a set of text documents (a corpus), or pick the NLTK
Brown corpus, and query it with classic information retrieval models:

- term search, with each document's frequency and TF-IDF score
- the boolean model (`web and (pdf or video)`, `module and not besoin`)
- the vector model, with inner product, Dice, cosine and Jaccard scores

The web interface is [irtk-frontend](https://github.com/mohammedi-haroune/irtk-frontend).

## Requirements

- Python 3.10 or newer (tested with 3.12)

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

`migrate` creates `db.sqlite3`, which only stores sessions. The session
remembers which corpus each browser selected.

To offer the Brown corpus as well, download it once (about 3 MB):

```
python -m nltk.downloader brown
```

## Run

```
python manage.py runserver
```

The API listens on http://127.0.0.1:8000/polls/. Start the frontend next; its
dev server forwards `/polls/` to this port.

## Corpora

Each folder in `corporas/` is a corpus: one `.txt` file per document and an
optional `README` with its description. `Corpus Test` holds four short French
texts about information retrieval. Corpora uploaded from the web interface are
saved here too.

## API

| Method | Path | Parameters | Returns |
|---|---|---|---|
| GET | `corporas/` | | every corpus with its description and files |
| POST | `select_corpus/` | `CORPUS_NAME` | 202, and the session remembers the corpus |
| GET | `selected_corpus/` | | the selected corpus, or 404 |
| GET | `docs/` | | each document with its first 200 characters |
| GET | `find/` | `term` | frequency and TF-IDF of the term per document |
| GET | `boolean_model/` | `query` | documents matching the query, or 400 for a bad query |
| GET | `vector_model/` | `query` | the four similarity scores per document |
| POST | `upload_file` | `CORPUS_NAME`, `CORPUS_DESCRIPTION`, `file` | 202 |

Every query endpoint answers 404 until a corpus is selected.

## Tests

```
python manage.py test polls
```

The tests run the models on `Corpus Test`, check that TF still follows the
2018 formula, and cover selection, every query endpoint and uploads.

## Changes in 2026

The project had not run since 2018. To bring it back:

- Moved to Django 5.2 LTS, Django REST framework 3.18, django-cors-headers 4.9
  and nltk 3.10. Django 2.1 imports `distutils`, which Python 3.12 removed.
- The session used to hold the whole corpus as a pickled Python object, and
  Django 5 removed the pickle session serializer. The session now stores the
  corpus name, and the server keeps one models object per corpus in memory.
- Selecting an NLTK corpus never worked, because `nltk` was not imported in the
  view. It now looks the corpus up by name instead of calling `eval` on the
  request.
- nltk 3.9 and later refuse to read corpora outside their data paths, so the
  `corporas/` folder is added to `nltk.data.path`.
- The boolean model still evaluates the query with `eval`, but only after
  checking that nothing besides `True`, `False`, `and`, `or`, `not` and
  parentheses is left.
- Upload rejects corpus names containing `/` or starting with `.`, which could
  write outside `corporas/`.
- TF used to count every word of a document for every word of the document.
  It now reads the counts computed when the corpus loads, and IDF is cached per
  term. On the Brown corpus (500 documents) a vector query went from 1.1 s to
  0.07 s, with the same scores.
- `db.sqlite3` is no longer committed. It held 39 MB of expired 2018 sessions.
- Added tests and removed debug prints.
