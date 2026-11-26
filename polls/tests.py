import os
import shutil
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from nltk.corpus.reader import PlaintextCorpusReader

from polls import views
from polls.irreader import IRModels

SAMPLE_CORPUS = 'Corpus Test'


class IRModelsTests(TestCase):
    def setUp(self):
        corpus = PlaintextCorpusReader(os.path.join(views.CORPORAS_ROOT, SAMPLE_CORPUS), r'.*\.txt')
        self.ir = IRModels(corpus)

    def test_tf_matches_the_original_formula(self):
        # the 2018 code computed freq(term) / max(freq(t) for t in text) by scanning the text
        for fileid in self.ir._corpus.fileids():
            text = list(self.ir._corpus.words(fileid))
            for term in ['information', 'recherche', 'module', 'absent']:
                expected = float(text.count(term)) / max(float(text.count(t)) for t in text)
                self.assertAlmostEqual(self.ir.tf(term, fileid=fileid), expected)

    def test_boolean_queries(self):
        self.assertEqual(self.ir.all_match('information'), ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt'])
        self.assertEqual(self.ir.all_match('module and not besoin'), ['D3.txt'])
        self.assertEqual(self.ir.all_match('besoin or (plusieurs and moteur)'), ['D1.txt', 'D2.txt', 'D4.txt'])

    def test_boolean_query_rejects_code(self):
        for query in ['information and', '__import__("os")', 'information + 1', '"x"']:
            with self.assertRaises(Exception):
                self.ir.all_match(query)


class ApiTests(TestCase):
    def select(self, name):
        return self.client.post('/polls/select_corpus/', {'CORPUS_NAME': name})

    def test_nothing_selected(self):
        self.assertEqual(self.client.get('/polls/selected_corpus/').status_code, 404)
        self.assertEqual(self.client.get('/polls/find/', {'term': 'x'}).status_code, 404)

    def test_corporas_lists_the_sample_corpus(self):
        names = [c['name'] for c in self.client.get('/polls/corporas/').json()]
        self.assertIn(SAMPLE_CORPUS, names)

    def test_select_then_query(self):
        self.assertEqual(self.select(SAMPLE_CORPUS).status_code, 202)

        selected = self.client.get('/polls/selected_corpus/').json()
        self.assertEqual(selected['name'], SAMPLE_CORPUS)
        self.assertEqual(selected['files'], ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt'])

        docs = self.client.get('/polls/docs/').json()
        self.assertEqual(len(docs), 4)
        self.assertTrue(docs[0]['sample'].startswith("La recherche d'information"))

        found = self.client.get('/polls/find/', {'term': 'information'}).json()
        self.assertEqual([r['fileid'] for r in found], ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt'])
        self.assertEqual(found[0]['freq'], 3)

        boolean = self.client.get('/polls/boolean_model/', {'query': 'module and not besoin'}).json()
        self.assertEqual(boolean, [{'fileid': 'D3.txt'}])
        bad = self.client.get('/polls/boolean_model/', {'query': 'module and'})
        self.assertEqual(bad.status_code, 400)

        vector = self.client.get('/polls/vector_model/', {'query': 'besoin information'}).json()
        self.assertEqual(len(vector), 4)
        best = max(vector, key=lambda r: r['cosinus_measure'])
        self.assertEqual(best['fileid'], 'D4.txt')

    def test_select_unknown_corpus(self):
        self.assertEqual(self.select('does-not-exist').status_code, 400)
        self.assertEqual(self.select('../polls').status_code, 400)


class UploadTests(TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)
        self._old_root = views.CORPORAS_ROOT
        views.CORPORAS_ROOT = self.root
        self.addCleanup(setattr, views, 'CORPORAS_ROOT', self._old_root)

    def upload(self, corpus, filename, text, description='About cats'):
        return self.client.post('/polls/upload_file', {
            'CORPUS_NAME': corpus,
            'CORPUS_DESCRIPTION': description,
            'file': SimpleUploadedFile(filename, text.encode()),
        })

    def test_upload_creates_a_corpus(self):
        self.assertEqual(self.upload('Cats', 'a.txt', 'cats sleep a lot').status_code, 202)
        self.assertEqual(self.upload('Cats', 'b.txt', 'dogs bark').status_code, 202)
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, 'Cats'))), ['README', 'a.txt', 'b.txt'])

        self.client.post('/polls/select_corpus/', {'CORPUS_NAME': 'Cats'})
        selected = self.client.get('/polls/selected_corpus/').json()
        self.assertEqual(selected['description'], 'About cats')

        # a file added after selection shows up in the next query
        self.upload('Cats', 'c.txt', 'cats again')
        found = self.client.get('/polls/find/', {'term': 'cats'}).json()
        self.assertEqual([r['fileid'] for r in found], ['a.txt', 'c.txt'])

    def test_upload_rejects_paths(self):
        self.assertEqual(self.upload('../escape', 'a.txt', 'x').status_code, 400)
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.root), 'escape')))
