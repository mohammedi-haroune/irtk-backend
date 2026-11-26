import os
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
import nltk.corpus
import nltk.data
from django.conf import settings
from nltk.corpus.reader import PlaintextCorpusReader
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response

# from polls.serializers import UserSerializer, GroupSerializer
from polls.irreader import IRModels
from polls.models import Document, VectorModelResult
from polls.serializers import DocumentSerializer, VectorModelResultSerializer, FindResultSerializer, CorporaSerializer
from .models import Choice, Question

CORPORAS_ROOT = os.path.join(settings.BASE_DIR, 'corporas')

# nltk 3.9+ only reads corpora under nltk.data.path, so allow the uploads folder
nltk.data.path.append(CORPORAS_ROOT)

# NLTK corpora offered next to the uploaded ones. Each needs its data first:
# python -m nltk.downloader brown
NLTK_CORPORA = ['brown']

# IRModels objects, one per corpus name. The session only stores the name,
# because Django 5 sessions are JSON and can't hold Python objects.
_IR_MODELS = {}


class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.order_by('-pub_date')[:5]


class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'polls/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))


def DocumentResponse(fileids):
    docs = [Document(fileid) for fileid in fileids]
    serializer = DocumentSerializer(docs, many=True)
    return Response(serializer.data)


def is_valid_corpus_name(name):
    return bool(name) and '/' not in name and '\\' not in name and not name.startswith('.')


def load_corpus(corpus_name):
    """Return the corpus reader for an uploaded corpus or an NLTK corpus, or None."""
    if not is_valid_corpus_name(corpus_name):
        return None
    path = os.path.join(CORPORAS_ROOT, corpus_name)
    if os.path.isdir(path):
        return PlaintextCorpusReader(path, r'.*\.txt')
    if corpus_name in NLTK_CORPORA:
        corpus = getattr(nltk.corpus, corpus_name)
        try:
            corpus.ensure_loaded()
        except LookupError:
            return None
        return corpus
    return None


def models(request):
    corpus_name = request.session.get('CORPUS_NAME')
    if corpus_name is None:
        raise NotFound('No corpus selected')
    if corpus_name not in _IR_MODELS:
        corpus = load_corpus(corpus_name)
        if corpus is None:
            raise NotFound('Corpus not found: ' + corpus_name)
        _IR_MODELS[corpus_name] = IRModels(corpus)
    return _IR_MODELS[corpus_name]


def corpus(request):
    return models(request)._corpus


def describe_corpus(name, c):
    o = {'name': name}

    try:
        o['description'] = c.readme()
    except Exception:
        o['description'] = 'No README file found'

    try:
        o['files'] = c.fileids()
    except Exception:
        pass

    try:
        o['categories'] = c.categories()
    except Exception:
        pass

    return o


@api_view(['GET'])
def docs_list(request):
    docs = [{'fileid': fileid, 'sample': corpus(request).raw(fileid)[:200]} for fileid in
            corpus(request).fileids()]
    serializer = DocumentSerializer(docs, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def fileids(request):
    return DocumentResponse(corpus(request).fileids())


@api_view(['GET'])
def boolean_model(request):
    query = request.query_params.get('query')
    ir = models(request)
    try:
        docs = ir.all_match(query)
    except Exception:
        return Response(status=status.HTTP_400_BAD_REQUEST)

    return DocumentResponse(docs)


@api_view(['GET'])
def vector_model(request):
    query = request.query_params.get('query') or ''
    ir = models(request)
    results = [VectorModelResult(fileid,
                                 ir.inner_product(query, fileid),
                                 ir.dice_coef(query, fileid),
                                 ir.cosinus_measure(query, fileid),
                                 ir.jaccard_coef(query, fileid))
               for fileid in ir._corpus.fileids()]

    serializer = VectorModelResultSerializer(results, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def find(request):
    term = request.query_params.get('term')
    if term is None:
        return Response(status=status.HTTP_404_NOT_FOUND)
    serializer = FindResultSerializer(models(request).find(term), many=True)
    return Response(serializer.data)


@api_view(['POST'])
@parser_classes((MultiPartParser, JSONParser))
def handle_file(request):
    corpus_name = request.data.get('CORPUS_NAME', '')
    description = request.data.get('CORPUS_DESCRIPTION', '')
    if not is_valid_corpus_name(corpus_name):
        return Response(status=status.HTTP_400_BAD_REQUEST)

    file = request.data['file']
    corpus_path = os.path.join(CORPORAS_ROOT, corpus_name)
    os.makedirs(corpus_path, exist_ok=True)
    with open(os.path.join(corpus_path, os.path.basename(file.name)), 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    readme_path = os.path.join(corpus_path, 'README')
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as readme:
            readme.write(description)

    # the corpus changed, so its models must be rebuilt on next use
    _IR_MODELS.pop(corpus_name, None)

    return Response(status=status.HTTP_202_ACCEPTED)


@api_view(['POST'])
def select_corpus(request):
    corpus_name = request.data.get('CORPUS_NAME')
    corpus = load_corpus(corpus_name)
    if corpus is None:
        return Response(status=status.HTTP_400_BAD_REQUEST)

    if corpus_name not in _IR_MODELS:
        _IR_MODELS[corpus_name] = IRModels(corpus)
    request.session['CORPUS_NAME'] = corpus_name

    return Response(status=status.HTTP_202_ACCEPTED)


@api_view(['GET'])
def selected_corpus(request):
    c = corpus(request)
    serializer = CorporaSerializer(describe_corpus(request.session['CORPUS_NAME'], c))
    return Response(serializer.data)


@api_view(['GET'])
def corporas(request):
    objects = []

    # NLTK corpora whose data is downloaded
    for name in NLTK_CORPORA:
        c = load_corpus(name)
        if c is not None:
            objects.append(describe_corpus(name, c))

    # user defined corporas
    for name in sorted(os.listdir(CORPORAS_ROOT)):
        c = load_corpus(name)
        if c is not None:
            objects.append(describe_corpus(name, c))

    serializer = CorporaSerializer(objects, many=True)
    return Response(serializer.data)
