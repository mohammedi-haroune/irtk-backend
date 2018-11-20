import os
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from nltk.corpus import *
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response

# from polls.serializers import UserSerializer, GroupSerializer
from polls.irreader import IRModels
from polls.models import Document, VectorModelResult
from polls.serializers import DocumentSerializer, VectorModelResultSerializer, FindResultSerializer, CorporaSerializer
from .models import Choice, Question

CORPORAS_ROOT = 'corporas/'


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


def corpus(request):
    return request.session['IR_MODELS']._corpus


def models(request):
    return request.session['IR_MODELS']


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
    try:
        docs = models(request).all_match(query)
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)

    return DocumentResponse(docs)


@api_view(['GET'])
def vector_model(request):
    query = request.query_params.get('query')
    results = [VectorModelResult(fileid,
                                 models(request).inner_product(query, fileid),
                                 models(request).dice_coef(query, fileid),
                                 models(request).cosinus_measure(query, fileid),
                                 models(request).jaccard_coef(query, fileid))
               for fileid in corpus(request).fileids()]

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
    corpus_name = request.POST['CORPUS_NAME']
    description = request.POST['CORPUS_DESCRIPTION']

    file = request.data['file']
    corpus_path = CORPORAS_ROOT + corpus_name + '/'
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)
    with open(corpus_path + file.name, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    readme_path = corpus_path + 'README'
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as readme:
            readme.write(description)

    return Response(status=status.HTTP_202_ACCEPTED)


@api_view(['POST'])
def select_corpus(request):
    corpus_name = request.POST['CORPUS_NAME']
    request.session['CORPUS_NAME'] = corpus_name

    if os.path.exists(CORPORAS_ROOT + corpus_name):
        corpus = PlaintextCorpusReader(CORPORAS_ROOT + corpus_name, '.*\.txt')
    else:
        try:
            corpus = eval('nltk.corpus.' + corpus_name)
        except:
            return Response(status=status.HTTP_400_BAD_REQUEST)

    ir = IRModels(corpus)
    request.session['IR_MODELS'] = ir

    return Response(status=status.HTTP_202_ACCEPTED)

@api_view(['GET'])
def selected_corpus(request):
    print(request.session.keys())
    print(request.session.session_key)

    if request.session.session_key is None:
        return Response(status=status.HTTP_404_NOT_FOUND)

    c = corpus(request)

    o = {'name': request.session['CORPUS_NAME']}

    try:
        o['description'] = c.readme()
    except:
        o['description'] = 'No README file found'

    try:
        o['files'] = c.fileids()
    except:
        pass

    try:
        o['categories'] = c.categories()
    except:
        pass

    serializer = CorporaSerializer(o)
    return Response(serializer.data)


@api_view(['GET'])
def corporas(request):
    # get nltk corporas
    # names = os.listdir(nltk.data.find("corpora"))
    names = ['brown']
    objects = []
    for name in names:
        try:
            c = eval(name)
        except:
            continue

        o = {'name': name}

        try:
            o['description'] = c.readme()
        except:
            o['description'] = 'No README file found'

        # try:
        #    o['words'] = len(c.words())
        # except:
        #    pass

        try:
            o['files'] = c.fileids()
        except:
            pass

        try:
            o['categories'] = c.categories()
        except:
            pass

        objects.append(o)

    # get user defined corporas
    defined_names = os.listdir(CORPORAS_ROOT)

    for name in defined_names:
        try:
            c = PlaintextCorpusReader(CORPORAS_ROOT + name, '.*\.txt')
        except:
            continue

        o = {'name': name}

        try:
            o['description'] = c.readme()
        except:
            o['description'] = 'No README file found'

        # try:
        #    o['words'] = len(c.words())
        # except:
        #    pass

        try:
            o['files'] = c.fileids()
        except:
            pass

        try:
            o['categories'] = c.categories()
        except:
            pass

        objects.append(o)

    serializer = CorporaSerializer(objects, many=True)
    return Response(serializer.data)
