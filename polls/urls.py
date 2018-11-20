from django.urls import path

from . import views

app_name = 'polls'

urlpatterns = [
    # path('', views.IndexView.as_view(), name='index'),
    path('docs/', views.docs_list),
    path('boolean_model/', views.boolean_model),
    path('vector_model/', views.vector_model),
    path('fileids/', views.fileids),
    path('upload_file', views.handle_file),
    path('corporas/', views.corporas),
    path('select_corpus/', views.select_corpus),
    path('selected_corpus/', views.selected_corpus),
    path('find/', views.find),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
]