from django.conf.urls import url, include
from .views import *
from django.views.generic import TemplateView

urlpatterns = [
    url(r'generate_summary',Summarizer.as_view({'post':'generate_summary'}), name="summary_numpy"),
    url(r'summarize_code', Summarizer.as_view({'post': 'summarize_code'}), name="summary_nltk"),
    url(r'display_template',TemplateView.as_view(template_name="display.html"),name="display"),
    url(r'download_dependecy$',Summarizer.as_view({'post':'download_dependencies'}),name="dependecy_download")
]