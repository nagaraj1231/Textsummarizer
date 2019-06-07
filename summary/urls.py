from django.conf.urls import url, include
from .views import *
from django.views.generic import TemplateView

urlpatterns = [
    url(r'summarize_code',Summarizer.as_view({'post':'summarize_code'}), name="summary"),
    url(r'display_template',TemplateView.as_view(template_name="display.html"),name="display"),
    url(r'download_dependecy$',Summarizer.as_view({'post':'download_dependencies'}),name="dependecy_download")
]