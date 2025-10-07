from django.urls import path
from . import views

urlpatterns = [
    #path('search/', views.search_companies_api, name='search_companies_api'),
    path('search-by-linkedin/', views.search_by_linkedin_api, name='search_by_linkedin_api'),
    path('stats/', views.api_usage_stats, name='api_usage_stats'),
]