from django.urls import path
from . import views
from .views import CustomLoginView, CustomLogoutView

app_name = 'core'

urlpatterns = [
    path('', views.dataset_list, name='home'),
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.dataset_upload, name='dataset_upload'),
    path('datasets/detail/', views.dataset_detail, name='dataset_detail_default'),
    path('datasets/<int:dataset_id>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<int:dataset_id>/analyze/', views.dataset_analyze, name='dataset_analyze'),
    path('datasets/<int:dataset_id>/analyze-with-grid/', views.analyze_with_grid, name='analyze_with_grid'),
    path('api/datasets/<int:dataset_id>/grid-data/', views.grid_data_api, name='grid_data_api'),
    path('datasets/<int:dataset_id>/generate-profile/', views.generate_profile, name='generate_profile'),
    path('datasets/<int:dataset_id>/profile-status/', views.profile_status, name='profile_status'),
    path('datasets/<int:dataset_id>/delete/', views.dataset_delete, name='dataset_delete'),
    path('datasets/<int:dataset_id>/validate/', views.validate_rules, name='validate_dataset'),
    
    path('rules/', views.rule_list, name='rule_list'),
    path('rules/create/', views.rule_create, name='rule_create'),
    path('rules/create/<int:dataset_id>/', views.rule_create, name='rule_create_with_dataset'),
    path('rules/<int:rule_id>/', views.rule_detail, name='rule_detail'),
    path('rules/<int:rule_id>/delete/', views.rule_delete, name='rule_delete'),
    path('rules/<int:rule_id>/validate/', views.rule_validate, name='rule_validate'),
    
    path('validation-results/', views.validation_results, name='validation_results'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('validation-dashboard/', views.validation_dashboard, name='validation_dashboard'),
    path('validation-dashboard/run-all-rules/', views.trigger_all_rules, name='trigger_all_rules'),
    
    path('documentation/', views.documentation_view, name='documentation'),
    path('login/', CustomLoginView.as_view(), name='login'),
    path('logout/', CustomLogoutView.as_view(), name='logout'),
    path('about/', views.about, name='about'),
    path('api/datasets/<int:dataset_id>/generate-profile/', views.generate_profile_api, name='generate_profile_api'),
    path('api/datasets/<int:dataset_id>/generate-ydata-profile/', views.generate_ydata_profile, name='generate_ydata_profile'),
    path('get-dataset-columns/<int:dataset_id>/', views.get_dataset_columns, name='get_dataset_columns'),
] 

