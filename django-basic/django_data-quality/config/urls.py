from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Admin and authentication URLs
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    
    # Core application URLs
    path('', include(('core.urls', 'core'), namespace='core')),
    
    # Health check URLs
    path('health/', include('health_check.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Add static and media URLs for development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 