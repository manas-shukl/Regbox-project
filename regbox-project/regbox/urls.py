"""regbox URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from kyc import views
from django.contrib.auth.models import User
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('', views.home, name='home'),
                  path('signup/', views.signup, name='signup'),
                  path('login/', views.loginuser, name='loginuser'),
                  path("logout/", views.logoutuser, name='logoutuser'),
                  path("upload/", views.uploadpage, name='upload'),
                  path("docs/", views.available_docs, name='available_docs'),
                  path("face_compare/", views.face_compare, name='face_compare'),
                  path("docs/<int:doc_pk>", views.view_doc, name='view_doc'),
                  path("todo/<int:doc_pk>/delete", views.delete_doc, name='delete_doc'),
                  path("regbox/", views.regbox, name='regbox'),
                  path('charts/', views.charts, name='charts'),
                  path("userhome/", views.userhome, name='userhome'),
                  path('dashboard/', views.index, name='index'),

              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
