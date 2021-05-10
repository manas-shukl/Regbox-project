from django.contrib import admin
from django.forms import Textarea
from .models import Aadhar

# Register your models here.

# class AadharAdmin(admin.ModelAdmin):
#     formfield_overrides = {
#         Aadhar.TextField: {'widget': Textarea(
#             attrs={'rows': 1,
#                    'cols': 40,
#                    'style': 'height: 1em;'})},
#     }
