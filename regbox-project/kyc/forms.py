from django.forms import ModelForm, Textarea
from .models import KYC, Aadhar, Pan, Passport


class KYC_Form(ModelForm):
    class Meta:
        model = KYC
        widgets = {
            'parameters': Textarea(attrs={'cols': 10, 'rows': 1}),
        }
        fields = ['file_name', 'card_choice', 'file_uploaded']


class Aadhar_Form(ModelForm):
    class Meta:
        model = Aadhar
        widgets = {
            'dob': Textarea(attrs={'cols': 40, 'rows': 1}),
            'num': Textarea(attrs={'cols': 40, 'rows': 1}),
        }
        fields = ['name', 'gender', 'dob', 'num']


class Pan_Form(ModelForm):
    class Meta:
        model = Pan
        widgets = {
            'pan_name': Textarea(attrs={'cols': 40, 'rows': 1}),
            'pan_fname': Textarea(attrs={'cols': 40, 'rows': 1}),
            'pan_num': Textarea(attrs={'cols': 40, 'rows': 1}),
            'pan_dob': Textarea(attrs={'cols': 40, 'rows': 1}),
        }
        fields = ["pan_name", "pan_fname", "pan_num", "pan_dob"]


class Passport_Form(ModelForm):
    class Meta:
        model = Passport
        widgets = {
            'passport_num': Textarea(attrs={'cols': 40, 'rows': 1}),
            'passport_dob': Textarea(attrs={'cols': 40, 'rows': 1}),
            'passport_expiry_date': Textarea(attrs={'cols': 40, 'rows': 1}),
        }
        fields = ["passport_name", "passport_gender", "passport_num", "passport_dob",
                  "passport_expiry_date"]
