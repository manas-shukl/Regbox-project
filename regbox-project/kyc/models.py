from django.db import models
from django.contrib.auth.models import User

CHOICES = (
    ('aadhar card', 'aadhar card'),
    ('pan card', 'pan card'),
    ('passport card', 'passport card'),
)


# Create your models here.
class KYC(models.Model):
    file_name = models.CharField(max_length=100)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    card_choice = models.CharField(max_length=20, choices=CHOICES, default='aadhar card')
    file_uploaded = models.ImageField()
    face_matched = models.BooleanField(default=None, null=True)


class Aadhar(models.Model):
    name = models.CharField(max_length=100, default=None)
    gender = models.CharField(max_length=100, default=None)
    dob = models.TextField(default=None, null=True)
    num = models.TextField(default=None, null=True)
    kyc_doc = models.ForeignKey(KYC, on_delete=models.CASCADE)


#     keyList = ["Name", "Father's Name", "Pan Number", 'DOB']
class Pan(models.Model):
    pan_name = models.TextField(default=None, null=True)
    pan_fname = models.TextField(default=None, null=True)
    pan_num = models.TextField(default=None, null=True)
    pan_dob = models.TextField(default=None, null=True)
    pan_doc = models.ForeignKey(KYC, on_delete=models.CASCADE)


#     keyList = ["Name", "Gender", "Passport Number", 'DOB', 'Expiry Date']
class Passport(models.Model):
    passport_name = models.CharField(max_length=100, default=None)
    passport_gender = models.CharField(max_length=100, default=None)
    passport_num = models.TextField(default=None, null=True)
    passport_dob = models.TextField(default=None, null=True)
    passport_expiry_date = models.TextField(default=None, null=True)
    passport_doc = models.ForeignKey(KYC, on_delete=models.CASCADE)

