from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.contrib.auth import login, authenticate, logout
from .forms import KYC_Form, Aadhar_Form, Pan_Form, Passport_Form
from .models import KYC, Aadhar, Pan, Passport
import pytesseract
import cv2
import re
from django.contrib.auth.decorators import login_required
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from numpy import asarray, array, expand_dims, vstack
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from regbox.settings import BASE_DIR
from matplotlib import pyplot as plt


#########################################################
# OTHER REQUIRED FUNCTIONALITIES

def extract_aadhar_details(img):
    img1 = cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)

    t1 = pytesseract.image_to_string(image=img1).strip()

    t2 = pytesseract.image_to_string(image=img).strip()

    # print("\n t1 = {} , t2 = {}".format(len(t1), len(t2)))
    if len(t1) < len(t2):
        # print(" \n t2 selected \n")
        text = t2
    else:
        # print(" \n t2 selected \n")
        text = t1

    keyList = ["Name", "Gender", "Aadhar Number", 'DOB']

    output = {}

    for i in keyList:
        output[i] = None

    # text = pytesseract.image_to_string(image=img).strip()

    text = text.replace("\n\n", " ")

    # print("-------------------------------------------------")
    # print("text = \n", text)
    # print("-------------------------------------------------")

    try:
        if "Goverment of India" in text:
            print("\n VALID ID VERIFIED\n ")
            text = text.replace('Goverment of India', '')
    except:
        pass

    aadhar_pattern = re.compile("\d{4}\s\d{4}\s\d{4}")
    aadhar_num = aadhar_pattern.findall(text)

    if aadhar_num:
        x = str(aadhar_num[0])
        x = x.replace(" ", '')
        output['Aadhar Number'] = int(x)
        pass

    if "Male" or "MALE" in text:
        output['Gender'] = "Male"
        text = text.split("Male")[0]
    elif "Female" or "FEMALE" in text:
        output['Gender'] = "Female"
        text = text.split("Female")[0]
    else:
        output['Gender'] = None

    # year_dob_pattern = re.compile("/\s\w+")  # pattern for year or dob matches[0][2:] results year or DOB
    # ye = year_dob_pattern.findall(text)

    if len(aadhar_num) > 0:
        text = text.split(aadhar_num[0])[0]
        pass

    if "Year of Birth" in text:
        year_pattern = re.compile(":\s\d{4}\s+")
        year = year_pattern.findall(text)
        if len(year) > 0:
            output['DOB'] = year[0][2:6]
            text = text.split(output['DOB'])[0]

    elif 'DOB' in text:
        dob_pattern = re.compile("\d{2}/\d{2}/\d{4}")
        dob = dob_pattern.findall(text)
        if len(dob) > 0:
            output['DOB'] = dob[0]
            text = text.split(output['DOB'])[0]

    else:
        output['DOB'] = None

    name_pattern = re.compile("[A-Z][a-z]+\s[A-Z][a-z]+")
    name = name_pattern.findall(text)
    if len(name) >= 1:
        output['Name'] = name[-1]

    print("\n\n\n aadhar output", output)
    return output


def extract_pan_details(img):
    img = cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)
    img = cv2.medianBlur(img, 1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    keyList = ["Name", "Father's Name", "Pan Number", 'DOB']
    output = {}
    for i in keyList:
        output[i] = None

    text = pytesseract.image_to_string(img)

    text = text.replace("\n", " ")
    text = text.replace("  ", " ")

    a = ["INCOME", "TAX", "DEPARTMENT", "GOVT.", "OF", "INDIA",
         'Permanent', 'Account', 'Number', 'Card']

    c = 0
    for i in a:
        if i in text:
            c += 1

    for i in a:
        if i in text:
            text = text.replace(i, "")

    # print("------------------------------------")
    # print("text = \n", text)
    # print("------------------------------------")

    pancard_num_pattern = re.compile("[A-Z]{5}[0-9]{4}[A-Z]")
    pancard_num = pancard_num_pattern.findall(text)

    if len(pancard_num) > 0:
        if pancard_num[0]:
            output['Pan Number'] = pancard_num[0]

    dob_pattern = re.compile("\d\d/\d\d/\d\d\d\d")
    dob = dob_pattern.findall(text)

    if len(dob) > 0:
        if dob[0]:
            output['DOB'] = dob[0]
            text = text.replace(output['DOB'], '')

    if "Name" in text:
        name_pattern = re.compile("Name\s[A-Z]+\s[A-Z]+")
        name = name_pattern.findall(text)
        if len(name) > 0:
            try:
                # if name[0][4:]:
                output['Name'] = name[0][4:].strip()
            except:
                pass
            # if name[1][4:]:
            try:
                output["Father's Name"] = name[1][4:].strip()
            except:
                pass
    else:
        name_pattern = re.compile("[A-Z]+\s[A-Z]+")
        name = name_pattern.findall(text)
        if len(name) > 0:
            try:
                # if name[0]:
                output['Name'] = name[0]
            except:
                pass
            # if name[1]:
            try:
                output["Father's Name"] = name[1]
            except:
                pass

    none_count = 0
    for i in output:
        if output[i] == None:
            none_count += 1
    print("------------------------------------")
    print("\n none_count = ", none_count)
    print("\n pan output = ", output)
    return output, none_count


def extract_pan_details2(img):
    img = cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)
    img = cv2.medianBlur(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    keyList = ["Name", "Father's Name", "Pan Number", 'DOB']
    output = {}
    for i in keyList:
        output[i] = None

    text = pytesseract.image_to_string(gray)

    text = text.replace("\n", " ")
    text = text.replace("  ", " ")

    a = ["INCOME", "TAX", "DEPARTMENT", "GOVT.", "OF", "INDIA",
         'Permanent', 'Account', 'Number', 'Card']

    c = 0
    for i in a:
        if i in text:
            c += 1

    for i in a:
        if i in text:
            text = text.replace(i, "")

    # print("------------------------------------")
    # print("text = \n", text)
    # print("------------------------------------")

    pancard_num_pattern = re.compile("[A-Z]{5}[0-9]{4}[A-Z]")
    pancard_num = pancard_num_pattern.findall(text)

    if len(pancard_num) > 0:
        if pancard_num[0]:
            output['Pan Number'] = pancard_num[0]

    dob_pattern = re.compile("\d\d/\d\d/\d\d\d\d")
    dob = dob_pattern.findall(text)

    if len(dob) > 0:
        if dob[0]:
            output['DOB'] = dob[0]
            text = text.replace(output['DOB'], '')

    if "Name" in text:
        name_pattern = re.compile("Name\s[A-Z]+\s[A-Z]+")
        name = name_pattern.findall(text)
        if len(name) > 0:
            try:
                # if name[0][4:]:
                output['Name'] = name[0][4:].strip()
            except:
                pass
            # if name[1][4:]:
            try:
                output["Father's Name"] = name[1][4:].strip()
            except:
                pass
    else:
        name_pattern = re.compile("[A-Z]+\s[A-Z]+")
        name = name_pattern.findall(text)
        if len(name) > 0:
            try:
                # if name[0]:
                output['Name'] = name[0]
            except:
                pass
            # if name[1]:
            try:
                output["Father's Name"] = name[1]
            except:
                pass
    none_count = 0
    for i in output:
        if output[i] == None:
            none_count += 1

    print("------------------------------------")
    print("\n none_count = ", none_count)
    print("\n pan output = ", output)
    return output, none_count


def extract_passport_details(img):
    img = cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)

    keyList = ["Name", "Gender", "Passport Number", 'DOB', 'Expiry Date']

    output = {}

    for i in keyList:
        output[i] = None

    text = pytesseract.image_to_string(image=img).strip()

    mrz1_pattern = re.compile("[A-Z]<.+<<.+<")
    mrz1 = mrz1_pattern.findall(text)

    mrz2_pattern = re.compile(".\d+.<\w+<")
    mrz2 = mrz2_pattern.findall(text)
    t1 = ""
    t2 = ""
    if len(mrz1) > 0:
        t1 = mrz1[0]
        t1 = t1.replace(" ", "")
        # print("\n t1 = ", t1)

    if len(mrz2) > 0:
        t2 = mrz2[0]
        t2 = t2.replace(" ", "")
        # print("\n t1 = ", t1)
    #######################################
    # print('-----------------------------------------------------')
    # print("\n t1 = ", t1)
    # print('-----------------------------------------------------')
    # print("\n t2 = ", t2)
    ########################################

    try:
        t1 = t1[2:]
        # t1 = t1.split("<<")
        # a, b = t1.split("<<")

        a = list(mrz1[0][:-1].split("<"))
        a = [i.strip() for i in a]
        # print("a ", a)
        for j in reversed(t1):
            if ord(j) == 60:
                t1 = t1.replace(j, " ")
            else:
                break

        t1 = t1.strip()
        b = list(t1.split(" "))
        # print("\n t1 = ", t1)
        # print("\n b = ", b)

        names = [i for i in b if len(i) > 0]
        # print("\n name =", names)

        lastname = names[0][3:]  # lastname
        firstname = ""
        for i in range(1, len(names)):
            firstname += names[i] + ' '

        # print('first name = ', firstname)

        output['Name'] = firstname + lastname
    except:
        pass

    try:
        t2 = list(mrz2[0][:-1].split("<"))
        output['Passport Number'] = t2[0]
        z1 = t2[1][4:]
        date = z1[:6]
        output['DOB'] = date[4:] + '/' + date[2:4] + '/' + date[:2]
        output['Gender'] = z1[7:8]
        expdate = z1[8:14]
        output['Expiry Date'] = expdate[4:] + '/' + expdate[2:4] + '/' + expdate[:2]
    except:
        pass

    # return output

    print('-----------------------------------------------------')
    print("\n passport output = \n", output)
    return output


def extract_face_from_image(image, required_size=(224, 224)):
    # load image and detect faces
    # image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    face_images = []

    for face in faces:
        # extract the bounding box from the requested face
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face_boundary = image[y1:y2, x1:x2]

        # resize pixels to the model size
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images


def highlight_faces(image, faces):
    # display image
    #   image = plt.imread(image_path)
    plt.imshow(image)

    ax = plt.gca()

    # for each face, draw a rectangle based on coordinates
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                                fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()


def get_model_scores(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    # samples = expand_dims(samples, axis=0)

    samples = preprocess_input(samples, version=2)
    # samples = samples.flatten()
    samples = vstack(samples)

    # samples = samples.reshape((224, 224, 3))

    print("sample size", samples.shape)

    # create a vggface model object
    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')

    # perform prediction
    # print("samples", samples)
    # print(samples.shape)
    return model.predict(samples)


def capture_cam_image():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 255)
    cap.set(4, 255)
    get_img = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('frame', frame2)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                return frame2

    pass


##########################################################
# Create your views here.
def home(request):
    return render(request, 'kyc/home.html')


def signup(request):
    if request.method == "GET":
        return render(request, 'kyc/signup.html',
                      {'form': UserCreationForm})
    else:
        if request.POST['password1'] == request.POST['password2']:
            try:
                user = User.objects.create_user(
                    request.POST['username'],
                    password=request.POST['password2'])
                user.save()
                login(request, user)
                return redirect('home')

            except IntegrityError:
                return render(request, 'kyc/signup.html',
                              {'form': UserCreationForm,
                               'error': "username is already taken, pick new username"})


def loginuser(request):
    if request.method == 'GET':
        return render(request, 'kyc/loginuser.html',
                      {'form': AuthenticationForm}
                      )
    else:
        user = authenticate(request, username=request.POST['username'],
                            password=request.POST['password'])
        if user is None:
            return render(request, 'kyc/loginuser.html',
                          {'error': 'username and password did not match'}
                          )
        else:
            login(request, user)
            return redirect('home')


def logoutuser(request):
    # if request.method == 'POST':
    try:
        logout(request)
    except:
        pass
    return redirect('home')


@login_required
def uploadpage(request):
    if request.method == 'GET':
        return render(request, 'kyc/upload.html',
                      {'form': KYC_Form()})
    else:
        # try:
        form = KYC_Form(request.POST, request.FILES)
        new_form = form.save(commit=False)
        new_form.user = request.user
        # x = plt.imread(KYC_Form['file_uploaded'])
        new_form.save()
        im = plt.imread(request.FILES['file_uploaded'])
        # print("\n\n\n file type = ", request.POST['card_choice'])
        im_details = dict()
        if request.POST['card_choice'] == 'aadhar card':
            im_details = extract_aadhar_details(im)
            aadhar_inst = Aadhar.objects.create(name=im_details['Name'],
                                                gender=im_details['Gender'],
                                                dob=im_details['DOB'],
                                                num=im_details['Aadhar Number'],
                                                kyc_doc=new_form
                                                )
            pass
        elif request.POST['card_choice'] == "pan card":
            im_details1, n1 = extract_pan_details(im)
            im_details2, n2 = extract_pan_details2(im)
            im_details = im_details1
            for i in im_details:
                if im_details[i] == None and im_details2[i] != None:
                    im_details[i] = im_details2[i]

                else:
                    im_details[i] = im_details1[i]

            # if n1 > n2:
            #     im_details = im_details2
            # else:
            #     im_details = im_details1

            #     keyList = ["Name", "Father's Name", "Pan Number", 'DOB']
            Pan.objects.create(pan_name=im_details['Name'],
                               pan_fname=im_details["Father's Name"],
                               pan_num=im_details['Pan Number'],
                               pan_dob=im_details['DOB'],
                               pan_doc=new_form
                               )
        else:
            im_details = extract_passport_details(im)
            #    keyList = ["Name", "Gender", "Passport Number", 'DOB', 'Expiry Date']

            Passport.objects.create(
                passport_name=im_details['Name'],
                passport_gender=im_details['Gender'],
                passport_num=im_details["Passport Number"],
                passport_dob=im_details["DOB"],
                passport_expiry_date=im_details["Expiry Date"],
                passport_doc=new_form
            )

        return redirect('home')


@login_required
def available_docs(request):
    docs = KYC.objects.filter(user=request.user)
    return render(request, 'kyc/available_docs.html',
                  {'docs': docs}
                  )


@login_required
def view_doc(request, doc_pk):
    doc = get_object_or_404(KYC, pk=doc_pk, user=request.user)
    form = KYC_Form(instance=KYC)
    if doc.card_choice == 'aadhar card':
        d2 = get_object_or_404(Aadhar, kyc_doc=doc.id)
        f2 = Aadhar_Form(instance=d2)
    elif doc.card_choice == "pan card":
        d2 = get_object_or_404(Pan, pan_doc=doc.id)
        f2 = Pan_Form(instance=d2)
    else:
        d2 = get_object_or_404(Passport, passport_doc=doc.id)
        f2 = Passport_Form(instance=d2)
    if request.method == 'GET':
        form = KYC_Form(instance=doc)
        return render(request, 'kyc/view_doc.html',
                      {'form': form,
                       'doc': doc,
                       'f2': f2
                       })
    else:
        form = KYC_Form(request.POST, request.FILES, instance=doc)
        form.save()
        return redirect("available_docs")


@login_required
def face_compare(request):
    if request.method == 'GET':
        docs = KYC.objects.filter(user=request.user)
        return render(request, "kyc/face_compare.html", {'docs': docs})
    else:
        img1 = capture_cam_image()
        # detector = MTCNN()
        # f1 = detector.detect_faces(img1)
        # highlight_faces(img1, f1)

        file_name_received = request.POST.get('doc_name')
        print("\n\n\n  got value from POST = ", str(file_name_received))  # my passport

        img2 = KYC.objects.filter(file_name=file_name_received).first().file_uploaded.url
        # print("\n\n\n\n\n\n img2 PRINT CHECK = ", img2)
        img_dir = str(BASE_DIR) + img2
        print("\n\n\n\n img_dir ", img_dir)
        img_dir = img_dir.replace("\\", "/")
        # img2 = Image.open(str(BASE_DIR) + '/' + img2)
        img2 = plt.imread(img_dir)

        print("\n\n\n\n\n\n img2 PRINT CHECK = ", img2)
        print("\n\n\n\n\n\n img2 DIR PRINT CHECK = ", img_dir)

        try:
            faces = [extract_face_from_image(image_path)
                     for image_path in [img1, img2]]

            model_scores = get_model_scores(faces)
            if cosine(model_scores[0], model_scores[1]) <= 0.45:
                print("\n\n\n\n Faces Matched \n\n")
                KYC.objects.filter(file_name=file_name_received).update(face_matched=True)
            else:
                print("\n\n\n\n Faces NOT Matched \n\n")
                pass

        except:
            print("SOME ERROR TRY DIFFERENT IMAGE")

        return redirect('home')


def delete_doc(request, doc_pk):
    doc = get_object_or_404(KYC, pk=doc_pk, user=request.user)
    if request.method == 'POST':
        doc.delete()
        return redirect('home')


def regbox(request):
    return render(request, 'kyc/regbox.html')


def charts(request):
    return render(request, 'kyc/charts.html')


def userhome(request):
    return render(request, 'kyc/userhome.html')


@login_required
def index(request):
    docs = KYC.objects.filter(user=request.user)
    files_uploaded = KYC.objects.count()
    no_of_aadhar = KYC.objects.filter(card_choice='aadhar card').count()
    no_of_pan = KYC.objects.filter(card_choice='pan card').count()
    no_of_passport = KYC.objects.filter(card_choice='passport card').count()
    return render(request, 'kyc/index.html', {'docs': docs,
                                              'files_uploaded': files_uploaded,
                                              'no_of_aadhar': no_of_aadhar,
                                              'no_of_pan': no_of_pan,
                                              'no_of_passport': no_of_passport,
                                              })

# data_path = os.path._getfullpathname(doc.file_uploaded.name)
# print("\n\n\n\n\n\n\n", doc.file_uploaded)
# im = Image.open(doc.file_uploaded)
