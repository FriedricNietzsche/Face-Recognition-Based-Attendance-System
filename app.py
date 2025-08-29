import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)

nimgs = 50

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def open_camera(index=0, warmup_frames=5, warmup_delay=0.1):
    """Try multiple backends to open the camera and perform a short warm-up.

    Returns an opened cv2.VideoCapture or None on failure.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    for backend in backends:
        try:
            if backend is None:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, backend)
        except Exception:
            continue

        if not (cap is not None and cap.isOpened()):
            try:
                cap.release()
            except Exception:
                pass
            continue

        # warm-up reads
        ok = False
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if ret and frame is not None and getattr(frame, 'size', 0) > 0:
                ok = True
                break
            time.sleep(warmup_delay)

        if ok:
            return cap

        try:
            cap.release()
        except Exception:
            pass

    return None


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img_path = os.path.join('static', 'faces', user, imgname)
            img = cv2.imread(img_path)
            # skip unreadable images
            if img is None:
                continue
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    # treat an attendance entry as unique by (Name, Roll) pair
    # normalize types for comparison
    df_rolls = df['Roll'].astype(str).tolist()
    df_names = df['Name'].astype(str).tolist()
    exists = False
    for n, r in zip(df_names, df_rolls):
        if n == username and r == str(userid):
            exists = True
            break

    if not exists:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)




################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    if not duser:
        userlist, names, rolls, l = getallusers()
        return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='No user specified to delete.')

    user_path = os.path.join('static', 'faces', duser)
    if os.path.isdir(user_path):
        deletefolder(user_path)
    else:
        userlist, names, rolls, l = getallusers()
        return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Specified user does not exist.')

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = open_camera(0)
    if cap is None:
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Unable to access the webcam. Make sure it is connected and not used by another application.')

    fail_count = 0
    MAX_FAILS = 30
    while True:
        ret, frame = cap.read()
        # handle failed reads
        if not ret or frame is None or frame.size == 0:
            fail_count += 1
            if fail_count >= MAX_FAILS:
                cap.release()
                cv2.destroyAllWindows()
                return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Failed to read frames from webcam. Aborting.')
            # small wait to allow camera to warm up
            key = cv2.waitKey(50)
            if key in (27, ord('q'), ord('Q')):
                break
            continue

        fail_count = 0
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # only show window if frame is valid
        # show instructive overlay and display frame
        try:
            if frame is not None and frame.size > 0:
                cv2.putText(frame, 'Press ESC or Q to stop', (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow('Attendance', frame)
        except cv2.error:
            # if imshow is not supported in the environment (headless), skip showing
            pass

        key = cv2.waitKey(1)
        if key in (27, ord('q'), ord('Q')):
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = open_camera(0)
    if cap is None:
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Unable to access the webcam. Make sure it is connected and not used by another application.')

    fail_count = 0
    MAX_FAILS = 60
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            fail_count += 1
            if fail_count >= MAX_FAILS:
                cap.release()
                cv2.destroyAllWindows()
                names, rolls, times, l = extract_attendance()
                return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='Failed to read frames from webcam while adding user. Aborting.')
                key = cv2.waitKey(50)
                if key in (27, ord('q'), ord('Q')):
                    break
            continue

        fail_count = 0
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break

        # show overlay and display frame
        try:
            if frame is not None and frame.size > 0:
                cv2.putText(frame, 'Press ESC or Q to stop', (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow('Adding new User', frame)
        except cv2.error:
            # headless environment: skip showing frame
            pass

        key = cv2.waitKey(1)
        if key in (27, ord('q'), ord('Q')):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
