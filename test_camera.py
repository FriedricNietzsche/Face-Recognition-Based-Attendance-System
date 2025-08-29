import cv2

print('OpenCV version:', cv2.__version__)

def try_open(index, flag=None):
    if flag is None:
        cap = cv2.VideoCapture(index)
        name = 'default'
    else:
        cap = cv2.VideoCapture(index, flag)
        name = {cv2.CAP_DSHOW: 'CAP_DSHOW', cv2.CAP_MSMF: 'CAP_MSMF'}.get(flag, str(flag))
    ok = cap.isOpened()
    print(f'open {name}:', ok)
    if ok:
        ret, frame = cap.read()
        print(f'  read returned: {ret}', 'frame is None' if frame is None else f'frame shape: {None if frame is None else frame.shape}')
    try:
        cap.release()
    except Exception as e:
        print('release error:', e)

# Try default, DirectShow and MSMF
try_open(0)
try_open(0, cv2.CAP_DSHOW)
try_open(0, cv2.CAP_MSMF)

print('Done')
