import cv2

port = int(input('Port? '))
try:
    cam = cv2.VideoCapture(port)
except Exception:
    print(f'No camera found on port {port}!')
    raise SystemExit()

ix = 0
prefix = input('Prefix? ')
clahe = cv2.createCLAHE()
while True:
    ret, img = cam.read()
    if not ret:
        continue

    cv2.imshow('frame', img)
    cv2.waitKey(1)

    fname = input()
    if fname == 'c':
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0,0), fx=0.5,fy=0.5)
        cv2.imwrite(f'{prefix}{ix}.png', clahe.apply(small))
        ix += 1
    elif fname == 'q':
        break
cv2.destroyAllWindows()
del cam
print('Bye!')
