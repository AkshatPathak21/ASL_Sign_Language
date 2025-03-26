import cv2
import os

dir = "Img48x48/"
print(os.getcwd())

if not os.path.exists(dir):
    os.mkdir(dir)
if not os.path.exists(f'{dir}/blank'):
    os.mkdir(f'{dir}/blank')

for i in range(65,91):
    let = chr(i)
    if not os.path.exists(f'{dir}/{let}'):
        os.mkdir(f'{dir}/{let}')

cap = cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    count = {
             'a': len(os.listdir(dir+"/A")),
             'b': len(os.listdir(dir+"/B")),
             'c': len(os.listdir(dir+"/C")),
             'd': len(os.listdir(dir+"/D")),
             'e': len(os.listdir(dir+"/E")),
             'f': len(os.listdir(dir+"/F")),
             'g': len(os.listdir(dir+"/G")),
             'h': len(os.listdir(dir+"/H")),
             'i': len(os.listdir(dir+"/I")),
             'j': len(os.listdir(dir+"/J")),
             'k': len(os.listdir(dir+"/K")),
             'l': len(os.listdir(dir+"/L")),
             'm': len(os.listdir(dir+"/M")),
             'n': len(os.listdir(dir+"/N")),
             'o': len(os.listdir(dir+"/O")),
             'p': len(os.listdir(dir+"/P")),
             'q': len(os.listdir(dir+"/Q")),
             'r': len(os.listdir(dir+"/R")),
             's': len(os.listdir(dir+"/S")),
             't': len(os.listdir(dir+"/T")),
             'u': len(os.listdir(dir+"/U")),
             'v': len(os.listdir(dir+"/V")),
             'w': len(os.listdir(dir+"/W")),
             'x': len(os.listdir(dir+"/X")),
             'y': len(os.listdir(dir+"/Y")),
             'z': len(os.listdir(dir+"/Z")),
             'blank': len(os.listdir(dir+"/blank"))
             }

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame,(0,150),(300,450),(255,255,255),2)
    cv2.imshow("data",frame)
    frame=frame[150:450,0:300]
    cv2.imshow("ROI",frame)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame,(48,48))
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(os.path.join(dir+'A/'+str(count['a']))+'.jpg',frame)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(os.path.join(dir+'B/'+str(count['b']))+'.jpg',frame)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(os.path.join(dir+'C/'+str(count['c']))+'.jpg',frame)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(os.path.join(dir+'D/'+str(count['d']))+'.jpg',frame)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(os.path.join(dir+'E/'+str(count['e']))+'.jpg',frame)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(os.path.join(dir+'F/'+str(count['f']))+'.jpg',frame)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(os.path.join(dir+'G/'+str(count['g']))+'.jpg',frame)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(os.path.join(dir+'H/'+str(count['h']))+'.jpg',frame)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(os.path.join(dir+'I/'+str(count['i']))+'.jpg',frame)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(os.path.join(dir+'J/'+str(count['j']))+'.jpg',frame)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(os.path.join(dir+'K/'+str(count['k']))+'.jpg',frame)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(os.path.join(dir+'L/'+str(count['l']))+'.jpg',frame)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(os.path.join(dir+'M/'+str(count['m']))+'.jpg',frame)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(os.path.join(dir+'N/'+str(count['n']))+'.jpg',frame)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(os.path.join(dir+'O/'+str(count['o']))+'.jpg',frame)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(os.path.join(dir+'P/'+str(count['p']))+'.jpg',frame)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(os.path.join(dir+'Q/'+str(count['q']))+'.jpg',frame)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(os.path.join(dir+'R/'+str(count['r']))+'.jpg',frame)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(os.path.join(dir+'S/'+str(count['s']))+'.jpg',frame)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(os.path.join(dir+'T/'+str(count['t']))+'.jpg',frame)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(os.path.join(dir+'U/'+str(count['u']))+'.jpg',frame)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(os.path.join(dir+'V/'+str(count['v']))+'.jpg',frame)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(os.path.join(dir+'W/'+str(count['w']))+'.jpg',frame)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(os.path.join(dir+'X/'+str(count['x']))+'.jpg',frame)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(os.path.join(dir+'Y/'+str(count['y']))+'.jpg',frame)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(os.path.join(dir+'Z/'+str(count['z']))+'.jpg',frame)
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(os.path.join(dir+'blank/' + str(count['blank']))+ '.jpg',frame)
