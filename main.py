import random 
import cv2
import imutils
import questions
from face_action_detection import face_action_detector

cv2.namedWindow('liveness_detection')
cam = cv2.VideoCapture(0)

# parameters
finished_questions = 0
total_questions = 6
failed_try = 0
total_try = 100



def show_image(cam,text,color = (0,0,255)):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    #im = cv2.flip(im, 1)
    cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    return im

fd = face_action_detector()

'''
while(True):
    im = show_image(cam, ' ')
    #cv2.imshow('liveness_detection', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    im = cv2.flip(im, 1)
    fd.det(im)

    im = fd.paint_shape(im)
    cv2.imshow('liveness_detection', im)
'''





for i_questions in range(0,total_questions):
    # genero aleatoriamente pregunta
    index_question = random.randint(0,5)
    question = questions.question_bank(index_question)
    
    im = show_image(cam,question)
    cv2.imshow('liveness_detection',im)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break 

    for i_try in range(total_try):
        ret, im = cam.read()
        im = imutils.resize(im, width=720)
        im = cv2.flip(im, 1)

        fd.det(im)
        challenge_res = questions.challenge_result(question, fd)

        im = show_image(cam,question)
        cv2.imshow('liveness_detection',im)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break 

        if challenge_res == "pass":
            fd.reset()
            im = show_image(cam,question+" : ok")
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break


            finished_questions += 1
            failed_try = 0
            break

        elif challenge_res == "fail":
            failed_try += 1
            show_image(cam,question+" : fail")
        elif i_try == total_try-1:
            break
            

    if finished_questions ==  total_questions:
        while True:
            im = show_image(cam,"LIFENESS SUCCESSFUL",color = (0,255,0))
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
    elif i_try == total_try-1:
        while True:
            im = show_image(cam,"LIFENESS FAIL")
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        break 

    else:
        continue
