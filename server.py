#-*- coding:utf-8 -*-
import json
from collections import OrderedDict
from socket import *
import threading
import random


def init():
    global status,distance,eye,yes,no,answer,rightSight,leftSight
    status=-2
    distance='0'
    eye='noooo'
    yes=0
    no=0
    answer='l'
    rightSight=0.1
    leftSight=0.1

init()
tmp=0
ans=['u','d','l','r']
r=0
HOST = ''
PORT = 10080
ADDR = (HOST, PORT)
# 소켓 생성
serverSocket = socket(AF_INET, SOCK_STREAM)
# 소켓 주소 정보 할당
serverSocket.setsockopt(SOL_SOCKET,SO_REUSEADDR, 1)
serverSocket.bind(ADDR)
print ('server open')
# 연결 수신 대기 상태
serverSocket.listen(1)
# 연결 수락
clientSocket, addr_info = serverSocket.accept()
print(str(addr_info),' is connected\n')

def msg():
    global status,eye,answer,no,leftSight,rightSight,distance
    # status(1):1~5, eye(5):front,noooo,right,leftt,noooo, answer(1):l,r,u,d, no(1):0~3, left(4), right(4), dis(~)
    msgs=''
    le=str(leftSight)
    ri= str(rightSight)
    if len(le)==3:
        le= le + '0'
    if len(ri)==3:
        ri=ri+'0'
    msgs= msgs+ str(status) + eye + answer+str(no)+le +ri+distance
    return msgs
def noteListen(): #노트북에서 데이터 날라올 때
    global tmp,distance,eye
    while True:
        try:
            data=clientSocket.recv(1024)
        except:
            print('user out')
            break
        if not data:
            init()
            break
        if data.decode() =='take' or data.decode() == 'check':
            tmp=1
        if status==0:
            distance=data.decode()
            tmp=1
        elif status==1 or status== 3:
            eye=data.decode()
            tmp=1

       
def cloudListen(sock): #클라우드 함수에서 서버로 접속할 때 
    global tmp,status,distance,eye,leftSight,rightSight,no,yes,r,ans
    data=sock.recv(1024).decode()    
    if data=='start':
        if status==-2:
            print('start')
            clientSocket.send('change')
            sock.send(msg())
            status +=1
        else:
            sock.send(msg())
    elif data=='face':
        if status==-1:
            print('face')
            tmp=0
            clientSocket.send('takeFac')
            while tmp == 0:
                pass
            sock.send(msg())
            status += 1
        else:
            sock.send(msg())
    elif data=='distance':
        if status == 0:
            print('distance')
            tmp=0
            clientSocket.send('takeDis') # 노트북한테 찍으라고 명령
            while tmp==0:
                pass
            sock.send(msg())
            if distance=='2':
                status +=1
            else:
                #send distance pic
                pass
        else:
            sock.send(msg())
    elif data=='eye':
        if status==1: #
            tmp=0
            clientSocket.send('takeEye')
            while tmp==0:
                pass
            print('leftEye')
            #eye='leftt' # 테스트는 left가 맞게ㅔ끔
            sock.send(msg())
            if eye=='leftt':
                status=2
                clientSocket.send(str(r)+'pic'+str(leftSight))
        elif status==3:
            tmp=0
            clientSocket.send('takeEye')
            while tmp==0:
                pass
            print('rightEye')
            #eye='right' #테스트는 right가 맞게끔
            sock.send(msg())
            if eye=='right':
                status=4
                r=random.randrange(0,4)
                clientSocket.send(str(r)+'pic'+str(rightSight))
        else:
            sock.send(msg())
    elif data=='l' or data == 'r' or data=='u' or data=='d':
        if status==2: #왼쪽눈
            if data== ans[r]:
                yes += 1
                ch=0
                if yes ==3:
                    if leftSight==1.5: # 시력 1.5 찍었을 시
                        no=3
                        sock.send(msg())
                        status +=1
                        ch=1
                        clientSocket.send("next")
                    else:
                        if leftSight == 0.1:
                            leftSight += 0.1
                        elif leftSight <= 1.0:
                            leftSight += 0.2
                        elif leftSight == 1.2:
                            leftSight += 0.3
                        no=0
                        yes=0
                        sock.send(msg())
                else:
                    sock.send(msg())
                if ch==0: # 시력 2.0 이 아닐 시 사진 보여주기냄
                    r=random.randrange(0,4)
                    clientSocket.send(str(r)+'pic'+str(leftSight))
                print ("status :",status,"yes",yes,"no",no)
            else:
                no += 1
                if no==3:
                    if leftSight == 0.2:
                        leftSight -= 0.1
                    elif leftSight > 0.2 and leftSight <= 1.2:
                        leftSight -= 0.2
                    elif leftSight == 1.5:
                        leftSight -= 0.3
                    yes=0
                    sock.send(msg())
                    no=0
                    clientSocket.send("next")
                    status +=1
                else:
                    sock.send(msg())
                    r=random.randrange(0,4)
                    clientSocket.send(str(r)+'pic'+str(leftSight))
                print("status: ",status,"yes",yes,"no",no)
        elif status==4: #오른쪽눈
            if data == ans[r]:
                yes +=1
                ch=0
                if yes == 3:
                    if rightSight == 2: # 최대 시력은 2.0 이므로
                        ch=1
                        no=3
                        status +=1
                        clientSocket.send('result'+str(leftSight)+'/'+str(rightSight))
                    else:
                        if rightSight == 0.1:
                            rightSight += 0.1
                        elif rightSight <= 1.0:
                            rightSight += 0.2
                        elif rightSight == 1.2:
                            rightSight += 0.3
                        no=0
                        yes=0
                sock.send(msg())
                if ch==0:
                    r=random.randrange(0,4)
                    clientSocket.send(str(r)+'pic'+str(rightSight))
                print("yes",yes,"no",no)
            else:
                no+=1
                if no==3:
                    if rightSight == 0.2:
                        rightSight -= 0.1
                    elif rightSight > 0.2 and rightSight <= 1.2:
                        rightSight -= 0.2
                    elif rightSight == 1.5:
                        rightSight -= 0.3
                    sock.send(msg())
                    clientSocket.send('result'+str(leftSight)+'/'+str(rightSight))
                    # send result
                    print('result: '+ str(leftSight)+' / '+str(rightSight))
                    init()
                else:
                    sock.send(msg())
                    r=random.randrange(0,4)
                    clientSocket.send(str(r)+'pic'+str(rightSight))
                    print('yes',yes,'no',no)
                    #send picture
    elif data=='exit':
        init()
    elif data=='take': # 테스트용으로 만든것이니 무시
        tmp=0
        clientSocket.send('take'.encode())
        while tmp==0:
            pass
        print ("send cloud")
        sock.send(str(4))
    sock.close()

notet=threading.Thread(target=noteListen)
notet.start()
serverSocket.listen(5)
while True:
    client, add = serverSocket.accept() 
    cloudt= threading.Thread(target=cloudListen,args=(client,))
    cloudt.start()
# 소켓 종료 
clientSocket.close()
serverSocket.close()
