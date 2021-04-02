import time

lastTime=0
def waitForNextFrame(frameTimeSeconds:float):
    global lastTime
    currTime=time.time()
    if currTime<lastTime+frameTimeSeconds:
        time.sleep(lastTime+frameTimeSeconds-currTime)
    lastTime=time.time()

