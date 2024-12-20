import functions as f
import utils
import matplotlib.image as mpimg
import os

def main():
    imgs = os.listdir('practica02/img')
    
    imgs.sort()
    
    #imgs = imgs[len(imgs)//2:]
    #imgs = imgs[len(imgs)//2:]

    for img in imgs:
        image = mpimg.imread('practica02/img/' + img)
        imgOut = f.detectField(image)
        
        lines = f.findGrassLines(imgOut)
        playersContours = f.findPlayers(imgOut)
        imgOut = utils.drawGrassLines(imgOut, lines)
        imgOut = utils.drawPlayers(imgOut, playersContours)
        name = "practica02/out/" + img 
        utils.show_imgs([image, imgOut])
        mpimg.imsave(name, imgOut)
        

if __name__ == "__main__":
    main()