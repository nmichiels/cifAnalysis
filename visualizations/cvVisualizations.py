import numpy as np

import cv2

from cifDataset.cifStreamer.dataPreparation import pad_or_crop, center_crop_pad
# from visualizations.filtering import hasOneBlob

def visualizeMask(maskImage, targetSize):
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    print("Visualizeing mask: ", maskImage.shape)
    maskStack = np.empty([maskImage.shape[0],0]) # maskStack = np.empty([targetSize,0])
    for channel in range(maskImage.shape[-1]):#range(0,1):#

        mask = maskImage[:,:,channel]

        maxMask = np.amax(mask)
        if (maxMask != 0):
            mask = mask / maxMask
        maskStack = np.hstack((maskStack, mask))
    cv2.imshow('Mask',maskStack)
    chr = cv2.waitKey(0)


def visualizeImage(image):
    img = cv2.imread(image, -1)

    if img is None:
        print("Could not open",image)
        return

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    multFactor = 1000000

    def rescaleImage(im, minRange, maxRange):
        return (im - minRange) / (maxRange - minRange)

    def changeRange(x):
        maxRange=cv2.getTrackbarPos("Max", 'Image') / float(multFactor)
        minRange=cv2.getTrackbarPos("Min", 'Image') / float(multFactor)
        imgScaled = rescaleImage(img, minRange, maxRange)
        cv2.imshow('Image',imgScaled)

    cv2.createTrackbar("Max", "Image",int(1*multFactor),int(1*multFactor),changeRange)
    cv2.createTrackbar("Min", "Image",int(0*multFactor),int(1*multFactor),changeRange)

    cv2.imshow('Image',img)

    changeRange(0)
    chr = cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def visualizeDatasetGamma(dataset, targetSize, oneBlobsOnly = False, filterChannel = 0):

    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    
        # apply gamma correction using the lookup table
        image = (image*255).astype("uint8")
        return cv2.LUT(image, table)


    # dataset.reset()

    imageCounter = 0
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    multFactor = 1000000
    gamma = 1.0


    def rescaleImage(im, minRange, maxRange):
        return (im - minRange) / (maxRange - minRange)
    
    numCells = 10

    while (not dataset.eod()):

        vimgStack = np.empty([0,dataset.num_channels*targetSize])

        for cell in range(numCells):
            image = dataset.nextImage()

            if isinstance(image, tuple):
                image = image[0]
            if (len(image.shape) == 4):
                image = image[0,:,:,:]
        
            print ("Image " + repr(imageCounter), image.shape)
            image = pad_or_crop(image, targetSize, 'symmetric')
    
            

            print ("Image " + repr(imageCounter), image.shape)

            imgStack = np.empty([targetSize,0])
    

            for channel in range(image.shape[-1]):#range(0,1):#
                img = image[:,:,channel]
                
                # img -= np.min(img)
                # img /= np.amax(img)
                # print(np.min(img), np.max(img))
            

                imgStack = np.hstack((imgStack, img))
                
                # cv2.putText(imgStack,"BF", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255)
            
            vimgStack = np.vstack((vimgStack, imgStack))
    
        #scale values per channel
        for channel in range(dataset.num_channels):#range(0,1):#
            maxIntensity = dataset.maxIntensityOfChannel(channel)
            minIntensity = dataset.minIntensityOfChannel(channel)
            # scale max intensity
            vimgStack[:,channel*targetSize:channel*targetSize+targetSize] -= minIntensity
            vimgStack[:,channel*targetSize:channel*targetSize+targetSize] /= 0.5*(maxIntensity-minIntensity)#-minIntensity)
            vimgStack[:,channel*targetSize:channel*targetSize+targetSize] = np.clip(vimgStack[:,channel*targetSize:channel*targetSize+targetSize], 0, 1)


        vimgStack = adjust_gamma(vimgStack, gamma)
        cv2.imshow('Image',vimgStack)

        key = cv2.waitKey(0)
        
        if key & 0xFF == ord('7'):
            gamma += 0.1
            print("Gamma: ", gamma) 
        if key & 0xFF == ord('4'):
            gamma -= 0.1
            print("Gamma: ", gamma) 

            
        imageCounter += 1
        if key==27: # Esc key to exit
            break 
            

    cv2.destroyAllWindows()



def visualizeDataset(dataset, targetSize, oneBlobsOnly = False, filterChannel = 0):

    imageCounter = 0
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    multFactor = 1000000

    def rescaleImage(im, minRange, maxRange):
        return (im - minRange) / (maxRange - minRange)

    def changeRange(x):
        maxRange=cv2.getTrackbarPos("Max", 'Image') / float(multFactor)
        minRange=cv2.getTrackbarPos("Min", 'Image') / float(multFactor)
        imgScaled = rescaleImage(imgStack, minRange, maxRange)
        cv2.imshow('Image',imgScaled)

    cv2.createTrackbar("Max", "Image",int(1*multFactor),int(1*multFactor),changeRange)
    cv2.createTrackbar("Min", "Image",int(0*multFactor),int(1*multFactor),changeRange)

    while (not dataset.eod()):


 
        image = dataset.nextImage()
        print ("Image " + repr(imageCounter), image.shape)
        if isinstance(image, tuple):
            image = image[0]
        if (len(image.shape) == 4):
            image = image[0,:,:,:]
       
        print ("Image " + repr(imageCounter), image.shape)
        print("target", targetSize)
        image = pad_or_crop(image, targetSize, 'symmetric')
 

        print ("Image " + repr(imageCounter), image.shape)

        imgStack = np.empty([targetSize,0])
   

        for channel in range(image.shape[-1]):#range(0,1):#
            img = image[:,:,channel]
            print(img.dtype)
            print("max", np.max(img))
            img /= np.amax(img)

         

            imgStack = np.hstack((imgStack, img))
            
            # cv2.putText(imgStack,"BF", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255)
          

    
        cv2.imshow('Image',imgStack)

        changeRange(0)
        chr = cv2.waitKey(0)
        
        imageCounter += 1
        if chr==27: # Esc key to exit
            break 
            

    cv2.destroyAllWindows()

def visualizeDataset_withmask(dataset, targetSize, oneBlobsOnly = False, filterChannel = 0):
    # dataset.reset()

    imageCounter = 0
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    multFactor = 1000000

    def rescaleImage(im, minRange, maxRange):
        return (im - minRange) / (maxRange - minRange)

    def changeRange(x):
        maxRange=cv2.getTrackbarPos("Max", 'Image') / float(multFactor)
        minRange=cv2.getTrackbarPos("Min", 'Image') / float(multFactor)
        imgScaled = rescaleImage(output, minRange, maxRange)
        cv2.imshow('Image',imgScaled)

    cv2.createTrackbar("Max", "Image",int(1*multFactor),int(1*multFactor),changeRange)
    cv2.createTrackbar("Min", "Image",int(0*multFactor),int(1*multFactor),changeRange)

    while (not dataset.eod()):


        # [image, _] = dataset.nextImage()
        image, mask = dataset.nextImage_withmask()


        if (len(image.shape) == 4):
            image = image[0,:,:,:]
        if (len(mask.shape) == 4):
            mask = mask[0,:,:,:]

        centerChannel = 0
        # use mask of first channel (bright field) to center the data
        uniqueVals, uniqueCount = np.unique(mask[:,:,centerChannel], return_counts = True)
        if uniqueCount.shape[0] == 1: # only black pixels in mask, no reference to center the cell
            print("skipping ", imageCounter)
            imageCounter += 1
            continue

        blobIntensity = uniqueVals[np.argmax(uniqueCount[1:]) + 1] # ignore black pixels in mask
        image, mask = center_crop_pad(image, mask, centerChannel, blobIntensity, targetSize)

        print ("Image " + repr(imageCounter), image.shape)

        imgStack = np.empty([targetSize,0])
        imgStack_msk = np.empty([targetSize,0])

        for channel in range(image.shape[-1]):#range(0,1):#
            img = image[:,:,channel]
            img /= np.amax(img)

            msk = mask[:,:,channel]
            msk /= np.amax(msk)

            #img, msk = center_crop_pad(img, msk, targetSize, 'symmetric')#pad_or_crop(img, targetSize, 'symmetric')# pad_or_crop(img, image_size, 'constant', constant_values=(0))
            #msk = pad_or_crop(msk, targetSize, 'symmetric')#center_crop_pad(mask, mask, targetSize, 'symmetric')#pad_or_crop(msk, targetSize, 'symmetric')
    

            # else:
                # break

            # cv2.imwrite("cell_" + repr(imageCounter) + "_c_" + repr(channel) + ".jpg", img*255)
            # cv2.imwrite("mask_" + repr(imageCounter) + "_c_" + repr(channel) + ".jpg", mask*255)
            imgStack = np.hstack((imgStack, img))
            imgStack_msk = np.hstack((imgStack_msk, msk))
            # cv2.putText(imgStack,"BF", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255)
          
        output = np.vstack((imgStack, imgStack_msk))
    
        cv2.imshow('Image',output)

        changeRange(0)
        chr = cv2.waitKey(0)
        
        imageCounter += 1
        if chr==27: # Esc key to exit
            break 
            

    cv2.destroyAllWindows()


def visualizeCIFDataset(dataset, targetSize, oneBlobsOnly = False, filterChannel = 0):
    dataset.reset()

    imageCounter = 0
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    multFactor = 1000000

    def rescaleImage(im, minRange, maxRange):
        return (im - minRange) / (maxRange - minRange)

    def changeRange(x):
        maxRange=cv2.getTrackbarPos("Max", 'Image') / float(multFactor)
        minRange=cv2.getTrackbarPos("Min", 'Image') / float(multFactor)
        imgScaled = rescaleImage(imgStack, minRange, maxRange)
        cv2.imshow('Image',imgScaled)

    cv2.createTrackbar("Max", "Image",int(1*multFactor),int(1*multFactor),changeRange)
    cv2.createTrackbar("Min", "Image",int(0*multFactor),int(1*multFactor),changeRange)

    while (not dataset.eod()):


        image, maskImage = dataset.nextImage()

        
        print ("Image " + repr(imageCounter), image.shape)

        imgStack = np.empty([targetSize,0])
        maskStack = np.empty([targetSize,0])
        
        print(maskImage.shape)
        # print(filterChannel)
        mask = maskImage[:,:,filterChannel]
        # if (not hasOneBlob(mask)):
        #     continue

        for channel in range(image.shape[-1]):#range(0,1):#
            img = image[:,:,channel]
            img = img / np.amax(img)
            mask = maskImage[:,:,channel]

            img = pad_or_crop(img, targetSize, 'symmetric')# pad_or_crop(img, image_size, 'constant', constant_values=(0))
            mask = pad_or_crop(mask, targetSize, 'symmetric')# pad_or_crop(img, image_size, 'constant', constant_values=(0))



            maxMask = np.amax(mask)
            if (maxMask != 0):
                mask = mask / maxMask
            # else:
                # break

            # cv2.imwrite("cell_" + repr(imageCounter) + "_c_" + repr(channel) + ".jpg", img*255)
            # cv2.imwrite("mask_" + repr(imageCounter) + "_c_" + repr(channel) + ".jpg", mask*255)
            imgStack = np.hstack((imgStack, img))
            maskStack = np.hstack((maskStack, mask))
        
      
        
        
        
    
        
        cv2.imshow('Image',imgStack)
        cv2.imshow('Mask',maskStack)
        changeRange(0)
        chr = cv2.waitKey(0)
        
        imageCounter += 1
        if chr==27: # Esc key to exit
            break 
            

    cv2.destroyAllWindows()


def showImgAndMask(image, maskImage, targetSize, oneBlobsOnly = False, filterChannel = 0):


    imageCounter = 0
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    multFactor = 1000000

    def rescaleImage(im, minRange, maxRange):
        return (im - minRange) / (maxRange - minRange)

    def changeRange(x):
        maxRange=cv2.getTrackbarPos("Max", 'Image') / float(multFactor)
        minRange=cv2.getTrackbarPos("Min", 'Image') / float(multFactor)
        imgScaled = rescaleImage(imgStack, minRange, maxRange)
        cv2.imshow('Image',imgScaled)

    cv2.createTrackbar("Max", "Image",int(1*multFactor),int(1*multFactor),changeRange)
    cv2.createTrackbar("Min", "Image",int(0*multFactor),int(1*multFactor),changeRange)

    



# (Optional) Stretch the intensity scale to visible 8-bit images.
def _rescale(image):
    import skimage
    import scipy.misc
    vmin, vmax = scipy.stats.scoreatpercentile(image, (0.01, 99.95))

    return skimage.exposure.rescale_intensity(image, in_range=(vmin, vmax), out_range=np.uint8).astype(np.uint8)


def showMontageOverlay(dataset, channels, targetSize, montageSize):
    
    if len(channels) > 3:
        print("Cannot show montage overlay for more than 3 channels.")
        return

    
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    while (not dataset.eod()):



        image, maskImage = dataset.nextImage()

        images = []
        for c in range(len(channels)):
            print("Channel",channels[c])
            
            img = pad_or_crop(image[:,:,channels[c]], targetSize, 'symmetric')# pad_or_crop(img, image_size, 'constant', constant_values=(0))
            # img = img / np.amax(img)
            img = _rescale(img)
            
            images.append(img)
            # print(image)
            # print(image.shape)

        if len(channels) < 3:
            images.append(images[0])
            if len(channels) < 3:
                images.append(images[0])
        # images[1] = images[1] + images[0] 

        
        img = cv2.merge(images)
        # img = _rescale(img)
        cv2.imshow('Image',img)

        chr = cv2.waitKey(0)
        
        
        if chr==27: # Esc key to exit
            break 
            

    cv2.destroyAllWindows() 
    

