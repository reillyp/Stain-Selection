import cv2
import numpy as np
#import sys
#from tkinter import messagebox

# Rework as a Class as per:
# http://introtopython.org/classes.html includes storing a class as a module
# Class to be instantiated in GUI support module.
# Input variables will be passed via settter and getter functions.

class SelectStains():
    
    def __init__(self):
        # Initial Values are blank. 
        # Values provided when GUI calls setters.
        self.image_in = ''
        self.base_ext = ''
        self.base = ''
        self.dir_name = ''
        self.lower_threshold_list = []
        self.upper_threshold_list = []
        self.area_threshold_value = ''
        self.gaussian_blur = ''

    def setImageIn(self, input_image, base_ext, base):
        self.image_in = input_image
        self.base_ext = base_ext
        self.base = base

    def setOutputDir(self, dirname):        
        self.dir_name = dirname

    def setHSVThresholdsBlurArea(self, lower_threshold_list, upper_threshold_list, gaussian_blur, area_threshold_value,):
        self.lower_threshold_list = lower_threshold_list
        self.upper_threshold_list = upper_threshold_list
        self.gaussian_blur = gaussian_blur
        self.area_threshold_value = area_threshold_value
        
    def resetVariables(self):
        self.image_in = ''
        self.base_ext = ''
        self.base = ''
        self.dir_name = ''
        self.lower_threshold_list = []
        self.upper_threshold_list = []
        self.area_threshold_value = ''
        self.gaussian_blur = ''

    def variableCheck(self):
        print("SelectStains Class Input Img: {}".format(self.image_in))
        print("SelectStains Class Filename.ext: {}".format(self.base_ext))
        print("SelectStains Class Filename: {}".format(self.base))
        print("SelectStains Class Output Dir: {}".format(self.dir_name))
        print("SelectStains Class Lower HSV Threshold: {}".format(self.lower_threshold_list))
        print("SelectStains Class Upper HSV Threshold: {}".format(self.upper_threshold_list))
        print("SelectStains Class Gaussian Blur: {}".format(self.gaussian_blur))
        print("SelectStains Class Area Threshold: {}".format(self.area_threshold_value))

    def convertToHSV(self):
        img=cv2.imread(self.image_in)
        cv2.imshow(self.image_in,img)
        img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        str_output_file_hsv = self.dir_name+"/"+self.base+"_hsv.tif"
        cv2.imshow(str_output_file_hsv,img_hsv)
        cv2.imwrite(str_output_file_hsv, img_hsv)

        input_file_labeled_objects = self.dir_name+"/"+self.base+"-labeled-objects.jpg"

        ## Hue range for red in 360 color wheel ranges from ~0-30 and ~330-360
        ## Python range for 360 color wheel ranges from 0-255 to fit an 8 bit value.
        lower_stain = np.array(self.lower_threshold_list)## USER: ENTER LOWER HSV RANGE BETWEEN []
        upper_stain = np.array(self.upper_threshold_list)## USER: ENTER UPPER HSV RANGE BETWEEN []

        str_lower_stain = "Lower stain HSV: {}\n".format(lower_stain)
        str_upper_stain = "Upper stain HSV: {}\n".format(upper_stain)

        mask_stain = cv2.inRange(img_hsv, lower_stain, upper_stain)
        cv2.imshow("mask_stain", mask_stain)
        ####cv2.imwrite("mask_stain.jpg", mask_stain)
        ##
        ### Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask_stain)
        ##
        cv2.imshow("res",res)
        ####cv2.imwrite("res.jpg",res)
        
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask_stain)

        cv2.imshow("res",res)
        ##cv2.imwrite("res.jpg",res)

        ret, structures_thresh = cv2.threshold(mask_stain, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("structures_thresholds", structures_thresh)
        ##cv2.imwrite("structures_thresholds.jpg", structures_thresh )

        ## USER: SET GAUSSIAN BLUR PARAMETERS IN THIS BLOCK ############################
        gaussian_range = self.gaussian_blur ## USER: MODIFY THIS VALUE IF NEEDED
        
        ## For GaussianBlur width and height of kernel which should be positive and odd. Reduces gaussian noise.
        gaussian_range_parameter = (gaussian_range,gaussian_range)
        print("gaussian_range_parameter: {}".format(gaussian_range_parameter))
        mask_stain_blur = cv2.GaussianBlur(mask_stain, gaussian_range_parameter,0)
        str_gaussian_blur = "Gaussian Blur Range: {}\n".format(gaussian_range)

        # NOTE: Threshold REQUIRES a greyscale image therefore using mask.
        ret, structures_thresh_blur = cv2.threshold(mask_stain_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("structures_thresholds_blur", structures_thresh_blur)
        #cv2.imwrite("structures_thresholds_blur.jpg", structures_thresh_blur)

        # Bitwise-AND mask and original image
        res_blur = cv2.bitwise_and(img, img, mask=structures_thresh_blur)
        cv2.imshow("res_blur",res_blur)
        ##cv2.imwrite("res_blur.jpg",res_blur)

        ## Corrected
        contours, hierarchy = cv2.findContours(structures_thresh_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        structures2 = structures_thresh_blur.copy()
        color = (255, 0, 0)
        ## Total number of contours found before threshold is applied.
        print(len(contours))
        print(" ")
        
        objects = np.zeros([structures2.shape[0], structures2.shape[1], 3], 'uint8')
        indexOfContours = 1
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

        ## USER: SET MINIMUM AREA THRESHOLD HERE #######################################
        area_treshold = self.area_threshold_value 
        ################################################################################
        logfilename = self.dir_name+"/"+self.base+"Contour_Data.csv"
        f = open(logfilename, "a")
        str_area_threshold = "Area Threshold: {}\n".format(area_treshold)
        str_contour_header = "Contour Index, Area, Perimeter,\n"
        f.write(str_lower_stain)
        f.write(str_upper_stain)
        f.write(str_gaussian_blur)
        f.write(str_area_threshold)
        f.write(str_contour_header)
        f.close()
        print(str_lower_stain)
        print(str_upper_stain)
        print(str_gaussian_blur)
        print(str_area_threshold)

        for c in contours:

            area = cv2.contourArea(c)
            
            if area > area_treshold:

                cv2.drawContours(objects, [c], -1, color, -1)
                perimeter = cv2.arcLength(c, True)

                M = cv2.moments(c)

                ## Divide by zero check
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                else:
                    cx = int(M['m10'] )
                if M['m00'] != 0:
                    cy = int(M['m01'] / M['m00'])
                else:
                    cy = int(M['m01'])
                
                indexString = str(indexOfContours)

                cv2.putText(objects, indexString, (cx, cy), font, .4, (255, 255, 255))

                #cv2.circle(objects, (cx, cy), 4, (255, 0, 0), -1)

        ##        contourString = "Contour Index: {}, Area: {}, perimeter: {}\n".format(indexOfContours, area, perimeter)
                contourString = "{}, {}, {}\n".format(indexOfContours, area, perimeter)
                print(contourString)        
                f = open(logfilename, "a")
                f.write(contourString)
                f.close()
                indexOfContours += 1

        cv2.imshow(input_file_labeled_objects, objects)
        cv2.imwrite(input_file_labeled_objects, objects)

        object_mask = cv2.imread(input_file_labeled_objects);

        object_lower_stain = np.array([250, 0, 0])
        object_upper_stain = np.array([255, 255, 255])
         
        object_mask_stain = cv2.inRange(object_mask, object_lower_stain, object_upper_stain)
        object_mask_res = cv2.bitwise_and(img, img, mask=object_mask_stain)

        input_file_object_mask = self.dir_name+"/"+self.base+"-object-mask.jpg"
        cv2.imshow(input_file_object_mask,object_mask_res)
        cv2.imwrite(input_file_object_mask,object_mask_res)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

##
## Following code from select-stain-count-label-generic-input-file rev1.py September 16, 2019
##import cv2
##import numpy as np
##
#### USER: ENTER FILE TO BE PROCESSED AS "input_file" IN THIS BLOCK###############
#### USER: FILE SIZE MAY NEED TO BE REDUCED ######################################
##input_file = "450_R_0051 - 2.tif" ## USER: ENTER FILE NAME BETWEEN QUOTES
##
##img=cv2.imread(input_file)
##cv2.imshow(input_file,img)
##img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
##str_input_file_hsv = input_file +"_hsv"
##cv2.imshow(str_input_file_hsv,img_hsv)
##str_input_file_hsv_tif = str_input_file_hsv+".tif"
##cv2.imwrite(str_input_file_hsv_tif, img_hsv)
##################################################################################
##
#### USER: SET AND RECORD HSV UPPER AND LOWER RANGES IN THIS BLOCK ###############
#### Hue range for red in 360 color wheel ranges from ~0-30 and ~330-360
#### Python range for 360 color wheel ranges from 0-255 to fit an 8 bit value.
##lower_stain = np.array([71, 56, 100])## USER: ENTER LOWER HSV RANGE BETWEEN []
##upper_stain = np.array([171, 156, 190])## USER: ENTER UPPER HSV RANGE BETWEEN []
##
##str_lower_stain = "Lower stain HSV: {}\n".format(lower_stain)
##str_upper_stain = "Upper stain HSV: {}\n".format(upper_stain)
##################################################################################
## 
##mask_stain = cv2.inRange(img_hsv, lower_stain, upper_stain)
##cv2.imshow("mask_stain", mask_stain)
####cv2.imwrite("mask_stain.jpg", mask_stain)
##
### Bitwise-AND mask and original image
##res = cv2.bitwise_and(img, img, mask=mask_stain)
##
##cv2.imshow("res",res)
####cv2.imwrite("res.jpg",res)
##
##ret, structures_thresh = cv2.threshold(mask_stain, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##cv2.imshow("structures_thresholds", structures_thresh)
####cv2.imwrite("structures_thresholds.jpg", structures_thresh )
##
#### USER: SET GAUSSIAN BLUR PARAMETERS IN THIS BLOCK ############################
##gaussian_range = 15 ## USER: MODIFY THIS VALUE IF NEEDED
#### For GaussianBlur width and height of kernel which should be positive and odd. Reduces gaussian noise.
##gaussian_range_parameter = (gaussian_range,gaussian_range)
##mask_stain_blur = cv2.GaussianBlur(mask_stain, gaussian_range_parameter,0)
##str_gaussian_blur = "Gaussian Blur Range: {}\n".format(gaussian_range)
##################################################################################
##
### NOTE: Threshold REQUIRES a greyscale image therefore using mask.
##ret, structures_thresh_blur = cv2.threshold(mask_stain_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##cv2.imshow("structures_thresholds_blur", structures_thresh_blur)
###cv2.imwrite("structures_thresholds_blur.jpg", structures_thresh_blur)
##
### Bitwise-AND mask and original image
##res_blur = cv2.bitwise_and(img, img, mask=structures_thresh_blur)
##cv2.imshow("res_blur",res_blur)
####cv2.imwrite("res_blur.jpg",res_blur)
##
#### Corrected
##contours, hierarchy = cv2.findContours(structures_thresh_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##
##structures2 = structures_thresh_blur.copy()
##color = (255, 0, 0)
#### Total number of contours found before threshold is applied.
##print(len(contours))
##print(" ")
##
##objects = np.zeros([structures2.shape[0], structures2.shape[1], 3], 'uint8')
##indexOfContours = 1
##font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
##
#### USER: SET MINIMUM AREA THRESHOLD HERE #######################################
##area_treshold = 290 
##################################################################################
##logfilename =input_file+"Contour_Data.csv"
##f = open(logfilename, "a")
##str_area_threshold = "Area Threshold: {}\n".format(area_treshold)
##str_contour_header = "Contour Index, Area, Perimeter,\n"
##f.write(str_lower_stain)
##f.write(str_upper_stain)
##f.write(str_gaussian_blur)
##f.write(str_area_threshold)
##f.write(str_contour_header)
##f.close()
##print(str_lower_stain)
##print(str_upper_stain)
##print(str_gaussian_blur)
##print(str_area_threshold)
##
##for c in contours:
##
##    area = cv2.contourArea(c)
##    
##    if area > area_treshold:
##
##        cv2.drawContours(objects, [c], -1, color, -1)
##        perimeter = cv2.arcLength(c, True)
##
##        M = cv2.moments(c)
##
##        ## Divide by zero check
##        if M['m00'] != 0:
##            cx = int(M['m10'] / M['m00'])
##        else:
##            cx = int(M['m10'] )
##        if M['m00'] != 0:
##            cy = int(M['m01'] / M['m00'])
##        else:
##            cy = int(M['m01'])
##        
##        indexString = str(indexOfContours)
##
##        cv2.putText(objects, indexString, (cx, cy), font, .4, (255, 255, 255))
##
##        #cv2.circle(objects, (cx, cy), 4, (255, 0, 0), -1)
##
####        contourString = "Contour Index: {}, Area: {}, perimeter: {}\n".format(indexOfContours, area, perimeter)
##        contourString = "{}, {}, {}\n".format(indexOfContours, area, perimeter)
##        print(contourString)        
##        f = open(logfilename, "a")
##        f.write(contourString)
##        f.close()
##        indexOfContours += 1
##
##input_file_labeled_objects = input_file+"-labeled-objects.jpg"
##cv2.imshow(input_file_labeled_objects, objects)
##cv2.imwrite(input_file_labeled_objects, objects)
##
##object_mask = cv2.imread(input_file_labeled_objects);
##
##object_lower_stain = np.array([250, 0, 0])
##object_upper_stain = np.array([255, 255, 255])
## 
##object_mask_stain = cv2.inRange(object_mask, object_lower_stain, object_upper_stain)
##object_mask_res = cv2.bitwise_and(img, img, mask=object_mask_stain)
##
##input_file_object_mask = input_file+"-object-mask.jpg"
##cv2.imshow(input_file_object_mask,object_mask_res)
##cv2.imwrite(input_file_object_mask,object_mask_res)
##
##cv2.waitKey(0)
##cv2.destroyAllWindows()

