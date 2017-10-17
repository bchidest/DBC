'''
This code assumes that the training data is in the form of images PLUS their corresponding masks.
'''
import numpy as np
import cv2
import os
from glob import glob
import sys, csv
import argparse
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

'''
Add path to the he_image_processing for unmixing stains.
'''
sys.path.append(os.path.abspath("./processing"))
import he_processing as he

window_radius = 25
n_neg = 2000

parser = argparse.ArgumentParser(description='Nuclei dataset preprocessing script.')
parser.add_argument('cw_dir')
#parser.add_argument('cw_patch_dir')
parser.add_argument('cw_save_dir')
args = parser.parse_args()

'''
Specify patch and save directories. Assuming png images. Change otherwise.
'''

#cw_dir = "/media/data2/chidest1/cw_nuclei/" #"/media/data2/CW_nuclei/"
#cw_patch_dir = "/media/data2/chidest1/cw_patches/" #"/media/data2/CW_patches/"
#cw_save_dir = "/media/data2/vaishnavi/cw_patches/on_the_fly"
mask_list = glob(os.path.join(args.cw_dir, "*.png"))

color_map = [(0, 0, 0), (0, 255, 0), (182, 109, 255), (182, 219, 255),
             (109, 182, 255), (0, 109, 219), (0, 73, 73), (0, 146, 146), (255, 0, 0), (255, 0, 0), (255, 0, 0)]
#color_map = [(0, 0, 0), (73, 0, 146), (182, 109, 255), (182, 219, 255),
#             (109, 182, 255), (0, 109, 219), (0, 73, 73), (0, 146, 146), (255, 0, 0), (255, 0, 0), (255, 0, 0)]


class ContourMasks():
    '''
    This class processes files. Initialize with filename. Unmix the images. Setup the folders required to save. Then save with the correct class (one of (1) interior of nucleus, (2) interior boundary of nucleus, (3) exterior boundary of nucleus, (4) non boundary)
    '''
    def __init__(self, filename):
        print('Working on file ' + filename)
        self.filename = filename
        self.basename = os.path.split(self.filename.split("_mask")[0])[1]
        self.save_dir = os.path.join(args.cw_save_dir, self.basename)
        self.img_filename = os.path.join(args.cw_dir, self.basename + "_original.tif")
        self.mask = cv2.imread(self.filename, flags=0)
        self.img = cv2.imread(self.img_filename)
        self.imSize = self.img.shape
        self.valid_mask = np.zeros((self.imSize[0], self.imSize[1]), dtype='bool')
        self.valid_mask[window_radius + 1:-window_radius,
                        window_radius + 1:-window_radius] = True

    def unmixSave(self):
        '''
        Unmix and save with rotations and flips for easy reading during training.
        This function also identifies those pixels which are non-nuclei, so as to add to the negative examples list.
        '''

        # If unmixed/rotated images already exist, exit
        if os.path.exists(os.path.join(self.save_dir, 'onlyE.png')):
            return

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img, self.img_h, self.img_e = he.stain_normalization(self.img)
        #ret1, th1 = cv2.threshold(cv2.GaussianBlur(self.img_h, (21,21), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #th1 = 255*(cv2.GaussianBlur(self.img_h, (21,21), 0) >= 255).astype('uint8')
        th1 = 255*((self.img_h >=200)*(self.img_e <=170)).astype('uint8')
        neg_examples = list(np.where(th1 == 255)) #[:2000]
        perm = np.arange(len(neg_examples[0])); np.random.shuffle(perm)
        self.neg_examples = np.array(zip(neg_examples[0][perm[:2000]], neg_examples[1][perm[:2000]])).T
        Image.fromarray(th1).save(os.path.join(self.save_dir, 'onlyE.png'))
        Image.fromarray(self.img_h).save(os.path.join(self.save_dir, '0_0_h.png'))
        Image.fromarray(self.img_e).save(os.path.join(self.save_dir, '0_0_e.png'))
        Image.fromarray(np.rot90(self.img_h, 1)).save(os.path.join(self.save_dir, '90_0_h.png'))
        Image.fromarray(np.rot90(self.img_e, 1)).save(os.path.join(self.save_dir, '90_0_e.png'))
        Image.fromarray(np.rot90(self.img_h, 2)).save(os.path.join(self.save_dir, '180_0_h.png'))
        Image.fromarray(np.rot90(self.img_e, 2)).save(os.path.join(self.save_dir, '180_0_e.png'))
        Image.fromarray(np.rot90(self.img_h, 3)).save(os.path.join(self.save_dir, '270_0_h.png'))
        Image.fromarray(np.rot90(self.img_e, 3)).save(os.path.join(self.save_dir, '270_0_e.png'))
        Image.fromarray(np.fliplr(self.img_h)).save(os.path.join(self.save_dir, '0_1_h.png'))
        Image.fromarray(np.fliplr(self.img_e)).save(os.path.join(self.save_dir, '0_1_e.png'))
        Image.fromarray(np.flipud(self.img_h)).save(os.path.join(self.save_dir, '0_2_h.png'))
        Image.fromarray(np.flipud(self.img_e)).save(os.path.join(self.save_dir, '0_2_e.png'))

    def folderSetup(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        #else:
        #    old_patches = glob(os.path.join(self.save_dir, "*"))
        #    for old_patch in old_patches:
        #        os.remove(old_patch)

    def saveColorMap(self, class_map):
        class_map_rgb = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype='uint8')
        class_map_overlay = np.array(Image.open(self.img_filename))
        ind = np.where(class_map > 0)
        for i in range(len(ind[0])):
            class_map_rgb[ind[0][i], ind[1][i]] = color_map[class_map[ind[0][i], ind[1][i]]]
            class_map_overlay[ind[0][i], ind[1][i]] = color_map[class_map[ind[0][i], ind[1][i]]]
        Image.fromarray(class_map_rgb).save(os.path.join(self.save_dir, self.basename + '_class_map.png'))
        Image.fromarray(class_map_overlay).save(os.path.join(self.save_dir, self.basename + '_class_map_overlay.png'))

    def getMaskGrades2(self):
        min_contour_area = 5
        #min_contour_perimeter = 15

        file_path = os.path.join(self.save_dir, 'labels.pkl')
        print(file_path)

        columns = ['Nucleus_ID', 'Y', 'X', 'Class']
        columns_dtype = zip(columns, ['int', 'int', 'int', 'int'])
        pixels_all_df = pd.DataFrame.from_records(data=np.zeros(0, dtype=columns_dtype))

        self.pos_boundary = []
        self.neg_boundary = []
        self.interior = []
        #center_kernel = np.ones((3, 3), np.uint8)
        #center_kernel = np.array([(0, 1, 0), (1, 1, 1), (0, 1, 0)])
        center_kernel = np.array([(0, 0, 1, 0, 0),
                                  (0, 1, 1, 1, 0),
                                  (1, 1, 1, 1, 1),
                                  (0, 1, 1, 1, 0),
                                  (0, 0, 1, 0, 0)], np.uint8)
        ones_3_3_kernel = np.ones((3, 3), np.uint8)
        boundary_kernel = np.array([(0, 1, 0), (1, 1, 1), (0, 1, 0)], np.uint8)
        n_boundary_levels = 2
        n_classes = n_boundary_levels + 4

        m = (self.mask == 255).astype("uint8")
        #m2 = np.invert(m)
        _, contours, h = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #_, contours2, h2 = cv2.findContours(m2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        pixels_all = np.zeros(m.shape, dtype="uint8")
        nucleus_ind = 1
        for j in range(1, len(contours)):

            # Compute moments of contour
            moments = cv2.moments(contours[j])
            if moments['m00'] <= min_contour_area:
                print("Contour of size 0!")
                continue
            #if cv2.arcLength(contours[j], True) < min_contour_perimeter:
            #    print("Contour perimeter to small!")
            #    continue
            pixels = np.zeros(m.shape, dtype="uint8")
            #centers = np.zeros(m.shape, dtype="uint8")
            # Locations of pixels belonging to this nucleus
            cv2.drawContours(pixels, contours, j, color=(1), thickness=-1)
            pixels = cv2.dilate(pixels, ones_3_3_kernel)
            # NOTE: need to ensure that pixels aren't too close to border
            pixels = pixels*self.valid_mask
            loc = np.where(pixels == 1)
            # After removing invalid pixels, check that enough pixels remain
            if len(loc[0]) <= min_contour_area:
                print("Not enough pixels away from boundary!")
                continue
            # Init array
            pixels_ar = np.zeros(np.sum(pixels), dtype=columns_dtype)
            # Store array in dataframe
            pixels_df = pd.DataFrame.from_records(data=pixels_ar)
            # Draw both nuclei pixels and negative boundary as '2'
            pixels[pixels == 1] = 2
            # Draw over interior with '4'
            cv2.drawContours(pixels, contours, j, color=(4), thickness=-1)
            # Draw over positive boundary with '3'
            cv2.drawContours(pixels, contours, j, color=(3))

            # Draw center pixels (exact center and neighboring pixels)
            #center_x = int(moments['m10']/moments['m00'])
            #center_y = int(moments['m01']/moments['m00'])
            #centers[center_y, center_x] = 1
            #centers = cv2.dilate(centers, center_kernel)
            #pixels[centers == 1] = 5

#                interior = list(np.where(pixels >= 4))
#                neg_boundary = list(np.where(pixels == 2))
#                pos_boundary = list(np.where(pixels == 3))
#                self.pos_boundary.append(np.array(zip(pos_boundary[0], pos_boundary[1])).T)
#                self.neg_boundary.append(np.array(zip(neg_boundary[0], neg_boundary[1])).T)
#                self.interior.append(np.array(zip(interior[0], interior[1])).T)

            # Draw interior boundaries by eroding successive boundaries
            for k in range(0, n_boundary_levels):
                pixels = pixels + cv2.erode((pixels >= (k + 4)).astype(np.uint8), boundary_kernel)
            pixels = pixels*self.valid_mask

            if len(np.unique(pixels)) != n_classes:
                print("Not enough classes away from boundary!")
                continue
            # Add nucleus id and locations to dataframe
            pixels_df.loc[:, columns[0]] = nucleus_ind
            pixels_df.loc[:, columns[1:3]] = np.transpose(np.array(loc))
            # Add labels to array (need to subtract 1)
            pixels_df.loc[:, columns[3]] = pixels[loc] - 1
            # Store labels in pixels_all
            pixels_ind = pixels > 0
            pixels_all[pixels_ind] = pixels[pixels_ind]
            # Join dataframe to running dataframe
            pixels_all_df = pd.concat([pixels_all_df, pixels_df])
            nucleus_ind += 1

        print("\tNumber of nuclei = " + str(nucleus_ind))

        # Generate stromal samples (with Nucleus_ID = 0)
        img_h = np.array(Image.open(os.path.join(self.save_dir, '0_0_h.png')))
        img_e = np.array(Image.open(os.path.join(self.save_dir, '0_0_e.png')))
        th = 255*((img_h >= 200)*(img_e <= 170)).astype('uint8')
        th = th*self.valid_mask*(pixels_all == 0)
        neg_examples = np.where(th == 255)
        perm = np.arange(len(neg_examples[0]))
        np.random.shuffle(perm)
        neg_examples = zip(neg_examples[0][perm[0: n_neg]], neg_examples[1][perm[0: n_neg]])
        pixels_ar = np.zeros(n_neg, dtype=columns_dtype)
        pixels_df = pd.DataFrame.from_records(data=pixels_ar)
        pixels_df.loc[:, columns[0]] = 0
        pixels_df.loc[:, columns[1:3]] = np.array(neg_examples)
        pixels_df.loc[:, columns[3]] = 0
        for i in range(len(neg_examples)):
            # NOTE: OpenCV point (X,Y) considers X=horizontal
            cv2.circle(pixels_all, (neg_examples[i][1], neg_examples[i][0]),
                       radius=3, color=1, thickness=-1)
        pixels_all_df = pd.concat([pixels_all_df, pixels_df])
        pixels_all_df['Nucleus_ID'] = pixels_all_df['Nucleus_ID'].astype('int16')
        pixels_all_df['X'] = pixels_all_df['X'].astype('int16')
        pixels_all_df['Y'] = pixels_all_df['Y'].astype('int16')
        pixels_all_df['Class'] = pixels_all_df['Class'].astype('uint8')

        # NOTE: for Pandas 0.20, should change to to_pickle(file_path,
        #                                                   compression=None)
        pixels_all_df.to_pickle(file_path)
        self.saveColorMap(pixels_all)

    def getMaskGrades(self):

        pixels = (self.mask==255).astype("uint8")
        kernel = np.ones((3,3), np.uint8)
        gradient = cv2.morphologyEx(self.mask, cv2.MORPH_GRADIENT, kernel)
        _, contours, hierarchy = cv2.findContours(gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        self.pos_boundary = []
        self.neg_boundary = []
        self.interior = []

        # Nuclei ID is implicit. Identify if the contour is an exterior boundary or an interoir boundary. Add to the list of boundaries accordingly.

        if (hierarchy[0][2]==1):
            i = 0
        else:
            i = 1

        pixels_all = np.zeros(pixels.shape, dtype='uint8')
        while(hierarchy[i][2]!=-1):
            pixels_temp = np.zeros(pixels.shape, dtype='uint8')
            cv2.drawContours(pixels_temp, contours, i, color=(2))
            cv2.drawContours(pixels_temp, contours, hierarchy[i][0], color=(3))
            cv2.drawContours(pixels_temp, contours[hierarchy[i][0]], -1, color=(4))
            interior =  list(np.where(pixels_temp==2))
            neg_boundary = list(np.where(pixels_temp==3))
            pos_boundary = list(np.where(pixels_temp==4))
            if( len(interior[0])!=0 and len(neg_boundary[0])!=0 and len(pos_boundary[0])!=0):
                self.pos_boundary.append(np.array(zip(pos_boundary[0], pos_boundary[1])).T)
                self.neg_boundary.append(np.array(zip(neg_boundary[0], neg_boundary[1])).T)
                self.interior.append(np.array(zip(interior[0], interior[1])).T)
            i = hierarchy[i][0]
            pixels_all += pixels_temp
            self.saveColorMap(pixels_all)


#def saveCSV(self, grade, list_nuclei, nuclei=True):
#		'''
#		Function to save the patches.
#		file save format:
#        (0) image_id, (1) nucleus id, (2) label, (3) pixel location_x, (4) pixel location_y
#        (5) rotation value (6) flip
#		'''
#		file_path = os.path.join(self.save_dir, 'mask_values.csv') #grade_dict[grade])
#		local_dir = self.save_dir.split('/')[-1].split('_')[0]
#		with open(file_path, 'a') as csvfile:
#			writer = csv.writer(csvfile, delimiter=',')
#			if nuclei:
#				for j in range(len(list_nuclei)):
#					points = list_nuclei[j]
#					for i in range(len(points[0])):
#						if not(((points[0][i]-window_radius <0) or (points[0][i]+window_radius+1 > self.imSize[0])) or ((points[1][i]-window_radius<0) or (points[1][i]+window_radius+1>self.imSize[1]))):
#							writer.writerow([local_dir, j, grade, points[0][i], points[1][i], 0, 0])
#							#writer.writerow([local_dir, j, grade, points[0][i], points[1][i], 90, 0])
#							#writer.writerow([local_dir, j, grade, points[0][i], points[1][i], 180, 0])
#							#writer.writerow([local_dir, j, grade, points[0][i], points[1][i], 270, 0])
#							#writer.writerow([local_dir, j, grade, points[0][i], points[1][i], 0, 1])
#							#writer.writerow([local_dir, j, grade, points[0][i], points[1][i], 0, 2])
#			else:
#                                points = list_nuclei
#				for i in range(len(points[0])):
#					if not(((points[0][i]-window_radius <0) or (points[0][i]+window_radius+1 > self.imSize[0])) or ((points[1][i]-window_radius<0) or (points[1][i]+window_radius+1>self.imSize[1]))):
#						writer.writerow([local_dir, -1, grade, points[0][i], points[1][i], 0, 0])
#						#writer.writerow([local_dir, -1, grade, points[0][i], points[1][i], 90, 0])
#						#writer.writerow([local_dir, -1, grade, points[0][i], points[1][i], 180, 0])
#						#writer.writerow([local_dir, -1, grade, points[0][i], points[1][i], 270, 0])
#						#writer.writerow([local_dir, -1, grade, points[0][i], points[1][i], 0, 1])
#						#writer.writerow([local_dir, -1, grade, points[0][i], points[1][i], 0, 2])
#

for mask_filename in mask_list:
    contour = ContourMasks(mask_filename)
    contour.folderSetup()
    contour.unmixSave()
    contour.getMaskGrades2()
    #contour.getMaskGrades()
    #contour.saveCSV(3, contour.interior,     nuclei=True )
    #contour.saveCSV(2, contour.pos_boundary, nuclei=True )
    #contour.saveCSV(1, contour.neg_boundary, nuclei=True )
    #contour.saveCSV(0, contour.neg_examples, nuclei=False)
