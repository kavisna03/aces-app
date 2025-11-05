import os
import sys
import numpy as np
import cv2 as cv
from scenedetect import detect, AdaptiveDetector
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as compare_ssim
import fast_colorthief
from PIL import Image as im
import math


# extract frames at a rate of 1fps (including the first frame) to reduce the number of extracted frames while retaining the essential information
def extract_frames(video_path):
    cap = None

    try:
        frame_num = 0
        frames = []

        cap = cv.VideoCapture(video_path)

        # check if the video file was opened successfully
        if not cap.isOpened():
            raise ValueError("Video file could not be opened.")

        while True:
            isTrue, frame = cap.read()

            if not isTrue:
                break

            if frame_num % int(cap.get(cv.CAP_PROP_FPS)) == 0:
                frames.append(frame)

            frame_num += 1

        return frames
    except Exception as e:
        sys.exit(f"Error: {e}")
    finally:
        if cap is not None:
            cap.release()


# 1. Frame Rate (fps)
# get the video's frame rate in frames per second (fps)
def get_frame_rate(video_path):
    cap = None

    try:
        cap = cv.VideoCapture(video_path)

        # check if the video file was opened successfully
        if not cap.isOpened():
            raise ValueError("Video file could not be opened.")

        frame_rate = cap.get(cv.CAP_PROP_FPS)

        return frame_rate
    except Exception as e:
        sys.exit(f"Error: {e}")
    finally:
        if cap is not None:
            cap.release()


# 2. Scene Transition Rate (Per Minute)
# calculate the scene transition rate (per minute) using the formula: rate = total no. of scene transitions / video length (in minutes)
# adaptive detector algorithm is used to detect the scenes as it uses content detector algorithm for detection with minimal risk of detecting false scenes in situations like camera movements
# content detector algorithm detects scenes based on changes in colour and intensity between frames
def get_scene_trans_rate(video_path):
    cap = None

    try:
        scene_list = detect(video_path, AdaptiveDetector())

        if len(scene_list) == 0:
            total_scene_trans = 0
        else:
            total_scene_trans = len(scene_list) - 1

        cap = cv.VideoCapture(video_path)

        # check if the video file was opened successfully
        if not cap.isOpened():
            raise ValueError("Video file could not be opened.")

        total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        frame_rate = cap.get(cv.CAP_PROP_FPS)
        video_length = (total_frames / frame_rate) / 60

        scene_trans_rate = total_scene_trans / video_length

        return scene_trans_rate
    except Exception as e:
        sys.exit(f"Error: {e}")
    finally:
        if cap is not None:
            cap.release()


# 3. Average Scene Duration (in Minutes)
# calculate the average scene duration in minutes using the formula: avg = total duration of all scenes (in minutes) / total no. of scenes
# if zero scenes are detected in a video, it means that there is only one scene spanning the whole video, so average scene duration = video length
# adaptive detector algorithm is used to detect the scenes as it uses content detector algorithm for detection with minimal risk of detecting false scenes in situations like camera movements
# content detector algorithm detects scenes based on changes in colour and intensity between frames
def get_avg_scene_dur(video_path):
    try:
        total_duration = 0

        scene_list = detect(video_path, AdaptiveDetector())

        for scene in scene_list:
            start_time = (scene[0].get_frames() / scene[0].get_framerate()) / 60
            end_time = (scene[1].get_frames() / scene[1].get_framerate()) / 60
            duration = end_time - start_time
            total_duration += duration

        avg_scene_dur = total_duration / len(scene_list)

        return avg_scene_dur
    except ZeroDivisionError:
        cap = None

        try:
            cap = cv.VideoCapture(video_path)

            # check if the video file was opened successfully
            if not cap.isOpened():
                raise ValueError("Video file could not be opened.")

            total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
            frame_rate = cap.get(cv.CAP_PROP_FPS)

            video_length = (total_frames / frame_rate) / 60
            avg_scene_dur = video_length

            return avg_scene_dur
        finally:
            if cap is not None:
                cap.release()
    except Exception as e:
        sys.exit(f"Error: {e}")


# 4. Average Motion Intensity
# calculate the average motion intensity between one pair of consecutive frames using the formula: avg = total average magnitude of optical flow for all frame pairs / total number of frame pairs
# motion intensity is calculated using the optical flow method which detects the motion of objects between two consecutive frames in a video
# specifically, dense optical flow will be computed using the Gunnar Farneback algorithm as it looks at all the points in the image instead of corners like other algorithms
def get_avg_motion_intensity(frames):
    try:
        total_frame_pairs = len(frames) - 1
        total_avg_mag = 0

        # assign the first frame as the previous frame
        prv_f = frames[0]

        for n in range(len(frames)-1):
            next_f = frames[n+1]

            # convert the previous frame from BGR to grayscale
            prv_f_gray = cv.cvtColor(prv_f, cv.COLOR_BGR2GRAY)
            # convert the next frame (subsequent frame in the list) from BGR to grayscale
            next_f_gray = cv.cvtColor(next_f, cv.COLOR_BGR2GRAY)

            # calculate optical flow between the two consecutive frames
            flow = cv.calcOpticalFlowFarneback(prv_f_gray, next_f_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # calculate the magnitude of the optical flow
            mag, _= cv.cartToPolar(flow[..., 0], flow[..., 1])

            total_avg_mag += np.mean(mag)

            # assign the next frame as the previous frame for the next optical flow computation
            prv_f = next_f

        avg_motion_intensity = total_avg_mag/total_frame_pairs

        return avg_motion_intensity
    except Exception as e:
        sys.exit(f"Error: {e}")


# 5. Texture Features - Average Texture Contrast, Average Texture Homogeneity
# calculate the average texture contrast and average texture homogeneity in one frame using the formula: avg = total average contrast or total average homogeneity / total number of frames
# compute the texture contrast and texture homegeneity features using Gray Level Co-occurrence Matrix (GLCM) method
# GLCM provides rich texture information by considering the pixel intensity relationships between neighbouring pixels
# texture contrast indicates the size of the variations between the neighbouring pixel intensities
# texture homogeneity indicates how uniform the texture is
def get_texture_features(frames):
    try:
        total_avg_contrast = 0
        total_avg_homogeneity = 0

        for frame in frames:
            # convert the frame from 3D BGR array to 2D grayscale array
            frame_gray = cv.cvtColor(np.array(frame), cv.COLOR_BGR2GRAY)
            # scale the grayscale frame pixel values from the normalized range [0, 1] to the 8-bit range [0, 255]
            # convert the scaled pixel values into uint8 data type to represent the values from 0 to 255
            image = (255 * frame_gray).astype(np.uint8)

            # create glcm from the image with 255 different gray intensity levels, using a distance of 50 between two pixels that are in the vertical direction as the parameters
            glcm = graycomatrix(image, distances=[50], angles=[np.pi/2], levels=256)

            # extract the texture features using glcm
            contrast = graycoprops(glcm, 'contrast')
            homogeneity = graycoprops(glcm, 'homogeneity')

            total_avg_contrast += np.mean(contrast)
            total_avg_homogeneity += np.mean(homogeneity)

        avg_contrast = total_avg_contrast/len(frames)
        avg_homogeneity = total_avg_homogeneity/len(frames)

        return avg_contrast, avg_homogeneity
    except Exception as e:
        sys.exit(f"Error: {e}")


# 6. Dominant Color Standard Deviation
# calculate the dominant color standard deviation which is the square root of variance in dominant color in each color channgel (R, G, B) across the frames in a video
def get_dom_color_sd(frames):
    try:
        dom_colors = []

        for frame in frames:
            # convert numpy array of frame to image format and save it because ColorThief method only accepts image file
            image = im.fromarray(frame)
            image.save('temp_img.jpg')

            try:
                # get the one most dominant color from each frame
                dom_color = fast_colorthief.get_dominant_color('temp_img.jpg', quality=1)
            except RuntimeError:
                # if the color is greater than rgb(250, 250, 250), it will not be included in the array of colors by fast_colorthief
                # this may result in an empty array of colors that will give an error during quantization
                # quantization is the process of picking the dominant color by reducing the number of colors in the array of colors obtained from the image
                # if this error occurs, dominant color will be set to the rgb(255, 255, 255) which is the maximum value
                dom_color = [255, 255, 255]

            # store the dominant color of every frame in a list
            dom_colors.append(dom_color)

            # remove the image file to save space for storing the next frame as image using the same name
            os.remove('temp_img.jpg')

        dom_color_sd = np.std(dom_colors, axis=0, dtype=np.float64)

        return dom_color_sd
    except Exception as e:
        sys.exit(f"Error: {e}")


# 7. Bit Rate (Mbps)
# get the video's bit rate in mega bits per second (Mbps) using the formula bit rate = ((video file size * 8)/video_length)/(10^6)
# bit rate is the number of bits or amount of data conveyed per unit time
def get_bit_rate(video_path):
    cap = None

    try:
        video_file_size = os.path.getsize(video_path)

        cap = cv.VideoCapture(video_path)

        # check if the video file was opened successfully
        if not cap.isOpened():
            raise ValueError("Video file could not be opened.")

        total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        frame_rate = cap.get(cv.CAP_PROP_FPS)
        video_length = total_frames / frame_rate

        bit_rate = ((video_file_size * 8) / video_length) / (10 ** 6)

        return bit_rate
    except Exception as e:
        sys.exit(f"Error: {e}")
    finally:
        if cap is not None:
            cap.release()


# 8. Compression Ratio
# calculate the compression ratio using the formula: ratio = compressed video file size (when downloaded)/uncompressed video file size
# uncompressed video file = frame height * frame width * bit depth * number of color channels * total number of frames
def get_comp_ratio(video_path):
    cap = None

    try:
        cap = cv.VideoCapture(video_path)

        # check if the video file was opened successfully
        if not cap.isOpened():
            raise ValueError("Video file could not be opened.")

        # video frame dimension
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)

        _, frame = cap.read()

        # bit depth which is the number of bits used to represent each color channel
        bit_depth = int(str(frame.dtype).removeprefix('uint'))
        # shape of RGB frame is (height, width, color channels)
        # shape of grayscale frame is (height, width)
        # number of color channels is the 3rd value in frame shape if it is RGB frame, otherwise it is 1 for grayscale frame
        num_col_channels = frame.shape[2] if len(frame.shape) == 3 else 1

        # total frames in the video
        total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)

        # calculate the uncompressed video file size in bytes
        uncompressed_size = (height * width * bit_depth * num_col_channels * total_frames) / 8

        # get the compressed video file size
        compressed_size = os.path.getsize(video_path)

        # calculate the compression ratip
        comp_ratio = compressed_size / uncompressed_size

        return comp_ratio
    except Exception as e:
        sys.exit(f"Error: {e}")
    finally:
        if cap is not None:
            cap.release()


# 9. Average Peak Signal-to-Noise Ratio (PSNR) (dB)
# calculate the PSNR in decibels (dB) between two consecutive images or frames
# PSNR indicates the quality of an image by computing the ratio between the maximum power of a signal and the power of noise in an image
def img_psnr(img1, img2):
    try:
        # compute mse
        mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)

        # compute psnr
        if mse < 1e-10:
            psnr = 100
        else:
            psnr = 20 * math.log10(1 / math.sqrt(mse))

        return psnr
    except Exception as e:
        sys.exit(f"Error: {e}")


# calculate the PSNR for all the frames in a video and find the average PSNR in decibals (dB)
def calculate_avg_psnr(frames):
    try:
        psnr_results = []

        for i in range(1, len(frames)):
            psnr = img_psnr(frames[i - 1], frames[i])
            psnr_results.append(psnr)

        mean_psnr = np.mean(psnr_results) if psnr_results else 0

        return mean_psnr
    except Exception as e:
        sys.exit(f"Error: {e}")


# 10. Average Structural Similarity Index (SSIM)
# calculate the SSIM between two consecutive images or frames
# SSIM measures the similarity between two images based on the difference in luminance, contrast, and structure
def img_ssim(img1, img2):
    try:
        # convert images to grayscale for SSIM calculation
        img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        ssim_index, _ = compare_ssim(img1_gray, img2_gray, full=True)

        return ssim_index
    except Exception as e:
        sys.exit(f"Error: {e}")


# calculate the SSIM for all the frames in a video and find the average SSIM
def calculate_avg_ssim(frames):
    try:
        ssim_results = []

        for i in range(1, len(frames)):
            ssim = img_ssim(frames[i - 1], frames[i])
            ssim_results.append(ssim)

        mean_ssim = np.mean(ssim_results) if ssim_results else 0

        return mean_ssim
    except Exception as e:
        sys.exit(f"Error: {e}")


# 11. Average Color Standard Deviation
# calculate the average color standard deviation which is the square root of variance in average color in each color channel (R, G, B) across all the frames in the video
def get_avg_col_sd(frames):
    try:
        avg_frame_cols = []

        for frame in frames:
            avg_fcol_row = np.mean(frame, axis=0)
            avg_fcol = np.mean(avg_fcol_row, axis=0)
            # store the average color of each frame in a list
            avg_frame_cols.append(avg_fcol)

        avg_col_sd = np.std(avg_frame_cols, axis=0, dtype=np.float64)

        return avg_col_sd
    except Exception as e:
        sys.exit(f"Error: {e}")


# define a function to perform feature engineering
def get_video_features(video_path):
    frames = extract_frames(video_path)
    print("Frame extraction has been completed.")

    frame_rate = get_frame_rate(video_path)
    scene_trans_rate = get_scene_trans_rate(video_path)
    avg_scene_dur = get_avg_scene_dur(video_path)
    avg_motion_int = get_avg_motion_intensity(frames)
    avg_contrast, avg_homogeneity = get_texture_features(frames)
    dom_col_sd = get_dom_color_sd(frames)
    bit_rate = get_bit_rate(video_path)
    compression_ratio = get_comp_ratio(video_path)
    avg_psnr = calculate_avg_psnr(frames)
    avg_ssim = calculate_avg_ssim(frames)
    avg_col_sd = get_avg_col_sd(frames)

    video_features = [frame_rate, scene_trans_rate, avg_scene_dur, avg_motion_int,
                      avg_contrast, avg_homogeneity, dom_col_sd[0], dom_col_sd[1],
                      dom_col_sd[2], bit_rate, compression_ratio, avg_psnr,
                      avg_ssim, avg_col_sd[0], avg_col_sd[1], avg_col_sd[2]]

    print("Feature engineering has been completed.")

    return video_features