import time
import numpy as np
import gc
import cv2, os, face_services.processors.wav2lip.audio as audio
from face_services.processors.face_enhancer import FaceEnhancer
import subprocess
from tqdm import tqdm
import torch, face_services.processors.wav2lip.face_detection as face_detection
from face_services.processors.wav2lip.models import Wav2Lip
from pkg_resources import resource_filename
from face_services.logger import logger
from face_services.jobs_database import jobs_database

class W2l:
    def __init__(self, face, audio, checkpoint, nosmooth, 
                 resize_factor, pad_top, pad_bottom, pad_left, pad_right, 
                 face_swap_img, output_folder, id):
        self.wav2lip_folder = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
        self.static = False
        # if os.path.isfile(face) and face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        #     self.static = True

        self.img_size = 96
        self.face = face
        self.audio = audio
        self.checkpoint = checkpoint
        self.mel_step_size = 16
        self.face_det_batch_size = 1

        self.id = id

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        logger.info('VisualDubber {} - Using {} for inference'.format(self.id, self.device))

        self.pads = [pad_top, pad_bottom, pad_left, pad_right]
        self.face_swap_img = face_swap_img
        self.nosmooth = nosmooth
        self.box = [-1, -1, -1, -1]
        # self.box = [56, 174, 470, 563]


        
        self.wav2lip_batch_size = 128
        self.fps = 25
        self.resize_factor = resize_factor
        self.rotate = False
        self.crop = [0, -1, 0, -1]
        self.checkpoint_path = os.path.join(os.getcwd(), 'face_services', 'models', 'visual_dubber', self.checkpoint + '.pth')
        self.outfile = output_folder + '/output/result_voice.mp4'

        self.ffmpeg_binary = self.find_ffmpeg_binary()
        self.output_folder = output_folder


        # self.status = 
        self.face_enhancer = FaceEnhancer()

    def find_ffmpeg_binary(self):
        for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
            try:
                package_path = resource_filename(package, 'binaries')
                files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0] if files else 'ffmpeg'
            except:
                return 'ffmpeg'

    def execute_command(self, command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr)

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):
        
        start_time = time.time()

        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=self.device)

        batch_size = self.face_det_batch_size

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        n = 0
        for rect, image in zip(predictions, images):
            if rect is None:
                print("Hum : " + str(n))
                cv2.imwrite(self.output_folder + '/debug/faulty_frame.jpg',
                            image)  # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])
            n += 1

        boxes = np.array(results)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector

        print("--- %s seconds ---" % (time.time() - start_time))

        return results

    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self.face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            # cv2.namedWindow('face', 0)
            # cv2.imshow('face', face)
            # cv2.waitKey(1)

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _load(self, checkpoint_path):
        # shared.cmd_opts.disable_safe_unpickle = True
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='mps')
        # shared.cmd_opts.disable_safe_unpickle = False
        return checkpoint

    def load_model(self, path):
        model = Wav2Lip()
        # print("Load checkpoint from: {}".format(path))
        checkpoint = self._load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def execute(self):
        if not os.path.isfile(self.face):
            raise ValueError('--face argument must be a valid path to video/image file')
        elif '.' in self.face and self.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(self.face)]
            fps = self.fps
        else:
            video_stream = cv2.VideoCapture(self.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            logger.info('VisualDubber {} - Reading video frames...'.format(self.id))

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1 and self.face_swap_img is None:
                    frame = cv2.resize(frame,
                                       (frame.shape[1] // self.resize_factor, frame.shape[0] // self.resize_factor))

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.crop
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        logger.info('VisualDubber {} - Number of frames available for inference: {}'.format(self.id, len(full_frames)))



        wav = audio.load_wav(self.audio, 16000)
        mel = audio.melspectrogram(wav)
    


        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        logger.info('VisualDubber {} - Length of mel chunks: {}'.format(self.id, len(mel_chunks)))


        full_frames = full_frames[:len(mel_chunks)]

        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            
            # print('batch : ', i)
            int(np.ceil(float(len(mel_chunks)) / batch_size))
            


            if i == 0:
                model = self.load_model(self.checkpoint_path)
                logger.info('VisualDubber {} - Model loaded'.format(self.id))

                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(self.output_folder + '/output/result.avi',
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))




            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for j, (p, f, c) in enumerate(zip(pred, frames, coords)):

                print('frame : ', j)

                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                # p = self.face_enhancer.run(p, blend_percentage=50)
                # out.write(enhanced_frame)

                f[y1:y2, x1:x2] = p

                out.write(f)

                # cv2.imwrite('test.png', p)
                # cv2.imwrite('test.png', f)

                jobs_database[self.id]['progress'] = np.round((j+i*self.wav2lip_batch_size)/len(full_frames), 2)

                # enhanced_frame = self.face_enhancer.run(f, model='codeformer', 
                #                                         blend_percentage=50)
                # out.write(enhanced_frame)


                # cv2.namedWindow('f', 0)
                # cv2.imshow('f', f)
                # cv2.waitKey(1)

        out.release()
        # release memory
        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

        command = [self.ffmpeg_binary, "-analyzeduration", "2147483647", 
                   "-probesize", "2147483647", 
                   "-y", "-i", self.audio, "-i", self.output_folder + '/output/result.avi',
                   "-strict", "-2", "-q:v", "1", self.outfile]
        self.execute_command(command)
