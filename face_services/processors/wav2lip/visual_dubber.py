from ast import List
import time
import numpy as np
import gc
import cv2, os, face_services.processors.wav2lip.audio as audio_lib
from face_services.processors.utilities import Timer
from face_services.components.audio import Audio
from face_services.components.video import Video
from face_services.processors.face_detector import FaceDetector
from face_services.processors.face_enhancer import FaceEnhancer
import subprocess
from tqdm import tqdm
import torch
from face_services.processors.wav2lip.models.wav2lip import Wav2Lip
from pkg_resources import resource_filename
from face_services.logger import logger
from face_services.jobs_database import jobs_database
from face_services.logger import logger



class VisualDubber:
  
    def __init__(self, video_path: str, audio_paths: List, output_folder):

        self.video = Video(video_path)
        self.audios = [Audio(audio_path) for audio_path in audio_paths]

        self.fps = self.video.fps


        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.checkpoint_path = os.path.join(os.getcwd(), 'face_services', 'models', 'visual_dubber', 'wav2lip' + '.pth')
        self.model = self.load_model(self.checkpoint_path)

        self.mel_step_size = 16
        self.batch_size = 128
        self.img_size = 96

        self.face_enhancer = FaceEnhancer()
        self.face_detector = FaceDetector()

        self.frames = []
        self.bboxes = []
        self.mel_chunks = []
        self.generated_frames = []

        self.output_folder = output_folder


    @Timer(name="load_model")
    def load_model(self, path):
        model = Wav2Lip()

        if torch.cuda.is_available():
            checkpoint = torch.load(self.checkpoint_path)
        elif torch.backends.mps.is_available():
            checkpoint = torch.load(self.checkpoint_path, map_location='mps')
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()
    

    @Timer(name="run")
    def run(self):

        dubbed_videos = []

        audio = self.audios[0]

        logger.info('Dubbing {} video with {} audio'.format(self.video.name, audio.name))

        start = time.time()

        self.video._video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.output_video = cv2.VideoWriter(os.path.join(self.output_folder, 'output', 'result.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'), self.video.fps, (self.video.width, self.video.height))

        total_mel_chuncks = self.extract_melspectrogram(audio.path)

        for batch_idx in tqdm(range(self.video.frame_number // self.batch_size + 1)):

            self.frames = []
            self.bboxes = []

            if batch_idx*self.batch_size+self.batch_size < len(total_mel_chuncks):
                self.mel_chunks = total_mel_chuncks[batch_idx*self.batch_size:batch_idx*self.batch_size+self.batch_size] 
            else:
                self.mel_chunks = total_mel_chuncks[batch_idx*self.batch_size:] 

            self.generated_frames = []

            self.extract_frames()

            self.detect_faces()

            self.frames = self.frames[:len(self.mel_chunks)]
            self.bboxes = self.bboxes[:len(self.mel_chunks)]

            mel_batch, face_batch, = self.prepare_data()

            self.inference(face_batch, mel_batch)


        self.clean()

        Video.add_audio_to_video(os.path.join(self.output_folder, 'output', 'result.mp4'), 
                                                audio.path, os.path.join(self.output_folder, 'output', self.video.name + '_' + audio.name + '.mp4'))

        dubbed_videos.append(os.path.join(self.output_folder, 'output', self.video.name + '_' + audio.name + '.mp4'))

        print(time.time() - start)

        return dubbed_videos

    @Timer(name="extract_frames")
    def extract_frames(self):
        for _ in range(self.batch_size):
            frame = self.video.get_frame()
            if frame is not None:
                self.frames.append(frame)

    @Timer(name="detect_faces")
    def detect_faces(self):
        for frame in self.frames:
            bbox = list(map(int, self.face_detector.run(frame)[0].bbox))
            bbox[3] += 20
            self.bboxes.append(bbox)

    @Timer(name="prepare_data")
    def prepare_data(self):
        face_batch, mel_batch = [], []

        for i, mel in enumerate(self.mel_chunks):
            idx = i % len(self.frames)
            frame = self.frames[idx]
            bbox = self.bboxes[idx]

            face = cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], (self.img_size, self.img_size))
            face_batch.append(face)

            mel_batch.append(mel)

        if len(face_batch) > 0:
            face_batch = np.asarray(face_batch)
            img_masked = face_batch.copy()
            img_masked[:, self.img_size // 2:] = 0
            face_batch = np.concatenate((img_masked, face_batch), axis=3) / 255.

            mel_batch = np.asarray(mel_batch)
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        return mel_batch, face_batch

    @Timer(name="inference")
    def inference(self, img_batch, mel_batch):

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

        with torch.no_grad():
            pred = self.model(mel_batch, img_batch)

        predictions = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        # generated_frames = []

        for frame_idx, (prediction, frame, bbox) in enumerate(zip(predictions, self.frames, self.bboxes)):
            x1, y1, x2, y2 = bbox
            prediction = cv2.resize(prediction.astype(np.uint8), (x2 - x1, y2 - y1))
            frame[y1:y2, x1:x2] = prediction

            # generated_frames.append(frame)
            # cv2.namedWindow('frame', 0)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)

            self.output_video.write(frame)
            
        return None

    @Timer(name="extract_melspectrogram")
    def extract_melspectrogram(self, audio):

        wav = audio_lib.load_wav(audio, 16000)
        mel = audio_lib.melspectrogram(wav)

        mel_idx_multiplier = 80. / self.fps
        i = 0
        mel_chunks = []
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        return mel_chunks

    @Timer(name="clean")
    def clean(self):
        self.output_video.release()
        torch.cuda.empty_cache()
        gc.collect()