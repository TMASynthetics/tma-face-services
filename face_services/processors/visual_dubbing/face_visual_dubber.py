from ast import List
import uuid
import numpy as np
import gc
from tqdm import tqdm
import torch
import cv2
import os
from face_services.processors.visual_dubbing.models.wav2lip import Wav2Lip
from face_services.logger import logger
from face_services.processors.utilities import Timer
from face_services.components.audio import Audio
from face_services.components.video import Video
from face_services.processors.face_detector import FaceDetector
from face_services.processors.face_enhancer import FaceEnhancer
from face_services.models_list import VISUAL_DUBBER_MODELS


class FaceVisualDubber:
  
    def __init__(self, video_path: str, audio_paths: List, model_name: str=None) -> None:

        self.id = str(uuid.uuid1())
        logger.debug('FaceVisualDubber {} - Initialize'.format(self.id))
          
        self.video = Video(video_path)
        self.audios = [Audio(path=audio_path, sample_rate=16e3) for audio_path in audio_paths]

        self.fps = self.video.fps

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        if model_name is None or model_name not in self.get_available_models():
            model_name = self.get_available_models()[0]
        logger.debug('VisualDubber {} - Current model is : {}'.format(self.id, model_name))
        self.checkpoint_path = VISUAL_DUBBER_MODELS[model_name]['path']
        self.model = self.load_model()

        self.mel_step_size = 16
        self.batch_size = 128
        self.img_size = 96

        # self.face_enhancer = FaceEnhancer()
        self.face_detector = FaceDetector()

        self.frames = []
        self.bboxes = []
        self.face_batch = []

        self.output_folder = os.path.join(os.getcwd(), 'face_services', 'processors', 'temp', self.id)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        for folder in ['frames', 'faces', 'audio', 'output', 'debug', 'faces_enhanced', 'frames_processed']:
            if not os.path.exists(os.path.join(self.output_folder, folder)):
                os.makedirs(os.path.join(self.output_folder, folder))

        self.dubbed_videos_paths = []

        self.output_videos = {}
        for audio in self.audios:
            self.output_videos[audio.name] = cv2.VideoWriter(os.path.join(self.output_folder, 'output', audio.name + '_result.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'), self.video.fps, (self.video.width, self.video.height))
            audio.mel_chunks = self.extract_melspectrogram(audio)

    @Timer(name="load_model")
    def load_model(self):
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

        for batch_idx in tqdm(range(max(self.video.frame_number, len(self.audios[0].mel_chunks)) // self.batch_size + 1)):

            if self.video.get_current_frame_position() < self.video.frame_number:
                self.extract_faces()

            for idx, audio in enumerate(self.audios):

                if batch_idx*self.batch_size+self.batch_size < len(audio.mel_chunks):
                    audio.current_mel_chunk = audio.mel_chunks[batch_idx*self.batch_size:batch_idx*self.batch_size+self.batch_size] 
                else:
                    audio.current_mel_chunk = audio.mel_chunks[batch_idx*self.batch_size:] 

                mel_batch = self.prepare_mel_batch(audio)

                if idx == 0:
                    self.frames = self.frames[:len(audio.current_mel_chunk)]
                    self.bboxes = self.bboxes[:len(audio.current_mel_chunk)] 
                    self.face_batch = self.face_batch[:len(audio.current_mel_chunk)] 

                self.inference(self.face_batch, mel_batch, audio)

        self.clean_and_close()

        return self.dubbed_videos_paths

    @Timer(name="extract_faces")
    def extract_faces(self):
        self.frames = []
        self.bboxes = []
        for _ in range(self.batch_size):
            frame = self.video.get_frame()
            if frame is not None:
                self.frames.append(frame)
                bbox = list(map(int, self.face_detector.run(frame)[0].bbox))
                bbox[3] += 20
                self.bboxes.append(bbox)
            else:
                self.frames.append(self.frames[-1])
                self.bboxes.append(self.bboxes[-1])
        self.prepare_face_batch()

    @Timer(name="prepare_face_batch")
    def prepare_face_batch(self):
        self.face_batch = np.asarray([cv2.resize(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], 
                                            (self.img_size, self.img_size)) for frame, bbox in 
                                            zip(self.frames, self.bboxes)])
        img_masked = self.face_batch.copy()
        img_masked[:, self.img_size // 2:] = 0
        self.face_batch = np.concatenate((img_masked, self.face_batch), axis=3) / 255.

    @Timer(name="prepare_mel_batch")
    def prepare_mel_batch(self, audio):
        mel_batch = np.asarray(audio.current_mel_chunk)
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        return mel_batch
    
    @Timer(name="inference")
    def inference(self, img_batch, mel_batch, audio):

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

        with torch.no_grad():
            pred = self.model(mel_batch, img_batch)

        predictions = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for _, (prediction, frame, bbox) in enumerate(zip(predictions, self.frames, self.bboxes)):
            output_frame = frame.copy()
            x1, y1, x2, y2 = bbox
            prediction = cv2.resize(prediction.astype(np.uint8), (x2 - x1, y2 - y1))
            output_frame[y1:y2, x1:x2] = prediction
            self.output_videos[audio.name].write(output_frame)

        return None

    @Timer(name="extract_melspectrogram")
    def extract_melspectrogram(self, audio):

        mel_idx_multiplier = 80. / self.fps
        i = 0
        mel_chunks = []
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(audio.mel[0]):
                mel_chunks.append(audio.mel[:, len(audio.mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(audio.mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        return mel_chunks

    @Timer(name="clean_and_close")
    def clean_and_close(self):

        if not os.path.exists(os.path.join('outputs', self.id)):
            os.makedirs(os.path.join('outputs', self.id))

        for audio in self.audios:
            self.output_videos[audio.name].release()
            output_path = os.path.join('outputs', self.id, self.video.name + '_' + audio.name + '.mp4')
            Video.add_audio_to_video(os.path.join(self.output_folder, 'output', audio.name + '_result.mp4'), 
                                     audio.path, output_path)
            self.dubbed_videos_paths.append(output_path)
        
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def get_available_models():
        return ['wav2lip', 'wav2lip_gan']
    

