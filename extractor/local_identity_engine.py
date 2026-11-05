"""Resident universal WavLM engine; reference-conditioned heads stay task-local."""
import torch


class LocalIdentityEngine:
    _local_identity_engine = True

    def __init__(self, device, directory):
        from transformers import Wav2Vec2FeatureExtractor,WavLMForXVector
        from .model_store import active_speaker_path
        self.device=torch.device(device)
        self.model=WavLMForXVector.from_pretrained(str(directory),local_files_only=True,
            use_safetensors=active_speaker_path(directory).suffix=='.safetensors').to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor=Wav2Vec2FeatureExtractor.from_pretrained(str(directory),local_files_only=True)

    def create_classifier(self, paths, **kwargs):
        from .local_identity_classifier import reference_conditioned_classifier
        return reference_conditioned_classifier(self.model,self.processor,self.device,paths,**kwargs)

    def score(self, classifier, wave):
        from .local_identity_classifier import pooled_features
        classifier.to(self.device)
        return float(classifier(pooled_features(self.model,self.processor,self.device,wave)).item())

    def close(self):
        self.model.to('cpu')
