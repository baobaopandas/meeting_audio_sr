# 1. visit hf.co/pyannote/speaker-diarization and hf.co/pyannote/segmentation and accept user conditions (only if requested)
# 2. visit hf.co/settings/tokens to create an access token (only if you had to go through 1.)
# 3. instantiate pretrained speaker diarization pipeline
from pyannote.audio import Pipeline

class DiarizationResult:
    def __init__(self, start_time, end_time, speaker):
        self.start_time = start_time
        self.end_time = end_time
        self.speaker = speaker


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="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")


def getDiarization(file_path):
    diarization = pipeline(file_path)
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result = DiarizationResult(turn.start, turn.end, f"speaker_{speaker}")
        results.append(result)
    return results
# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...