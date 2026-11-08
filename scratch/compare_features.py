import numpy as np

daisee = np.load('data/processed/fusion_daisee_v3_hardened.npz')['features']
clip004 = np.load('data/processed/classroom_clip_004_sequences.npz')['features']

descriptions = [
    'Head Pose Yaw [-1, 1]', 'Head Pose Pitch [-1, 1]', 'Head Pose Roll [-1, 1]',
    'Gaze Yaw [-1, 1]', 'Gaze Pitch [-1, 1]',
    'Emotion: Angry', 'Emotion: Disgust', 'Emotion: Fear', 'Emotion: Happy',
    'Emotion: Sad', 'Emotion: Surprise', 'Emotion: Neutral',
    'SCB: Hand-Raising', 'SCB: Reading', 'SCB: Writing', 'SCB: Using Phone',
    'SCB: Bowing Head', 'SCB: Leaning Over Table',
    'Optical Motion', 'Eye Aspect Ratio', 'Mouth Aspect Ratio',
    'Facing-Front Score', 'Torso Lean Angle', 'Upper-Body Confidence',
    'Reserved Padding 24', 'Reserved Padding 25', 'Reserved Padding 26',
    'Reserved Padding 27', 'Reserved Padding 28', 'Reserved Padding 29',
    'Reserved Padding 30', 'Reserved Padding 31'
]

print(f"{'Ch':<4} | {'DAiSEE Mean':<12} | {'Clip004 Mean':<12} | {'DAiSEE NonZero':<14} | {'Clip004 NonZero':<14} | Description")
print('-' * 95)

for ch in range(32):
    d_m = float(daisee[:, :, ch].mean())
    c_m = float(clip004[:, :, ch].mean())
    d_nz = float((daisee[:, :, ch] != 0).sum()) / daisee[:, :, ch].size * 100.0
    c_nz = float((clip004[:, :, ch] != 0).sum()) / clip004[:, :, ch].size * 100.0
    desc = descriptions[ch] if ch < len(descriptions) else f'Channel {ch}'
    print(f"{ch:<4} | {d_m:<12.4f} | {c_m:<12.4f} | {d_nz:<13.1f}% | {c_nz:<13.1f}% | {desc}")
