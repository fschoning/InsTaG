# InsTaG - Franz's Grace-Blackwell (ARM64) Build Notes

This document tracks the specific commands, patches, and workflows required to successfully run the InsTaG 3D Gaussian Splatting model natively on an NVIDIA Grace-Blackwell (GB10) superchip using PyTorch 2.6 Nightly and CUDA 13.0.

## Core Architectural Patches Applied
* **C++17 Enforcement:** All `setup.py` and `CMakeLists.txt` files patched to `-std=c++17` for PyTorch 2.6 JIT JIT compatibility.
* **LPIPS Bounding Box:** `patch_lpips.py` applied to prevent 2x2 AlexNet matrix crashes.
* **Optimizer Sieve:** `gaussian_model.py` patched to safely transplant `[32, 74]` Face matrices into `[16, 68]` Mouth matrices without triggering strict shape crashing.
* **Motion Network Fix:** `scene/motion_net.py` patched to dynamically handle `NoneType` eye arrays during Phase 2 training.

## Execution Pipeline (With DeepSpeech Priors)

### Phase 1: Train Face
    python train_face.py -s data/juanita -m /tmp/output_juanita --port 6009 --long --pretrain_path pretrained_weights/pretrain_ds/chkpnt_ema_face_latest.pth

### Phase 2: Train Mouth
    python train_mouth.py -s data/juanita -m /tmp/output_juanita --long --pretrain_path pretrained_weights/pretrain_ds/chkpnt_ema_mouth_latest.pth

### Phase 3: Fuse Models
    python train_fuse_con.py -s data/juanita -m /tmp/output_juanita

## Phase 4: Custom Audio Inference
To puppet the 3D model with a custom voice, you must first inject your audio file from the host machine into the Docker container, then extract the acoustic features.

    # 1. Push the audio file from the HOST machine into the running container (Run on Host)
    docker cp "$HOME/instag/Juanita Cloning Reference Audio.m4a" instag_spark:/workspace/InsTaG/data/juanita/custom_audio.m4a

    # 2. Convert custom audio format (Run inside Container)
    ffmpeg -y -i data/juanita/custom_audio.m4a -ar 16000 -ac 1 data/juanita/custom_audio.wav

    # 3. Extract the DeepSpeech acoustic features (.npy)
    python data_utils/deepspeech_features/extract_ds_features.py --input data/juanita/custom_audio.wav

    # 4. Render the custom talking head video
    python synthesize_fuse.py -s data/juanita -m /tmp/output_juanita --use_train --audio data/juanita/custom_audio.npy

    # 5. Mux the audio back into the video
    ffmpeg -y -i /tmp/output_juanita/train/ours_None/renders/out.mp4 -i data/juanita/custom_audio.wav -c:v copy -c:a aac -strict experimental /tmp/output_juanita/train/ours_None/renders/Juanita_Premium_Final.mp4

## Phase 5: Extraction
Pull the final customized video OUT of the container.

    # Run on Host Machine
    docker cp instag_spark:/tmp/output_juanita/train/ours_None/renders/Juanita_Premium_Final.mp4 "$HOME/instag/Juanita_Premium_Final.mp4"
