# Self-Driving Car in a Simulator

I taught a car to drive itself around a track in the [Donkey Car simulator](https://github.com/tawnkramer/gym-donkeycar).
I drove it by hand and recorded what the car's camera saw. then trained a neural network to copy me. now it drives on its own.

I started this as a complete beginner, following a write-up by Hari ([@harixganesh](https://instagram.com/harixganesh)) on *behavioral cloning*:
learning a skill by copying examples instead of being given rules.

**Result: 49 of 50 laps completed without leaving the track, best lap 10.51 s.**

![Training loss](assets/loss.png)

<!-- Drag your lap video into this file on github.com while editing and it will appear here. -->

## How it works

1. **Record.** For every camera picture, the simulator saves the steering value, those pairs (picture, steering) are the training data.
2. **Prepare the pictures.** shrink each picture to 66 x 200 pixels and convert it to YUV colour.
   I hold back 20% of the data to check the model on pictures it has never trained on. I also flip pictures left to right
   and use fewer straight-driving pictures so the network doesn't just learn to go straight.
3. **Train.** The network is the one from Nvidia's 2016 paper, "End to End Learning for Self-Driving Cars":
   5 convolutional layers and 3 dense layers, about 252,000 weights. It learns by guessing the steering, measuring the miss
   (mean squared error), and nudging its weights with the Adam optimiser. Dropout helps stop it memorising the pictures.
4. **Drive.** The network looks at each camera frame and decides the steering. A simple rule sets the speed.
5. **Fix the failures.** This took most of the time. Each time the car left the road, I recorded more driving at that spot, retrained and tried again.

## What happened

My first model couldn't even take the first turn, even after about 40 tries. After more recording it started finishing about 70% of its laps.
The big improvement was recording more driving at the turns where it failed, especially *recovery* driving:
letting the car drift towards the edge and then steering back to the middle. Without that, the network never learns what to do after it makes a small mistake.
After more rounds of that, it reached 49 of 50 laps.

| Model | Laps completed |
|-------|----------------|
| First version | Couldn't pass the first turn |
| After more data | about 70% |
| Best model | 49 / 50 complete successful laps |

Setup was also part of the journey. The old Udacity simulator wouldn't start on my PC, so I switched to the Donkey simulator.
TensorFlow doesn't support the newest Python, so I had to use Python 3.11 in a separate environment.

## Run it yourself
You need **Python 3.11** (TensorFlow 2.15 doesn't work with newer versions).
```
python -m venv venv
venv\Scripts\activate
pip install numpy"<2" pandas pillow opencv-python-headless scikit-learn matplotlib pygame-ce tensorflow==2.15.1
pip install https://github.com/tawnkramer/gym-donkeycar/archive/refs/heads/master.zip
```
Download the Windows simulator from the [gym-donkeycar releases](https://github.com/tawnkramer/gym-donkeycar/releases), unzip it, then:
```
python step1_connection_test.py --sim-path "PATH\TO\donkey_sim.exe"
python step2_record.py --out my_recording --sim-path "PATH\TO\donkey_sim.exe"
python step3_check_data.py --data my_recording
python step4_train.py --data my_recording --epochs 15 --out model_v1.keras
python step5_drive.py models/model_best.keras --sim-path "PATH\TO\donkey_sim.exe"
```
My recordings are not in this repo because they are thousands of pictures.

## Credits

- Hari ([@harixganesh](https://twitter.com/harixganesh)), whose write-up this project follows
- Nvidia, [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) (Bojarski et al., 2016)
- [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar) and the Donkey Car simulator
- I built this with help from Claude (Anthropic) as a learning project.
