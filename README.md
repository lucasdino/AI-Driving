# AI-Driving

## About the Project
I documented this project on my [Notion](https://www.notion.so/Q-Learning-214e45ce19704294bd365308a77c68b4) in a bit more detail.

---

Built a simple python game to achieve the following objectives:
1. Reacquaint myself with python and engage in programming larger projects
2. Train a car to drive on a track with some level of generalizability across unseen tracks

---

Learned a lot and got satisfactory results. Few highlights:
1. The car could complete the course it was trained on reasonably often. On unseen maps that were of similar difficulty it performed well, on harder maps it did not 🫠
    -   MORE ON THE NOTION...but there are several improvements I could make to it that I think would be low hanging fruit. Just need to move on from here
2. After building the game I went through and optimized algorithms / processes to see a 3x improvement in FPS during training (from ~40 to ~120 FPS)
3. Refactored the code into a state that I'm proud of. Again, few changes I see that are clear and if I built it from scratch again I would approach differently in areas

---

Few Gifs!
![Trained model running on the original track](https://github.com/lucasdino/AI-Driving/blob/main/gifs/OriginalTrainingTrack.gif)
![Trained model running on unseen, similar but slightly more difficult track](https://github.com/lucasdino/AI-Driving/blob/main/gifs/HardCartoonTrack.gif)
![Trained model running on harder Google Maps track](https://github.com/lucasdino/AI-Driving/blob/main/gifs/GoogleMapsTrack.gif)
![Trained model not doing so hot on DALLE Generated difficult track](https://github.com/lucasdino/AI-Driving/blob/main/gifs/DALLEUnderwaterTrack.gif)

## Setup Instructions

### Creating the Environment
To set up the project environment, run the following command in the project directory (where `environment.yml` is located):

```bash
conda env create -f environment.yml
```

### Updating Settings
The settings are currently set such that you run the trained model. If you want to train your own model or play as a human, update these settings in `/utility/gamesettings.py`. Update these settings to change the map, draw your own lines, or train a model. 